"""Roofline helpers for speculative target-verify K_hw calibration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TargetModelShape:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_hidden_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _get_config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def target_model_shape_from_config(config: Any) -> TargetModelShape:
    hidden_size = int(_get_config_value(config, "hidden_size"))
    num_attention_heads = int(_get_config_value(config, "num_attention_heads"))
    num_key_value_heads = int(
        _get_config_value(config, "num_key_value_heads", num_attention_heads)
    )
    return TargetModelShape(
        num_hidden_layers=int(_get_config_value(config, "num_hidden_layers")),
        hidden_size=hidden_size,
        intermediate_size=int(_get_config_value(config, "intermediate_size")),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )


def estimate_dense_weight_elements(shape: TargetModelShape) -> int:
    hidden = shape.hidden_size
    intermediate = shape.intermediate_size
    kv_hidden = shape.kv_hidden_size
    qkv = hidden * (hidden + 2 * kv_hidden)
    output = hidden * hidden
    mlp = 3 * hidden * intermediate
    return shape.num_hidden_layers * (qkv + output + mlp)


def estimate_dense_weight_bytes(
    shape: TargetModelShape,
    bytes_per_weight: float,
) -> float:
    return estimate_dense_weight_elements(shape) * bytes_per_weight


def estimate_k_roof(
    *,
    peak_tflops: float,
    memory_bw_gbps: float,
    bytes_per_weight: float,
) -> float:
    """Dense-layer roofline K where 2K/bytes reaches peak FLOPS / memory BW."""
    if peak_tflops <= 0:
        raise ValueError("peak_tflops must be positive")
    if memory_bw_gbps <= 0:
        raise ValueError("memory_bw_gbps must be positive")
    if bytes_per_weight <= 0:
        raise ValueError("bytes_per_weight must be positive")
    peak_flops = peak_tflops * 1e12
    memory_bw = memory_bw_gbps * 1e9
    return (bytes_per_weight / 2.0) * (peak_flops / memory_bw)


def draft_lens_from_roofline(
    *,
    batch_size: int,
    k_roof: float,
    multipliers: list[float],
    max_draft_len: int,
) -> list[int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if k_roof <= 0:
        raise ValueError("k_roof must be positive")
    if max_draft_len <= 0:
        raise ValueError("max_draft_len must be positive")

    draft_lens = set()
    for multiplier in multipliers:
        if multiplier <= 0:
            continue
        candidate_k = k_roof * multiplier
        draft_lens.add(max(1, min(max_draft_len, math.ceil(candidate_k / batch_size))))
    return sorted(draft_lens)


def estimate_target_verify_flops(
    *,
    shape: TargetModelShape,
    batch_size: int,
    draft_token_num: int,
    seq_lens_sum: int | None,
) -> dict[str, float]:
    """Estimate verification FLOPs for QKV/O/MLP plus a coarse attention term."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if draft_token_num <= 0:
        raise ValueError("draft_token_num must be positive")

    k = batch_size * draft_token_num
    hidden = shape.hidden_size
    intermediate = shape.intermediate_size
    kv_hidden = shape.kv_hidden_size

    qkv_flops = shape.num_hidden_layers * 2.0 * k * hidden * (hidden + 2 * kv_hidden)
    output_flops = shape.num_hidden_layers * 2.0 * k * hidden * hidden
    gate_up_flops = shape.num_hidden_layers * 4.0 * k * hidden * intermediate
    gate_down_flops = shape.num_hidden_layers * 2.0 * k * intermediate * hidden

    avg_seq_len = 0.0 if seq_lens_sum is None else max(0.0, seq_lens_sum / batch_size)
    attention_flops = shape.num_hidden_layers * 4.0 * k * avg_seq_len * hidden

    dense_flops = qkv_flops + output_flops + gate_up_flops + gate_down_flops
    total_flops = dense_flops + attention_flops
    return {
        "qkv_flops": qkv_flops,
        "output_flops": output_flops,
        "gate_up_flops": gate_up_flops,
        "gate_down_flops": gate_down_flops,
        "attention_flops": attention_flops,
        "dense_flops": dense_flops,
        "total_flops": total_flops,
        "avg_seq_len": avg_seq_len,
    }


def achieved_tflops(flops: float, latency_ms: float) -> float | None:
    if latency_ms <= 0:
        return None
    return flops / (latency_ms / 1000.0) / 1e12
