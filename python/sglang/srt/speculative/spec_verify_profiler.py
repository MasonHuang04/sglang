"""Debug profiler for speculative target verification.

This module is intentionally lightweight and env-gated.  It writes one JSONL
record per target-verify forward so K-hardware sweeps can compute
``num_verify_tokens / latency`` without parsing server logs.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_PROFILE_LOCK = threading.Lock()


def is_spec_verify_profile_enabled() -> bool:
    return bool(envs.SGLANG_DEBUG_SPEC_VERIFY_PROFILE.get())


def _resolve_profile_path(tp_rank: int | None, dp_rank: int | None) -> str:
    path = envs.SGLANG_DEBUG_SPEC_VERIFY_PROFILE_PATH.get()
    if not path:
        path = "/tmp/sglang_spec_verify_profile.jsonl"

    return path.format(
        pid=os.getpid(),
        tp_rank=0 if tp_rank is None else tp_rank,
        dp_rank=0 if dp_rank is None else dp_rank,
    )


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _safe_tensor_sum(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value.sum().item())
    except Exception:
        return None


def _safe_tensor_max(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if value.numel() == 0:
            return 0
        return int(value.max().item())
    except Exception:
        return None


def _write_profile_record(
    record: dict[str, Any],
    tp_rank: int | None,
    dp_rank: int | None,
) -> None:
    path = _resolve_profile_path(tp_rank, dp_rank)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with _PROFILE_LOCK:
            with open(path, "a", encoding="utf-8") as fout:
                fout.write(line)
                fout.write("\n")
    except Exception:
        logger.exception("Failed to write speculative verify profile record.")


def run_with_spec_verify_profile(
    forward_fn: Callable[[], Any],
    *,
    algorithm: str,
    batch_size: int,
    draft_token_num: int,
    can_run_cuda_graph: bool,
    device: torch.device | str,
    seq_lens: torch.Tensor | None = None,
    seq_lens_sum: int | None = None,
    tp_rank: int | None = None,
    dp_rank: int | None = None,
    attention_backend: str | None = None,
) -> Any:
    """Run ``forward_fn`` and optionally append a target-verify profile record."""
    if not is_spec_verify_profile_enabled():
        return forward_fn()

    start_event = None
    end_event = None
    device_module = None
    start_time = time.perf_counter()
    try:
        device_module = torch.get_device_module(device)
        start_event = device_module.Event(enable_timing=True)
        end_event = device_module.Event(enable_timing=True)
        start_event.record()
    except Exception:
        start_event = None
        end_event = None

    result = forward_fn()

    latency_ms: float
    if start_event is not None and end_event is not None:
        end_event.record()
        end_event.synchronize()
        latency_ms = float(start_event.elapsed_time(end_event))
    else:
        if device_module is not None and hasattr(device_module, "synchronize"):
            try:
                device_module.synchronize()
            except Exception:
                pass
        latency_ms = (time.perf_counter() - start_time) * 1000.0

    batch_size = int(batch_size)
    draft_token_num = int(draft_token_num)
    num_verify_tokens = batch_size * draft_token_num
    # DFlash/EAGLE chain settings use one current-token row plus draft rows.
    # Keep both counts because K_hw should be plotted against target-forward
    # rows, while accept-rate denominators use proposed drafts.
    num_proposed_drafts = batch_size * max(draft_token_num - 1, 0)
    seq_lens_sum_value = _safe_int(seq_lens_sum)
    if seq_lens_sum_value is None:
        seq_lens_sum_value = _safe_tensor_sum(seq_lens)

    record = {
        "time": time.time(),
        "pid": os.getpid(),
        "algorithm": algorithm,
        "batch_size": batch_size,
        "draft_token_num": draft_token_num,
        "num_verify_tokens": num_verify_tokens,
        "num_proposed_drafts": num_proposed_drafts,
        "latency_ms": latency_ms,
        "verify_tokens_per_s": (
            num_verify_tokens / (latency_ms / 1000.0) if latency_ms > 0 else None
        ),
        "proposed_drafts_per_s": (
            num_proposed_drafts / (latency_ms / 1000.0)
            if latency_ms > 0
            else None
        ),
        "can_run_cuda_graph": bool(can_run_cuda_graph),
        "seq_lens_sum": seq_lens_sum_value,
        "seq_lens_max": _safe_tensor_max(seq_lens),
        "tp_rank": _safe_int(tp_rank),
        "dp_rank": _safe_int(dp_rank),
        "device": str(device),
        "attention_backend": attention_backend,
    }
    _write_profile_record(record, tp_rank, dp_rank)
    return result
