"""Roofline-guided K_hw calibration for speculative target verification.

The workflow is:

1. Estimate a dense-layer roofline knee K_roof for the target model and GPU.
2. Generate a small set of draft lengths around K_roof for each batch size.
3. Launch one speculative server per candidate, profile target verification,
   and select the smallest K that reaches the profiled hardware-efficient region.

The output files are written under --output-dir:
- summary.jsonl: one row per measured (batch_size, draft_token_num).
- calibration.json: roofline metadata, all measurements, and selected K_hw values.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import statistics
from pathlib import Path
from typing import Any

import requests

from sglang.srt.speculative.k_hw_roofline import (
    TargetModelShape,
    achieved_tflops,
    draft_lens_from_roofline,
    estimate_dense_weight_bytes,
    estimate_k_roof,
    estimate_target_verify_flops,
    target_model_shape_from_config,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)


PROMPTS = [
    "Write a detailed technical note about cache locality in GPU inference.",
    "List practical steps for debugging a distributed inference timeout.",
    "Explain why batching changes arithmetic intensity in transformer decode.",
    "Draft a concise incident report for a slow model serving deployment.",
]

RTX_6000_ADA_PEAK_TFLOPS_FP16 = 362.1
RTX_6000_ADA_MEMORY_BW_GBPS = 960.0
DEFAULT_ROOFLINE_MULTIPLIERS = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0]


def parse_int_list(value: str) -> list[int]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def parse_float_list(value: str) -> list[float]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one float")
    return out


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    try:
        with open(path, encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        pass
    return records


def read_profile_records(profile_glob: str) -> dict[str, list[dict[str, Any]]]:
    return {path: read_jsonl(path) for path in sorted(glob.glob(profile_glob))}


def load_target_model_shape(model_path: str) -> tuple[TargetModelShape, dict[str, Any]]:
    expanded = Path(model_path).expanduser()
    config_path = expanded / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as fin:
            config = json.load(fin)
        return target_model_shape_from_config(config), {
            "source": str(config_path),
            "model_type": config.get("model_type"),
        }

    try:
        from transformers import AutoConfig

        config_obj = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load model config for {model_path!r}; pass a local path "
            "with config.json or install transformers."
        ) from exc

    return target_model_shape_from_config(config_obj), {
        "source": model_path,
        "model_type": getattr(config_obj, "model_type", None),
    }


def build_sampling_params(batch_size: int, max_tokens: int) -> list[dict[str, Any]]:
    return [
        {
            "temperature": 0,
            "max_new_tokens": max_tokens,
            "ignore_eos": True,
        }
        for _ in range(batch_size)
    ]


def post_batched_generate(base_url: str, batch_size: int, max_tokens: int) -> None:
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch_size)]
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": prompts,
            "sampling_params": build_sampling_params(batch_size, max_tokens),
            "return_logprob": False,
        },
        timeout=max(300, max_tokens * 20),
    )
    response.raise_for_status()


def cuda_graph_args(batch_size: int, disable_cuda_graph: bool) -> list[str]:
    if disable_cuda_graph:
        return ["--disable-cuda-graph"]
    return ["--cuda-graph-bs-decode", str(batch_size)]


def speculative_args(
    args: argparse.Namespace,
    draft_len: int,
) -> list[str]:
    result = [
        "--speculative-algorithm",
        args.speculative_algorithm,
        "--speculative-draft-model-path",
        args.draft_model,
        "--speculative-num-draft-tokens",
        str(draft_len),
    ]
    if args.speculative_algorithm in ("EAGLE", "EAGLE3"):
        num_steps = args.speculative_num_steps
        if num_steps is None:
            num_steps = max(0, draft_len - 1)
        result.extend(
            [
                "--speculative-num-steps",
                str(num_steps),
                "--speculative-eagle-topk",
                str(args.speculative_eagle_topk),
            ]
        )
    return result


def launch_args(
    args: argparse.Namespace,
    batch_size: int,
    draft_len: int,
) -> list[str]:
    result = [
        "--trust-remote-code",
        "--attention-backend",
        args.attention_backend,
        "--page-size",
        str(args.page_size),
        "--max-running-requests",
        str(batch_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
    ]
    if args.dtype:
        result.extend(["--dtype", args.dtype])
    result.extend(speculative_args(args, draft_len))
    result.extend(cuda_graph_args(batch_size, args.disable_cuda_graph))
    if args.disable_overlap_schedule:
        result.append("--disable-overlap-schedule")
    if args.skip_server_warmup:
        result.append("--skip-server-warmup")
    if not args.enable_piecewise_cuda_graph:
        result.extend(["--cuda-graph-backend-prefill", "disabled"])
    result.extend(shlex.split(args.extra_server_args))
    return result


def summarize_config(
    records: list[dict[str, Any]],
    *,
    model_shape: TargetModelShape,
    batch_size: int,
    draft_len: int,
    bytes_per_weight: float,
    profile_files: list[str],
) -> dict[str, Any]:
    records = [
        record
        for record in records
        if record.get("batch_size") == batch_size
        and record.get("draft_token_num") == draft_len
    ]
    latencies = [float(r["latency_ms"]) for r in records if r.get("latency_ms")]
    verify_tps = [
        float(r["verify_tokens_per_s"])
        for r in records
        if r.get("verify_tokens_per_s") is not None
    ]
    proposed_tps = [
        float(r["proposed_drafts_per_s"])
        for r in records
        if r.get("proposed_drafts_per_s") is not None
    ]
    graph_flags = [bool(r.get("can_run_cuda_graph")) for r in records]
    seq_lens_sum_values = [
        int(r["seq_lens_sum"]) for r in records if r.get("seq_lens_sum") is not None
    ]
    k_verify_tokens = batch_size * draft_len
    k_proposed_drafts = batch_size * max(draft_len - 1, 0)

    estimated_tflops = []
    dense_tflops = []
    for record in records:
        latency_ms = float(record["latency_ms"]) if record.get("latency_ms") else 0.0
        flops = estimate_target_verify_flops(
            shape=model_shape,
            batch_size=batch_size,
            draft_token_num=draft_len,
            seq_lens_sum=record.get("seq_lens_sum"),
        )
        total_tflops = achieved_tflops(flops["total_flops"], latency_ms)
        if total_tflops is not None:
            estimated_tflops.append(total_tflops)
        dense_only_tflops = achieved_tflops(flops["dense_flops"], latency_ms)
        if dense_only_tflops is not None:
            dense_tflops.append(dense_only_tflops)

    return {
        "batch_size": batch_size,
        "draft_token_num": draft_len,
        "k_verify_tokens": k_verify_tokens,
        "k_proposed_drafts": k_proposed_drafts,
        "arithmetic_intensity_dense": 2.0 * k_verify_tokens / bytes_per_weight,
        "num_records": len(records),
        "latency_ms_mean": mean_or_none(latencies),
        "latency_ms_p50": median_or_none(latencies),
        "latency_ms_p90": percentile(latencies, 0.90),
        "verify_tokens_per_s_mean": mean_or_none(verify_tps),
        "verify_tokens_per_s_max": max(verify_tps) if verify_tps else None,
        "proposed_drafts_per_s_mean": mean_or_none(proposed_tps),
        "estimated_total_tflops_mean": mean_or_none(estimated_tflops),
        "estimated_dense_tflops_mean": mean_or_none(dense_tflops),
        "seq_lens_sum_mean": mean_or_none(seq_lens_sum_values),
        "cuda_graph_rate": (
            sum(1 for flag in graph_flags if flag) / len(graph_flags)
            if graph_flags
            else None
        ),
        "profile_files": profile_files,
    }


def clean_profile_files(profile_glob: str) -> None:
    for path in glob.glob(profile_glob):
        os.remove(path)


def run_one_config(
    args: argparse.Namespace,
    model_shape: TargetModelShape,
    batch_size: int,
    draft_len: int,
    run_index: int,
) -> dict[str, Any]:
    port = find_available_port(args.base_port + run_index * 100)
    base_url = f"http://{args.host}:{port}"
    profile_name = f"spec_verify_B{batch_size}_L{draft_len}"
    profile_path = (
        args.output_dir / f"{profile_name}_tp{{tp_rank}}_dp{{dp_rank}}.jsonl"
    )
    profile_glob = str(args.output_dir / f"{profile_name}_tp*_dp*.jsonl")
    clean_profile_files(profile_glob)

    env = {
        "SGLANG_DEBUG_SPEC_VERIFY_PROFILE": "1",
        "SGLANG_DEBUG_SPEC_VERIFY_PROFILE_PATH": str(profile_path),
    }
    process = None
    try:
        process = popen_launch_server(
            args.model,
            base_url,
            timeout=args.launch_timeout,
            other_args=launch_args(args, batch_size, draft_len),
            env=env,
        )

        for _ in range(args.warmup_rounds):
            post_batched_generate(base_url, batch_size, args.max_tokens)

        warmup_counts = {
            path: len(records)
            for path, records in read_profile_records(profile_glob).items()
        }

        for _ in range(args.measure_rounds):
            post_batched_generate(base_url, batch_size, args.max_tokens)

        records_by_file = read_profile_records(profile_glob)
        measurement_records = []
        for path, records in records_by_file.items():
            measurement_records.extend(records[warmup_counts.get(path, 0) :])

        return summarize_config(
            measurement_records,
            model_shape=model_shape,
            batch_size=batch_size,
            draft_len=draft_len,
            bytes_per_weight=args.bytes_per_weight,
            profile_files=sorted(records_by_file),
        )
    finally:
        if process is not None:
            kill_process_tree(process.pid)


def print_summary_row(summary: dict[str, Any]) -> None:
    throughput = summary["verify_tokens_per_s_mean"]
    latency = summary["latency_ms_p50"]
    total_tflops = summary["estimated_total_tflops_mean"]
    graph_rate = summary["cuda_graph_rate"]
    throughput_text = (
        f"{throughput:12.1f}" if throughput is not None else "        None"
    )
    latency_text = f"{latency:8.3f}" if latency is not None else "    None"
    tflops_text = f"{total_tflops:8.1f}" if total_tflops is not None else "    None"
    graph_text = f"{graph_rate:6.2f}" if graph_rate is not None else "  None"
    print(
        f"{summary['batch_size']:>5} {summary['draft_token_num']:>5} "
        f"{summary['k_verify_tokens']:>8} {summary['num_records']:>7} "
        f"{latency_text} {throughput_text} {tflops_text} {graph_text}"
    )


def select_k_hw_for_batch(
    summaries: list[dict[str, Any]],
    *,
    rho: float,
    marginal_gain_epsilon: float,
) -> dict[str, Any] | None:
    valid = [
        summary
        for summary in summaries
        if summary.get("verify_tokens_per_s_mean") is not None
        and summary.get("latency_ms_p50") is not None
    ]
    valid.sort(key=lambda summary: summary["k_verify_tokens"])
    if not valid:
        return None

    best = max(summary["verify_tokens_per_s_mean"] for summary in valid)
    threshold = rho * best
    fallback = None
    for idx, summary in enumerate(valid):
        throughput = summary["verify_tokens_per_s_mean"]
        if throughput < threshold:
            continue
        if fallback is None:
            fallback = summary
        if idx == len(valid) - 1:
            return summary
        next_throughput = valid[idx + 1]["verify_tokens_per_s_mean"]
        marginal_gain = (next_throughput - throughput) / max(throughput, 1e-9)
        if marginal_gain <= marginal_gain_epsilon:
            result = dict(summary)
            result["next_relative_marginal_gain"] = marginal_gain
            return result
    return fallback


def resolve_run_grid(
    args: argparse.Namespace,
    *,
    k_roof: float,
) -> dict[int, list[int]]:
    if args.draft_lens != "auto":
        draft_lens = parse_int_list(args.draft_lens)
        return {batch_size: draft_lens for batch_size in args.batch_sizes}

    grid = {}
    extra_lens = parse_int_list(args.extra_draft_lens) if args.extra_draft_lens else []
    for batch_size in args.batch_sizes:
        draft_lens = draft_lens_from_roofline(
            batch_size=batch_size,
            k_roof=k_roof,
            multipliers=args.roofline_multipliers,
            max_draft_len=args.max_draft_len,
        )
        draft_lens = sorted(set(draft_lens + extra_lens))
        grid[batch_size] = [v for v in draft_lens if 1 <= v <= args.max_draft_len]
    return grid


def build_roofline_metadata(
    args: argparse.Namespace,
    model_shape: TargetModelShape,
    model_config_info: dict[str, Any],
    k_roof: float,
    run_grid: dict[int, list[int]],
) -> dict[str, Any]:
    dense_weight_bytes = estimate_dense_weight_bytes(
        model_shape, args.bytes_per_weight
    )
    return {
        "method": "roofline_guided_profile_calibrated_k_hw",
        "model": args.model,
        "draft_model": args.draft_model,
        "speculative_algorithm": args.speculative_algorithm,
        "model_config": model_config_info,
        "target_model_shape": model_shape.to_dict(),
        "hardware": {
            "peak_tflops": args.peak_tflops,
            "memory_bw_gbps": args.memory_bw_gbps,
            "bytes_per_weight": args.bytes_per_weight,
            "dense_weight_bytes": dense_weight_bytes,
        },
        "roofline": {
            "k_roof": k_roof,
            "multipliers": args.roofline_multipliers,
            "candidate_run_grid": run_grid,
        },
        "launch": {
            "attention_backend": args.attention_backend,
            "dtype": args.dtype,
            "page_size": args.page_size,
            "disable_cuda_graph": args.disable_cuda_graph,
            "enable_piecewise_cuda_graph": args.enable_piecewise_cuda_graph,
            "disable_overlap_schedule": args.disable_overlap_schedule,
            "extra_server_args": args.extra_server_args,
        },
        "selection": {
            "rho": args.rho,
            "marginal_gain_epsilon": args.marginal_gain_epsilon,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find K_hw for speculative target verification."
    )
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL_EAGLE3)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL_EAGLE3)
    parser.add_argument(
        "--speculative-algorithm",
        choices=["DFLASH", "EAGLE", "EAGLE3"],
        default="EAGLE3",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=30000)
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[32, 64, 128])
    parser.add_argument(
        "--draft-lens",
        default="auto",
        help="Comma-separated draft lengths, or 'auto' to use roofline candidates.",
    )
    parser.add_argument("--extra-draft-lens", default="")
    parser.add_argument("--max-draft-len", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--measure-rounds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/sglang_k_hw"))
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument(
        "--launch-timeout",
        type=float,
        default=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    )
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--enable-overlap-schedule", action="store_true")
    parser.add_argument("--enable-piecewise-cuda-graph", action="store_true")
    parser.add_argument("--skip-server-warmup", action="store_true")
    parser.add_argument("--speculative-num-steps", type=int)
    parser.add_argument("--speculative-eagle-topk", type=int, default=1)
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument("--peak-tflops", type=float, default=RTX_6000_ADA_PEAK_TFLOPS_FP16)
    parser.add_argument("--memory-bw-gbps", type=float, default=RTX_6000_ADA_MEMORY_BW_GBPS)
    parser.add_argument("--bytes-per-weight", type=float, default=2.0)
    parser.add_argument(
        "--roofline-multipliers",
        type=parse_float_list,
        default=DEFAULT_ROOFLINE_MULTIPLIERS,
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.90,
        help="K_hw threshold as a fraction of peak verify throughput.",
    )
    parser.add_argument(
        "--marginal-gain-epsilon",
        type=float,
        default=0.05,
        help="Stop at K when the next measured point improves throughput by <= epsilon.",
    )
    args = parser.parse_args()
    args.disable_overlap_schedule = not args.enable_overlap_schedule
    if args.speculative_algorithm == "DFLASH" and args.draft_model == DEFAULT_DRAFT_MODEL_EAGLE3:
        args.draft_model = DEFAULT_DRAFT_MODEL_DFLASH
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_shape, model_config_info = load_target_model_shape(args.model)
    k_roof = estimate_k_roof(
        peak_tflops=args.peak_tflops,
        memory_bw_gbps=args.memory_bw_gbps,
        bytes_per_weight=args.bytes_per_weight,
    )
    run_grid = resolve_run_grid(args, k_roof=k_roof)
    calibration = build_roofline_metadata(
        args, model_shape, model_config_info, k_roof, run_grid
    )

    summary_path = args.output_dir / "summary.jsonl"
    calibration_path = args.output_dir / "calibration.json"
    summaries = []
    run_index = 0
    with open(summary_path, "w", encoding="utf-8") as fout:
        print("    B     L        K records  p50_ms  verify_tok/s   TFLOPs  graph")
        for batch_size in args.batch_sizes:
            for draft_len in run_grid[batch_size]:
                summary = run_one_config(
                    args, model_shape, batch_size, draft_len, run_index
                )
                summaries.append(summary)
                fout.write(json.dumps(summary, sort_keys=True))
                fout.write("\n")
                fout.flush()
                print_summary_row(summary)
                run_index += 1

    k_hw_by_batch = {}
    for batch_size in args.batch_sizes:
        batch_summaries = [
            summary for summary in summaries if summary["batch_size"] == batch_size
        ]
        selected = select_k_hw_for_batch(
            batch_summaries,
            rho=args.rho,
            marginal_gain_epsilon=args.marginal_gain_epsilon,
        )
        k_hw_by_batch[str(batch_size)] = selected

    calibration["measurements"] = summaries
    calibration["k_hw_by_batch"] = k_hw_by_batch
    with open(calibration_path, "w", encoding="utf-8") as fout:
        json.dump(calibration, fout, indent=2, sort_keys=True)
        fout.write("\n")

    print("\nK_hw candidates:")
    for batch_size in args.batch_sizes:
        selected = k_hw_by_batch[str(batch_size)]
        if selected is None:
            print(f"  B={batch_size}: no valid measurement")
            continue
        print(
            f"  B={batch_size}: K={selected['k_verify_tokens']} "
            f"L={selected['draft_token_num']} "
            f"throughput={selected['verify_tokens_per_s_mean']:.1f} verify tok/s"
        )
    print(f"Summary written to {summary_path}")
    print(f"Calibration written to {calibration_path}")


if __name__ == "__main__":
    main()
