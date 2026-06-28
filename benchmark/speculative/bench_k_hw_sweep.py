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
import time
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
from sglang.utils import download_and_cache_file
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)


BUILTIN_PROMPTS = [
    "Write a detailed technical note about cache locality in GPU inference.",
    "List practical steps for debugging a distributed inference timeout.",
    "Explain why batching changes arithmetic intensity in transformer decode.",
    "Draft a concise incident report for a slow model serving deployment.",
]

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
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


def gsm8k_example(line: dict[str, Any], *, include_answer: bool) -> str:
    result = f"Question: {line['question']}\nAnswer:"
    if include_answer:
        result += f" {line['answer']}"
    return result


def load_gsm8k_prompts(
    *,
    data_path: str,
    num_shots: int,
    num_prompts: int,
) -> list[str]:
    lines = read_jsonl(data_path)
    if len(lines) <= num_shots:
        raise ValueError(
            f"GSM8K data at {data_path!r} has {len(lines)} rows, "
            f"which is not enough for {num_shots} few-shot examples."
        )

    few_shot_prompt = "".join(
        gsm8k_example(lines[i], include_answer=True) + "\n\n"
        for i in range(num_shots)
    )
    eval_lines = lines[num_shots:]
    if num_prompts > 0:
        eval_lines = eval_lines[:num_prompts]
    return [
        few_shot_prompt + gsm8k_example(line, include_answer=False)
        for line in eval_lines
    ]


def load_prompt_pool(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    if args.prompt_source == "builtin":
        return BUILTIN_PROMPTS, {
            "source": "builtin",
            "num_prompts": len(BUILTIN_PROMPTS),
        }

    data_path = args.gsm8k_data_path
    downloaded = False
    if not data_path:
        data_path = str(args.output_dir / "gsm8k_test.jsonl")
        data_path = download_and_cache_file(GSM8K_URL, filename=data_path)
        downloaded = True

    prompts = load_gsm8k_prompts(
        data_path=data_path,
        num_shots=args.gsm8k_num_shots,
        num_prompts=args.num_prompts,
    )
    if not prompts:
        raise ValueError("GSM8K prompt pool is empty.")
    return prompts, {
        "source": "gsm8k",
        "data_path": data_path,
        "downloaded": downloaded,
        "num_prompts": len(prompts),
        "num_shots": args.gsm8k_num_shots,
        "min_measured_prompts": args.min_measured_prompts,
        "measured_prompt_requests_by_batch": {
            str(batch_size): batch_size * args.measure_rounds
            for batch_size in args.batch_sizes
        },
        "measured_unique_prompt_count_by_batch": {
            str(batch_size): min(len(prompts), batch_size * args.measure_rounds)
            for batch_size in args.batch_sizes
        },
        "url": GSM8K_URL if downloaded else None,
    }


def validate_prompt_coverage(
    args: argparse.Namespace,
    prompt_pool: list[str],
) -> None:
    if args.prompt_source != "gsm8k":
        return

    if len(prompt_pool) < args.min_measured_prompts:
        raise ValueError(
            f"GSM8K prompt pool has only {len(prompt_pool)} prompts, but "
            f"--min-measured-prompts requires at least {args.min_measured_prompts}."
        )

    for batch_size in args.batch_sizes:
        measured_prompt_requests = batch_size * args.measure_rounds
        if measured_prompt_requests < args.min_measured_prompts:
            raise ValueError(
                f"batch_size={batch_size} with measure_rounds={args.measure_rounds} "
                f"would send only {measured_prompt_requests} measured prompts. "
                f"Increase --measure-rounds or lower --min-measured-prompts."
            )


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


def post_batched_generate(
    base_url: str,
    prompt_pool: list[str],
    batch_size: int,
    max_tokens: int,
    prompt_offset: int,
) -> dict[str, Any]:
    prompts = [
        prompt_pool[(prompt_offset + i) % len(prompt_pool)]
        for i in range(batch_size)
    ]
    start_time = time.perf_counter()
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": prompts,
            "sampling_params": build_sampling_params(batch_size, max_tokens),
            "return_logprob": False,
        },
        timeout=max(300, max_tokens * 20),
    )
    latency_s = time.perf_counter() - start_time
    response.raise_for_status()
    response_payload = response.json()
    items = response_payload if isinstance(response_payload, list) else [response_payload]

    completion_tokens = 0
    spec_verify_ct = 0
    has_completion_tokens = False
    has_spec_verify_ct = False
    for item in items:
        if not isinstance(item, dict):
            continue
        meta_info = item.get("meta_info") or {}
        usage = item.get("usage") or {}

        item_completion_tokens = meta_info.get("completion_tokens")
        if item_completion_tokens is None:
            item_completion_tokens = usage.get("completion_tokens")
        if item_completion_tokens is None and isinstance(item.get("output_ids"), list):
            item_completion_tokens = len(item["output_ids"])
        if item_completion_tokens is not None:
            completion_tokens += int(item_completion_tokens)
            has_completion_tokens = True

        item_spec_verify_ct = meta_info.get("spec_verify_ct", item.get("spec_verify_ct"))
        if item_spec_verify_ct is not None:
            spec_verify_ct += int(item_spec_verify_ct)
            has_spec_verify_ct = True

    # The sweep sets ignore_eos=True and max_new_tokens=max_tokens. Some server
    # versions omit completion_tokens from HTTP responses, so keep a deterministic
    # fallback instead of dropping the end-to-end throughput signal.
    if not has_completion_tokens:
        completion_tokens = batch_size * max_tokens

    return {
        "latency_s": latency_s,
        "completion_tokens": completion_tokens,
        "spec_verify_ct": spec_verify_ct if has_spec_verify_ct else None,
        "num_requests": len(items),
    }


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
    prompt_pool_size: int,
    warmup_prompt_requests: int,
    measured_prompt_requests: int,
    request_metrics: list[dict[str, Any]],
    min_profile_batch_fraction: float,
) -> dict[str, Any]:
    min_profile_batch_size = max(1, int(batch_size * min_profile_batch_fraction))
    records = [
        record
        for record in records
        if record.get("draft_token_num") == draft_len
        and int(record.get("batch_size") or 0) >= min_profile_batch_size
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
    profile_batch_sizes = [
        int(r["batch_size"]) for r in records if r.get("batch_size") is not None
    ]
    profile_verify_tokens = [
        int(r["num_verify_tokens"])
        for r in records
        if r.get("num_verify_tokens") is not None
    ]
    profile_proposed_drafts = [
        int(r["num_proposed_drafts"])
        for r in records
        if r.get("num_proposed_drafts") is not None
    ]
    request_latencies_ms = [
        float(metric["latency_s"]) * 1000.0
        for metric in request_metrics
        if metric.get("latency_s") is not None
    ]
    completion_tokens_total = sum(
        int(metric.get("completion_tokens") or 0) for metric in request_metrics
    )
    spec_verify_ct_values = [
        int(metric["spec_verify_ct"])
        for metric in request_metrics
        if metric.get("spec_verify_ct") is not None
    ]
    spec_verify_ct_total = sum(spec_verify_ct_values) if spec_verify_ct_values else None
    e2e_latency_s_total = sum(float(metric["latency_s"]) for metric in request_metrics)
    k_verify_tokens = int(median_or_none(profile_verify_tokens) or batch_size * draft_len)
    k_proposed_drafts = int(
        median_or_none(profile_proposed_drafts)
        or batch_size * max(draft_len - 1, 0)
    )

    estimated_tflops = []
    dense_tflops = []
    for record in records:
        latency_ms = float(record["latency_ms"]) if record.get("latency_ms") else 0.0
        record_batch_size = int(record.get("batch_size") or batch_size)
        record_draft_len = int(record.get("draft_token_num") or draft_len)
        flops = estimate_target_verify_flops(
            shape=model_shape,
            batch_size=record_batch_size,
            draft_token_num=record_draft_len,
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
        "min_profile_batch_fraction": min_profile_batch_fraction,
        "min_profile_batch_size": min_profile_batch_size,
        "profile_batch_size_mean": mean_or_none(profile_batch_sizes),
        "profile_batch_size_p50": median_or_none(profile_batch_sizes),
        "num_records": len(records),
        "latency_ms_mean": mean_or_none(latencies),
        "latency_ms_p50": median_or_none(latencies),
        "latency_ms_p90": percentile(latencies, 0.90),
        "e2e_batch_latency_ms_mean": mean_or_none(request_latencies_ms),
        "e2e_batch_latency_ms_p50": median_or_none(request_latencies_ms),
        "e2e_output_tokens_per_s": (
            completion_tokens_total / e2e_latency_s_total
            if e2e_latency_s_total > 0
            else None
        ),
        "e2e_completion_tokens_total": completion_tokens_total,
        "e2e_spec_verify_ct_total": spec_verify_ct_total,
        "e2e_accept_length_from_meta": (
            completion_tokens_total / spec_verify_ct_total
            if spec_verify_ct_total
            else None
        ),
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
        "prompt_pool_size": prompt_pool_size,
        "warmup_prompt_requests": warmup_prompt_requests,
        "measured_prompt_requests": measured_prompt_requests,
        "measured_unique_prompt_count": min(prompt_pool_size, measured_prompt_requests),
        "profile_files": profile_files,
    }


def clean_profile_files(profile_glob: str) -> None:
    for path in glob.glob(profile_glob):
        os.remove(path)


def run_one_config(
    args: argparse.Namespace,
    model_shape: TargetModelShape,
    prompt_pool: list[str],
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

        request_index = 0
        for _ in range(args.warmup_rounds):
            post_batched_generate(
                base_url,
                prompt_pool,
                batch_size,
                args.max_tokens,
                prompt_offset=request_index * batch_size,
            )
            request_index += 1

        warmup_counts = {
            path: len(records)
            for path, records in read_profile_records(profile_glob).items()
        }

        request_metrics = []
        for _ in range(args.measure_rounds):
            request_metrics.append(
                post_batched_generate(
                    base_url,
                    prompt_pool,
                    batch_size,
                    args.max_tokens,
                    prompt_offset=request_index * batch_size,
                )
            )
            request_index += 1

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
            prompt_pool_size=len(prompt_pool),
            warmup_prompt_requests=batch_size * args.warmup_rounds,
            measured_prompt_requests=batch_size * args.measure_rounds,
            request_metrics=request_metrics,
            min_profile_batch_fraction=args.min_profile_batch_fraction,
        )
    finally:
        if process is not None:
            kill_process_tree(process.pid)


def print_summary_row(summary: dict[str, Any]) -> None:
    throughput = summary["verify_tokens_per_s_mean"]
    latency = summary["latency_ms_p50"]
    output_throughput = summary["e2e_output_tokens_per_s"]
    accept_length = summary["e2e_accept_length_from_meta"]
    total_tflops = summary["estimated_total_tflops_mean"]
    graph_rate = summary["cuda_graph_rate"]
    throughput_text = (
        f"{throughput:12.1f}" if throughput is not None else "        None"
    )
    latency_text = f"{latency:8.3f}" if latency is not None else "    None"
    output_throughput_text = (
        f"{output_throughput:12.1f}"
        if output_throughput is not None
        else "        None"
    )
    accept_length_text = (
        f"{accept_length:8.3f}" if accept_length is not None else "    None"
    )
    tflops_text = f"{total_tflops:8.1f}" if total_tflops is not None else "    None"
    graph_text = f"{graph_rate:6.2f}" if graph_rate is not None else "  None"
    print(
        f"{summary['batch_size']:>5} {summary['draft_token_num']:>5} "
        f"{summary['k_verify_tokens']:>8} {summary['num_records']:>7} "
        f"{latency_text} {output_throughput_text} {accept_length_text} "
        f"{throughput_text} {tflops_text} {graph_text}"
    )


def select_serving_best_for_batch(
    summaries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    valid = [
        summary
        for summary in summaries
        if summary.get("e2e_output_tokens_per_s") is not None
    ]
    if not valid:
        return None
    return max(valid, key=lambda summary: summary["e2e_output_tokens_per_s"])


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
        grid[batch_size] = [value for value in draft_lens if 1 <= value <= args.max_draft_len]
    return grid


def build_roofline_metadata(
    args: argparse.Namespace,
    model_shape: TargetModelShape,
    model_config_info: dict[str, Any],
    prompt_metadata: dict[str, Any],
    k_roof: float,
    run_grid: dict[int, list[int]],
) -> dict[str, Any]:
    dense_weight_bytes = estimate_dense_weight_bytes(
        model_shape, bytes_per_weight=args.bytes_per_weight
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
            "candidate_run_grid": {str(key): value for key, value in run_grid.items()},
        },
        "selection": {
            "rho": args.rho,
            "marginal_gain_epsilon": args.marginal_gain_epsilon,
        },
        "launch": {
            "attention_backend": args.attention_backend,
            "page_size": args.page_size,
            "dtype": args.dtype,
            "disable_overlap_schedule": args.disable_overlap_schedule,
            "disable_cuda_graph": args.disable_cuda_graph,
            "enable_piecewise_cuda_graph": args.enable_piecewise_cuda_graph,
            "extra_server_args": args.extra_server_args,
        },
        "prompt_source": prompt_metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find K_hw for speculative target verification."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TARGET_MODEL_EAGLE3,
        help="Target model path or HF id.",
    )
    parser.add_argument(
        "--draft-model",
        default=DEFAULT_DRAFT_MODEL_EAGLE3,
        help="Speculative draft model path or HF id.",
    )
    parser.add_argument(
        "--speculative-algorithm",
        default="EAGLE3",
        choices=["DFLASH", "EAGLE", "EAGLE3", "DRAFT", "DRAFT_EXTEND", "NGGRAM"],
    )
    parser.add_argument(
        "--draft-lens",
        default="auto",
        help="Comma-separated draft lengths, or 'auto' to use roofline candidates.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=[32, 64, 128],
        help="Comma-separated batch sizes.",
    )
    parser.add_argument("--extra-draft-lens", default="")
    parser.add_argument("--max-draft-len", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--measure-rounds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/sglang_k_hw"))
    parser.add_argument("--base-port", type=int, default=30000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--launch-timeout", type=float, default=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--disable-overlap-schedule", action="store_true", default=True)
    parser.add_argument("--enable-overlap-schedule", dest="disable_overlap_schedule", action="store_false")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--enable-piecewise-cuda-graph", action="store_true")
    parser.add_argument("--skip-server-warmup", action="store_true")
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument("--speculative-num-steps", type=int, default=None)
    parser.add_argument("--speculative-eagle-topk", type=int, default=1)
    parser.add_argument(
        "--prompt-source",
        choices=["builtin", "gsm8k"],
        default="gsm8k",
        help="Prompt pool used to drive speculative decode.",
    )
    parser.add_argument(
        "--gsm8k-data-path",
        default=None,
        help="Optional local GSM8K jsonl path. Downloads the test set when omitted.",
    )
    parser.add_argument("--gsm8k-num-shots", type=int, default=5)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=0,
        help="Number of GSM8K eval prompts to load after few-shot rows; 0 means all.",
    )
    parser.add_argument(
        "--min-measured-prompts",
        type=int,
        default=50,
        help="Require at least this many measured prompt requests for GSM8K runs.",
    )
    parser.add_argument(
        "--min-profile-batch-fraction",
        type=float,
        default=0.90,
        help=(
            "Keep target-verification profile records whose actual batch size is "
            "at least this fraction of the requested batch size."
        ),
    )
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
        help="Stop at the first near-peak point whose next relative gain is below this.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if (
        args.speculative_algorithm in ("DFLASH", "DRAFT_EXTEND")
        and args.draft_model == DEFAULT_DRAFT_MODEL_EAGLE3
    ):
        args.draft_model = DEFAULT_DRAFT_MODEL_DFLASH

    prompt_pool, prompt_metadata = load_prompt_pool(args)
    validate_prompt_coverage(args, prompt_pool)
    model_shape, model_config_info = load_target_model_shape(args.model)
    k_roof = estimate_k_roof(
        peak_tflops=args.peak_tflops,
        memory_bw_gbps=args.memory_bw_gbps,
        bytes_per_weight=args.bytes_per_weight,
    )
    run_grid = resolve_run_grid(args, k_roof=k_roof)
    calibration = build_roofline_metadata(
        args, model_shape, model_config_info, prompt_metadata, k_roof, run_grid
    )

    summary_path = args.output_dir / "summary.jsonl"
    calibration_path = args.output_dir / "calibration.json"
    summaries = []
    run_index = 0
    with open(summary_path, "w", encoding="utf-8") as fout:
        print(
            "    B     L        K records  p50_ms  output_tok/s accept_l "
            " verify_tok/s   TFLOPs  graph"
        )
        for batch_size in args.batch_sizes:
            for draft_len in run_grid[batch_size]:
                summary = run_one_config(
                    args, model_shape, prompt_pool, batch_size, draft_len, run_index
                )
                summaries.append(summary)
                fout.write(json.dumps(summary, sort_keys=True))
                fout.write("\n")
                fout.flush()
                print_summary_row(summary)
                run_index += 1

    k_hw_by_batch = {}
    serving_best_by_batch = {}
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
        serving_best_by_batch[str(batch_size)] = select_serving_best_for_batch(
            batch_summaries
        )

    calibration["measurements"] = summaries
    calibration["k_hw_by_batch"] = k_hw_by_batch
    calibration["serving_best_by_batch"] = serving_best_by_batch
    with open(calibration_path, "w", encoding="utf-8") as fout:
        json.dump(calibration, fout, indent=2, sort_keys=True)
        fout.write("\n")

    print("\nTarget-verify K_hw candidates:")
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

    print("\nServing output-throughput best candidates:")
    for batch_size in args.batch_sizes:
        selected = serving_best_by_batch[str(batch_size)]
        if selected is None:
            print(f"  B={batch_size}: no valid measurement")
            continue
        print(
            f"  B={batch_size}: K={selected['k_verify_tokens']} "
            f"L={selected['draft_token_num']} "
            f"throughput={selected['e2e_output_tokens_per_s']:.1f} output tok/s"
        )
    print(f"Summary written to {summary_path}")
    print(f"Calibration written to {calibration_path}")


if __name__ == "__main__":
    main()
