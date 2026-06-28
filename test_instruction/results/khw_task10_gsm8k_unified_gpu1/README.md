# Task 10 K_hw GSM8K Unified Sweep

This directory contains the unified Task 10 calibration run for target-model
verification hardware efficiency.

## What Was Tested

Task 10 asks for a low-overhead method to locate the hardware-optimal overall
draft token budget:

```text
K = sum_i draft_len_i
K = B * L  # uniform draft length case
```

The intended quantity is `K_hw`: the smallest target-verified draft-token count
that reaches the hardware-efficient region for the target model verification
step. This is not model loading time, CUDA graph capture time, or end-to-end
request latency.

The test uses:

| Item | Value |
|---|---|
| target model | `/home/hhuang/.cache/huggingface/LLM-Research/llama_3_1` |
| draft model | `/home/hhuang/.cache/huggingface/yuhuili/sglang-eagle3-8b` |
| algorithm | EAGLE3 |
| GPU | physical GPU 1 on the remote server |
| container | `sglang_cu13` |
| dataset | GSM8K test set, 0-shot prompts |
| batch sizes | 64, 128, 256 |
| draft lengths | 2, 3, 4, 6, 8, 10, 12, 16 |
| max new tokens | 32 |
| context length | 512 |
| warmup / measure | 1 / 1 batched request per config |
| profile filter | records with actual batch size >= 90% of requested batch |
| selection rule | `rho=0.90`, marginal gain <= 5% |

The main run used `mem_fraction_static=0.82`. The `B=256,L=16` point OOMed
under that setting before a measurement could complete, because the static KV
pool left too little free memory after CUDA graph capture. It was then measured
separately with `mem_fraction_static=0.70`, which keeps the same model, dataset,
batch size, draft length, context length, and generation length while reducing
the static KV allocation.

`L=1` is not included. In the current EAGLE3 path this maps to
`speculative_num_steps=0`, which is not a valid target-verification benchmark in
this setup.

## Output Files

| File | Meaning |
|---|---|
| `summary.jsonl` | Raw 23-row main run, excluding the OOMed `B=256,L=16` point. |
| `combined_summary.jsonl` | 24-row analysis table, including the `B=256,L=16` supplemental measurement. |
| `calibration_combined.json` | Combined K_hw and serving-best selections. |
| `retest_summary.jsonl` | Combined machine-readable rows from the three supplemental validation tests. |
| `task10_final_validation.md` | Final item-10 validation summary after the supplemental tests. |
| `../khw_task10_gsm8k_unified_gpu1_b256_l16_mem070/` | Raw supplemental `B=256,L=16` run. |
| `spec_verify_B*_L*_tp0_dp0.jsonl` | Raw per-batch target verification profile records. |

The earlier result directories were useful exploratory runs, but they did not
satisfy the full Task 10 requirement because they were fragmented across
different batch sizes and settings, some used non-GSM8K prompts, B256 coverage
was missing or filtered out, and older summaries did not consistently preserve
end-to-end and target-verification metrics.

## Combined Results

`K_verify` is taken from the profiler median actual verify-token count. For
B128 and B256 it can be `127 * L` or `255 * L` because the health/tail request
can make the actual profiled batch one request smaller than the requested batch.

| B | L | K_verify | records | profile B p50 | T_verify p50 ms | verify tok/s | TFLOPs | output tok/s | accept len | mem frac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2 | 128 | 19 | 64 | 28.705 | 4313.8 | 60.4 | 1393.4 | 1.488 | 0.82 |
| 64 | 3 | 192 | 17 | 64 | 38.731 | 4837.5 | 67.7 | 1251.0 | 1.662 | 0.82 |
| 64 | 4 | 256 | 16 | 64 | 47.134 | 5291.2 | 74.1 | 1115.5 | 1.740 | 0.82 |
| 64 | 6 | 384 | 15 | 64 | 70.231 | 5368.7 | 75.1 | 839.1 | 1.814 | 0.82 |
| 64 | 8 | 512 | 14 | 64 | 86.968 | 5798.7 | 81.2 | 691.4 | 1.812 | 0.82 |
| 64 | 10 | 640 | 14 | 64 | 102.180 | 6161.4 | 86.2 | 598.7 | 1.814 | 0.82 |
| 64 | 12 | 768 | 14 | 64 | 126.332 | 5976.4 | 83.7 | 519.3 | 1.816 | 0.82 |
| 64 | 16 | 1024 | 14 | 64 | 160.776 | 6258.6 | 87.6 | 422.9 | 1.820 | 0.82 |
| 128 | 2 | 254 | 20 | 127 | 52.259 | 4789.9 | 67.1 | 1130.5 | 1.456 | 0.82 |
| 128 | 3 | 381 | 17 | 127 | 74.146 | 5028.8 | 70.4 | 914.4 | 1.627 | 0.82 |
| 128 | 4 | 508 | 15 | 127 | 91.854 | 5463.2 | 76.5 | 824.0 | 1.710 | 0.82 |
| 128 | 6 | 762 | 14 | 127 | 134.720 | 5615.8 | 78.6 | 641.1 | 1.759 | 0.82 |
| 128 | 8 | 1016 | 14 | 127 | 168.315 | 5964.8 | 83.5 | 554.8 | 1.775 | 0.82 |
| 128 | 10 | 1270 | 14 | 127 | 213.601 | 5888.8 | 82.4 | 452.5 | 1.776 | 0.82 |
| 128 | 12 | 1524 | 14 | 127 | 252.732 | 5974.3 | 83.6 | 392.6 | 1.778 | 0.82 |
| 128 | 16 | 2032 | 14 | 127 | 313.275 | 6421.0 | 89.9 | 322.9 | 1.777 | 0.82 |
| 256 | 2 | 510 | 20 | 255 | 101.618 | 4983.4 | 69.8 | 1184.1 | 1.459 | 0.82 |
| 256 | 3 | 765 | 17 | 255 | 144.296 | 5275.7 | 73.8 | 1036.1 | 1.627 | 0.82 |
| 256 | 4 | 1020 | 15 | 255 | 178.302 | 5691.1 | 79.7 | 945.7 | 1.706 | 0.82 |
| 256 | 6 | 1530 | 14 | 255 | 263.363 | 5788.2 | 81.0 | 717.8 | 1.763 | 0.82 |
| 256 | 8 | 2040 | 14 | 255 | 325.597 | 6210.1 | 86.9 | 620.6 | 1.771 | 0.82 |
| 256 | 10 | 2550 | 14 | 255 | 392.457 | 6454.3 | 90.3 | 530.8 | 1.771 | 0.82 |
| 256 | 12 | 3060 | 14 | 255 | 471.882 | 6429.1 | 90.0 | 454.4 | 1.773 | 0.82 |
| 256 | 16 | 4080 | 14 | 255 | 667.327 | 6064.2 | 84.9 | 336.0 | 1.771 | 0.70 |

## Selected Points

| B | K_hw L | K_hw tokens | verify tok/s | peak verify tok/s | next gain | serving-best L | serving output tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 10 | 640 | 6161.4 | 6258.6 | -0.030 | 2 | 1393.4 |
| 128 | 8 | 1016 | 5964.8 | 6421.0 | -0.013 | 2 | 1130.5 |
| 256 | 8 | 2040 | 6210.1 | 6454.3 | 0.039 | 2 | 1184.1 |

## Interpretation

The previous results were not sufficient for Task 10. The unified run is closer
to the requested experiment because it uses one dataset, one prompt format, one
model pair, one context/generation setting, and a common B/L sweep.

The hardware curve behaves as expected: target-verification throughput rises
with `K` and then reaches a broad plateau. With the current selection rule,
`K_hw` is the earliest point near that plateau, not necessarily the largest
measured `K`.

Serving throughput peaks at `L=2` for all three batch sizes. That does not
contradict the K_hw result. It means that for GSM8K with this EAGLE3 draft
model, accepted length saturates around 1.7-1.8, so larger draft budgets improve
target verification hardware utilization but waste more rejected draft work in
the end-to-end serving objective.

The practical conclusion is:

```text
K_hw: hardware budget for target verification
K*: serving/workload optimum after acceptance behavior is included
```

For dynamic draft allocation, this means `K_hw` should be used as the hardware
budget that the scheduler distributes across requests. The scheduler should
still choose a smaller serving budget when the marginal accepted-token gain is
low, as this GSM8K run shows.
