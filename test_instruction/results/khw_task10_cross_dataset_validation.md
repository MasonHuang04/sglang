# Task 10 Cross-Dataset Validation

This file extends the original GSM8K Task 10 validation to MT-Bench and
Spec-Bench without replacing the GSM8K results.

Task 10 asks for a low-overhead method to choose the hardware-efficient overall
draft-token budget:

```text
K = sum_i draft_len_i
K ~= batch_size * draft_len
```

The method is the same as the GSM8K validation: use the roofline model to define
the target-verification budget region, then calibrate it with real profiling.
The selection rule is the first measured point on the target-verification
plateau (`rho=0.90`) whose next marginal verify-throughput gain is small
(`<= 0.05`).

## Data

| Dataset | Prompt pool used | Source | Result directory |
|---|---:|---|---|
| GSM8K | Existing Task 10 run | `khw_task10_gsm8k_unified_gpu1/` | `khw_task10_gsm8k_unified_gpu1/` |
| MT-Bench | 160 prompts | FastChat MT-Bench question file, 80 questions split into two independent turns | `khw_task10_mtbench_160_gpu1_ctx512/` |
| Spec-Bench | 480 prompts | RACER `data/spec_bench/question.jsonl` | `khw_task10_specbench_480_gpu2_ctx2048/` |

Dataset source links:

- MT-Bench: https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl
- Spec-Bench: https://raw.githubusercontent.com/hkr04/RACER/main/data/spec_bench/question.jsonl

## Setup

| Field | Value |
|---|---|
| Target model | `/home/hhuang/.cache/huggingface/LLM-Research/llama_3_1` |
| Draft model | `/home/hhuang/.cache/huggingface/yuhuili/sglang-eagle3-8b` |
| Speculative algorithm | EAGLE3 |
| GPUs | MT-Bench on GPU 1, Spec-Bench on GPU 2 |
| Hardware parameters | peak FP16 362.1 TFLOPS, memory BW 960 GB/s |
| Batch sizes | 64, 128, 256 |
| Draft lengths | 2, 3, 4, 6, 8, 10, 12, 16 |
| Max generation tokens | 32 |
| MT-Bench context length | 512 |
| Spec-Bench context length | 2048 |
| New-run rounds | 1 warmup round, 1 measure round |

GSM8K remains the previously validated result set. The two new datasets use the
same sweep grid and selection logic, but one measure round to keep the added
validation cost bounded.

## Selected K_hw

`K_hw` is the hardware-efficient target-verification budget. It is not the same
as the serving-optimal draft length.

| Dataset | B | Selected L | K_verify | Verify tok/s | Output tok/s | Accept length | Est. TFLOPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 64 | 10 | 640 | 6161.4 | 598.7 | 1.814 | - |
| GSM8K | 128 | 8 | 1016 | 5964.8 | 554.8 | 1.775 | - |
| GSM8K | 256 | 8 | 2040 | 6210.1 | 620.6 | 1.771 | - |
| MT-Bench | 64 | 10 | 640 | 6136.0 | 538.9 | 1.870 | 85.9 |
| MT-Bench | 128 | 8 | 1024 | 6084.2 | 698.8 | 1.693 | 85.1 |
| MT-Bench | 256 | 8 | 2048 | 6297.1 | 751.0 | 1.700 | 88.1 |
| Spec-Bench | 64 | 8 | 512 | 6042.1 | 627.7 | 1.531 | 84.5 |
| Spec-Bench | 128 | 8 | 1016 | 5491.9 | 285.2 | 2.009 | 78.0 |
| Spec-Bench | 256 | 8 | 2032 | 5924.3 | 396.8 | 1.799 | 83.5 |

Interpretation:

- GSM8K and MT-Bench select nearly the same hardware knee: B64 prefers L=10,
  while B128 and B256 prefer L=8.
- Spec-Bench shifts B64 earlier to L=8. Its prompts are longer and more mixed,
  so the profiled plateau is already reached at a lower `K_verify`.
- For all three datasets, B128 and B256 select L=8 as the calibrated hardware
  knee.

## Serving-Best Draft Length

Serving-best is selected by end-to-end output tokens per second, not by target
verification throughput.

| Dataset | B | Serving-best L | K_verify | Output tok/s | Accept length |
|---|---:|---:|---:|---:|---:|
| GSM8K | 64 | 2 | 128 | 1393.4 | 1.488 |
| GSM8K | 128 | 2 | 254 | 1130.5 | 1.456 |
| GSM8K | 256 | 2 | 510 | 1184.1 | 1.459 |
| MT-Bench | 64 | 2 | 128 | 1276.1 | 1.486 |
| MT-Bench | 128 | 2 | 256 | 2081.7 | 1.421 |
| MT-Bench | 256 | 2 | 512 | 2312.9 | 1.420 |
| Spec-Bench | 64 | 2 | 128 | 1303.1 | 1.369 |
| Spec-Bench | 128 | 2 | 254 | 354.7 | 1.542 |
| Spec-Bench | 256 | 2 | 510 | 580.9 | 1.455 |

Conclusion: all three datasets still have serving-best L=2. This does not
contradict `K_hw`. Larger draft budgets make target verification more
hardware-efficient, but the accepted length saturates around roughly 1.4 to 2.0
tokens in these runs. After that point, extra drafted tokens add verification
work faster than they add accepted output tokens.

## New Dataset Full Sweep Highlights

MT-Bench:

| B | L=2 output tok/s | L=8 output tok/s | L=10 output tok/s | L=16 output tok/s | Selected K_hw |
|---:|---:|---:|---:|---:|---:|
| 64 | 1276.1 | 622.8 | 538.9 | 369.7 | L=10 |
| 128 | 2081.7 | 698.8 | 553.8 | 380.7 | L=8 |
| 256 | 2312.9 | 751.0 | 622.5 | 369.8 | L=8 |

Spec-Bench:

| B | L=2 output tok/s | L=8 output tok/s | L=10 output tok/s | L=16 output tok/s | Selected K_hw |
|---:|---:|---:|---:|---:|---:|
| 64 | 1303.1 | 627.7 | 532.9 | 354.3 | L=8 |
| 128 | 354.7 | 285.2 | 261.0 | 218.5 | L=8 |
| 256 | 580.9 | 396.8 | 360.6 | 259.8 | L=8 |

Machine-readable details are in each result directory:

- `summary.jsonl`: one row per `(batch_size, draft_len)` sweep point.
- `calibration.json`: selected `k_hw_by_batch`, `serving_best_by_batch`, and
  hardware/model metadata.
- `spec_verify_B*_L*_tp0_dp0.jsonl`: raw target-verification profiling records.
