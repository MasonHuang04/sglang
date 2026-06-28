# Task 10 Final Validation

This file records the three supplemental tests for instruction.md item 10.
The scope is only the hardware-optimal overall draft token number for target
model verification:

```text
K = sum_i draft_len_i
K = B * L  # uniform draft length sweep
```

The selection rule is the one used by `bench_k_hw_sweep.py`: choose the first
point that reaches the target-verification plateau (`rho=0.90` of the measured
peak) and whose next relative gain is small (`<= 0.05`). Therefore `K_hw` is the
early hardware-efficient knee, not necessarily the single largest measured
verified-token throughput.

## Supplemental Runs

All supplemental runs use GSM8K 0-shot prompts, Llama-3.1-8B as target,
EAGLE3-8B as draft, GPU 1 on the remote server, `sglang_cu13`,
`context_length=512`, `max_tokens=32`, and `measure_rounds=3`.

| Test class | Result directory | Purpose |
|---|---|---|
| Small-L serving retest | `../khw_task10_gsm8k_retest_serving_smallL_gpu1/` | Recheck serving/output trend for L=2,4,8 and provide same-K B128/B256 support points. |
| Candidate and same-K retest | `../khw_task10_gsm8k_retest_b64_candidates_samek_gpu1/` | Recheck B64 near the selected hardware knee and include B64,L16 for K around 1024. |
| Candidate neighbor retests | `../khw_task10_gsm8k_retest_b128_candidates_gpu1/`, `../khw_task10_gsm8k_retest_b256_candidates_gpu1/` | Recheck neighboring L values around the selected B128/B256 hardware knees. |

Machine-readable combined retest rows are in `retest_summary.jsonl`.

## Test 1: Candidate Stability

The table compares supplemental retest points against the peak from the full
main sweep for each batch size.

| B | L | K_verify | verify tok/s | % of main peak | output tok/s | Interpretation |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 8 | 512 | 5758.9 | 92.0% | 686.2 | Near plateau, but L10 still gives more than 5% relative gain. |
| 64 | 10 | 640 | 6091.7 | 97.3% | 593.3 | Stable selected knee. |
| 64 | 12 | 768 | 5879.2 | 93.9% | 498.1 | No useful gain over L10. |
| 64 | 16 | 1024 | 6189.6 | 98.9% | 403.6 | Slightly higher verify throughput, but much later in the plateau. |
| 128 | 6 | 762 | 5570.0 | 86.7% | 699.2 | Below the 90% plateau threshold from the full sweep. |
| 128 | 8 | 1016 | 5898.3 | 91.9% | 594.0 | Stable selected knee. |
| 128 | 10 | 1270 | 5797.1 | 90.3% | 480.3 | No useful gain over L8 in the retest. |
| 256 | 6 | 1530 | 5683.8 | 88.1% | 744.1 | Below the 90% plateau threshold. |
| 256 | 8 | 2040 | 6122.5 | 94.9% | 641.5 | Stable selected knee. |
| 256 | 10 | 2550 | 6340.9 | 98.2% | 549.2 | Higher absolute verify throughput, but only 3.6% above L8. |
| 256 | 12 | 3060 | 6334.6 | 98.1% | 470.9 | No useful gain over L10. |

Conclusion: the selected knees remain:

| B | Selected L | K_hw tokens | Reason |
|---:|---:|---:|---|
| 64 | 10 | 640 | First stable plateau point after L8 still has more than 5% next gain. |
| 128 | 8 | 1016 | First retested point above 90% of the full-sweep peak. |
| 256 | 8 | 2040 | First retested point above 90%; L10 is only a small verify-throughput gain. |

## Test 2: Same-K Check

This check asks whether total verified draft tokens `K` is the right hardware
budget axis. The three points below are all around `K ~= 1024`.

| B | L | K_verify | verify tok/s | Relative to B64,L16 |
|---:|---:|---:|---:|---:|
| 64 | 16 | 1024 | 6189.6 | 100.0% |
| 128 | 8 | 1016 | 5898.3 | 95.3% |
| 256 | 4 | 1020 | 5579.9 | 90.1% |

Conclusion: `K` is the dominant budget variable, but not a perfect invariant in
this implementation. At the same `K`, batch size, sequence mix, attention cost,
and runtime overhead still move throughput by roughly 10%. This is exactly why
item 10 needs profiling calibration after the roofline estimate.

## Test 3: Serving Small-L Retest

Serving output throughput still prefers small draft length on GSM8K, even when
target verification becomes more hardware-efficient at larger `K`.

| B | L=2 output tok/s | L=4 output tok/s | L=8 output tok/s | Serving best |
|---:|---:|---:|---:|---:|
| 64 | 1369.8 | 1069.1 | 686.2 | 2 |
| 128 | 1154.0 | 899.7 | 594.0 | 2 |
| 256 | 1175.9 | 954.6 | 641.5 | 2 |

Conclusion: this does not contradict `K_hw`. It separates two quantities:

```text
K_hw = hardware-efficient target verification budget
K*   = serving/workload optimum after acceptance behavior is included
```

For this GSM8K/EAGLE3 setup, accepted length saturates around 1.7 to 1.8, so
larger draft budgets are good for target-verification utilization but bad for
end-to-end output throughput.

## Final Status

Instruction.md item 10 is complete for the current model/GPU/setup:

1. The code now supports a roofline-guided, profile-calibrated K_hw sweep.
2. The main GSM8K sweep covers B in {64,128,256} and L in {2,3,4,6,8,10,12,16}.
3. The three supplemental tests validate candidate stability, same-K behavior,
   and the distinction between hardware K_hw and serving K*.

Items 11 to 13 in instruction.md cover request-level draft-budget allocation
and early-exit verification. Those are outside the current item-10 scope.
