# Extended GSM8K K_hw Sweep Summary

This summarizes the extended hardware-oriented target verify sweep for
Llama 3.1 8B + EAGLE3 on GSM8K.

## Setup

- Dataset: GSM8K test split
- Prompt format: 5-shot GSM8K prompt plus one held-out GSM8K question
- Measured unique prompts per candidate: 256
- Host: `47.116.170.43`
- Container: `sglang_cu13`
- Device: physical GPU 1 exposed as `cuda:0` via `CUDA_VISIBLE_DEVICES=1`
- Target model: `/home/hhuang/.cache/huggingface/LLM-Research/llama_3_1`
- Draft model: `/home/hhuang/.cache/huggingface/yuhuili/sglang-eagle3-8b`
- Attention backend: `triton`
- Dtype: `float16`
- Context length: `2048`

`B=64` used `max_tokens=64` so that `draft_token_num=64` is valid.
`B=128` and `B=160` used `max_tokens=32`.

## Interpretation

There are two useful definitions of "best":

- Strict peak: the measured candidate with highest target verify throughput.
- Calibrated choice: the smaller candidate selected by the benchmark rule
  (`rho=0.9`, `marginal_gain_epsilon=0.05`), which avoids increasing
  `draft_token_num` when the next gain is small.

There is also one SGLang/EAGLE3 naming detail:

- `draft_token_num` in these measurements is SGLang's
  `--speculative-num-draft-tokens`.
- For topk=1 EAGLE3, this target-verify width includes one current/root row.
- Therefore `K_verify = batch_size * draft_token_num` is the hardware-facing
  flattened target verify token count.
- The number of newly proposed draft tokens is
  `proposed_draft_tokens = batch_size * (draft_token_num - 1)`.

For roofline/profiling, `K_verify` is the right hardware axis. For scheduler
budget wording in the paper, `proposed_draft_tokens` is closer to "overall
draft number".

## Recommended Settings

| batch_size | measured draft_token_num range | calibrated draft_token_num | calibrated K_verify | calibrated proposed drafts | calibrated verify tok/s | strict-peak draft_token_num | strict-peak K_verify | strict-peak proposed drafts | strict-peak verify tok/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 16, 24, 32, 48, 64 | 32 | 2048 | 1984 | 6361.1 | 48 | 3072 | 3008 | 6660.5 |
| 128 | 2, 4, 8, 12, 16, 20, 24, 32 | 16 | 2048 | 1920 | 6167.3 | 24 | 3072 | 2944 | 6358.0 |
| 160 | 8, 12, 16, 20, 24, 28, 32 | 16 | 2560 | 2400 | 6218.2 | 28 | 4480 | 4320 | 6427.2 |

For the current objective of low-overhead target-model hardware calibration,
the calibrated settings are the safer choices. The strict peaks improve target
verify throughput only modestly, while larger `draft_token_num` values also
lower EAGLE3 accept rate and slow end-to-end generation.

## Measurements

| batch_size | draft_token_num | K_verify | proposed_draft_tokens | profiler_records | p50_verify_ms | verify tok/s | estimated TFLOPs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 16 | 1024 | 960 | 99 | 174.985 | 5848.3 | 83.9 |
| 64 | 24 | 1536 | 1472 | 100 | 260.829 | 5903.8 | 84.7 |
| 64 | 32 | 2048 | 1984 | 100 | 322.536 | 6361.1 | 91.3 |
| 64 | 48 | 3072 | 3008 | 100 | 461.995 | 6660.5 | 95.6 |
| 64 | 64 | 4096 | 4032 | 100 | 658.582 | 6227.5 | 89.4 |
| 128 | 2 | 256 | 128 | 17 | 74.925 | 3440.6 | 49.4 |
| 128 | 4 | 512 | 384 | 12 | 116.972 | 4428.6 | 63.5 |
| 128 | 8 | 1024 | 896 | 10 | 190.411 | 5430.7 | 77.9 |
| 128 | 12 | 1536 | 1408 | 10 | 276.410 | 5620.7 | 80.6 |
| 128 | 16 | 2048 | 1920 | 10 | 335.426 | 6167.3 | 88.5 |
| 128 | 20 | 2560 | 2432 | 10 | 406.214 | 6313.5 | 90.6 |
| 128 | 24 | 3072 | 2944 | 10 | 484.197 | 6358.0 | 91.2 |
| 128 | 32 | 4096 | 3968 | 10 | 683.422 | 6015.8 | 86.3 |
| 160 | 8 | 1280 | 1120 | 22 | 243.855 | 5280.2 | 75.7 |
| 160 | 12 | 1920 | 1760 | 22 | 340.885 | 5661.4 | 81.2 |
| 160 | 16 | 2560 | 2400 | 22 | 411.369 | 6218.2 | 89.2 |
| 160 | 20 | 3200 | 3040 | 22 | 513.094 | 6227.2 | 89.3 |
| 160 | 24 | 3840 | 3680 | 22 | 613.289 | 6266.5 | 89.9 |
| 160 | 28 | 4480 | 4320 | 22 | 696.641 | 6427.2 | 92.2 |
| 160 | 32 | 5120 | 4960 | 22 | 803.988 | 6368.5 | 91.4 |

## Notes

- `B=128, draft_token_num=16` was measured twice. The extended run value is
  used in this summary; it matches the original run closely.
- `B=256` was not run in this round. With GSM8K 5-shot prompts, the estimated
  prompt KV load is roughly 256 * 737 tokens before generation, which exceeds
  the current `max_total_num_tokens=141191` configuration. Running it fairly
  would require changing memory settings or the workload.
- Raw per-run files are in:
  - `test_instruction/results/khw_llama3_8b_eagle3_gsm8k_b64_ext_gpu1/`
  - `test_instruction/results/khw_llama3_8b_eagle3_gsm8k_b128_gpu1/`
  - `test_instruction/results/khw_llama3_8b_eagle3_gsm8k_b128_ext_gpu1/`
  - `test_instruction/results/khw_llama3_8b_eagle3_gsm8k_b160_ext_gpu1/`
  - `test_instruction/results/khw_llama3_8b_eagle3_gsm8k_b160_high_gpu1/`
