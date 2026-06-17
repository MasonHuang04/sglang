# K_hw Calibration Result

This run calibrates the hardware-optimal target verify batch size for EAGLE3 speculative decoding with Llama 3.1 8B.

## Environment

- Host: `47.116.170.43`
- Container: `sglang_cu13`
- Device: physical GPU 1 exposed as `cuda:0` via `CUDA_VISIBLE_DEVICES=1`
- Model: `/home/hhuang/.cache/huggingface/LLM-Research/llama_3_1`
- Draft model: `/home/hhuang/.cache/huggingface/yuhuili/sglang-eagle3-8b`
- Attention backend: `triton`
- Dtype: `float16`
- Context length: `2048`
- Batch size: `128`
- Selection rule: smallest candidate with throughput at least `0.9 * peak` and next marginal gain at most `0.05`

## Result

The calibrated hardware-optimal setting for `batch_size=128` is:

- `draft_token_num = 8`
- `K_hw = batch_size * draft_token_num = 1024` verify tokens per batch

## Measurements

| batch_size | draft_token_num | K_verify_tokens | records | p50_verify_ms | mean_verify_tok_s | estimated_total_tflops |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 2 | 256 | 42 | 50.194 | 5163.96 | 72.15 |
| 128 | 4 | 512 | 34 | 89.103 | 5778.60 | 80.73 |
| 128 | 8 | 1024 | 36 | 167.436 | 6171.94 | 86.23 |
| 128 | 12 | 1536 | 36 | 252.328 | 6111.94 | 85.39 |
| 128 | 16 | 2048 | 36 | 315.630 | 6519.00 | 91.08 |

The roofline estimate was `K_roof = 377.1875`; it was used to guide the candidate grid, and profiling selected `K_hw = 1024` after calibration.

Raw low-overhead verify profiles are stored in `spec_verify_B128_L*_tp0_dp0.jsonl`. The machine-readable summary is in `summary.jsonl` and `calibration.json`.
