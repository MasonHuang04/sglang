# K_hw Calibration Result With GSM8K

This run calibrates the hardware-optimal target verify batch size for EAGLE3 speculative decoding with Llama 3.1 8B using GSM8K prompts.

## Dataset

- Source: GSM8K test split from the OpenAI grade-school-math repository.
- Prompt format: 5-shot GSM8K prompt plus one held-out GSM8K question.
- Prompt pool used for this run: 256 GSM8K questions.
- Minimum measured prompts requested per candidate: 50.
- Actual measured prompts per candidate: 256.
- Actual measured unique prompts per candidate: 256.

The `records` column in `summary.jsonl` is not the number of GSM8K questions. It is the number of low-overhead target verify profiler records collected after warmup.

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

## Result

The calibrated hardware-optimal setting for `batch_size=128` on this GSM8K run is:

- `draft_token_num = 16`
- `K_hw = batch_size * draft_token_num = 2048` verify tokens per batch

## Measurements

| batch_size | draft_token_num | K_verify_tokens | measured_unique_prompts | profiler_records | p50_verify_ms | mean_verify_tok_s | estimated_total_tflops |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 2 | 256 | 256 | 17 | 74.925 | 3440.6 | 49.4 |
| 128 | 4 | 512 | 256 | 12 | 116.972 | 4428.6 | 63.5 |
| 128 | 8 | 1024 | 256 | 10 | 190.411 | 5430.7 | 77.9 |
| 128 | 12 | 1536 | 256 | 10 | 276.410 | 5620.7 | 80.6 |
| 128 | 16 | 2048 | 256 | 22 | 335.012 | 6115.3 | 87.7 |

Raw low-overhead verify profiles are stored in `spec_verify_B128_L*_tp0_dp0.jsonl`. The machine-readable summary is in `summary.jsonl` and `calibration.json`.
