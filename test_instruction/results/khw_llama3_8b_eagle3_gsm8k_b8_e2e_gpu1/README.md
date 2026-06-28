# B=8 GSM8K E2E Check

This run separates target-verify hardware efficiency from serving throughput.

## Setup

- Target model: `/home/hhuang/.cache/huggingface/LLM-Research/llama_3_1`
- Draft model: `/home/hhuang/.cache/huggingface/yuhuili/sglang-eagle3-8b`
- Dataset: GSM8K test JSONL, 5-shot prompts
- Batch size: 8
- Measured prompts: 56
- Draft token numbers: 2, 4, 8, 16, 32, 64
- GPU: physical GPU 1 exposed inside `sglang_cu13` with `CUDA_VISIBLE_DEVICES=1`

## Results

| draft_token_num | K_verify | verify p50 ms | verify tok/s | output tok/s | accept length |
|---:|---:|---:|---:|---:|---:|
| 2 | 16 | 21.16 | 754.0 | 391.5 | 1.409 |
| 4 | 32 | 22.33 | 1424.2 | 376.3 | 1.631 |
| 8 | 64 | 23.23 | 2742.1 | 323.6 | 1.673 |
| 16 | 128 | 25.04 | 5085.7 | 254.9 | 1.674 |
| 32 | 256 | 38.81 | 6634.6 | 149.4 | 1.676 |
| 64 | 512 | 71.72 | 7154.2 | 78.6 | 1.674 |

## Interpretation

The target-verify hardware metric selects `draft_token_num=64` because it maximizes verified-token throughput. That is not the serving optimum.

For actual end-to-end output throughput at `batch_size=8`, the best measured setting is `draft_token_num=2`. `draft_token_num=4` is close, `8` is already slower, and `16/32/64` are clearly bad despite higher target-verify throughput.

The earlier large recommendations came from optimizing `K_verify = batch_size * draft_token_num` only. That is a hardware saturation budget, not a per-request speculative draft-length recommendation. For serving, `K_hw` must be combined with acceptance and end-to-end throughput; for this B=8 GSM8K run, the practical per-request range is 2 to 4, with 8 as an upper bound to test, not 16+.
