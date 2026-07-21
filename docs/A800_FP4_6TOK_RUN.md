# A800 FP4 ~6 token/s configuration

This branch contains only the FP4 A800 inference path. It does not require or
include IQ2, Q2, GGUF, or external inference-engine code.

Use four otherwise-idle A800 GPUs (physical GPUs 0,2,3,4 in the measured host):

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 \
NCCL_PROTO=LL128 \
A800_USE_QWARP_TOPK=1 \
torchrun --nproc-per-node=4 python_infer_deepseek_v4_flash.py \
  --ckpt-path /data3/ledi/deepseekv4_engin/models/DeepSeek-V4-Flash-converted \
  --config DeepSeek-V4-Flash/inference/config.json \
  --prompt '请详细介绍人工智能的发展历史。' \
  --max-new-tokens 128 --max-seq-len 512 --warmup --bench-iters 3
```

Throughput depends on prompt length and whether prefill is included. The
measured steady FP4 decode path is approximately 6 token/s; short responses
under-report it because prompt processing is amortized over fewer output tokens.
