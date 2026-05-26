# LLM CUDA Inference Engine

一个从零实现的 CUDA / Ascend 大模型推理引擎，当前主要面向 DeepSeek-R1-Distill-Qwen-7B 的单 batch decode 推理。

项目目标是尽量少依赖外部推理框架，用 C++ / CUDA / AscendCL / ACLNN 手写核心推理路径，便于学习、实验和性能优化。

## 特点

- 不依赖 PyTorch、Transformers、vLLM、llama.cpp 的直接推理路径
- CUDA 路线手写 RMSNorm、RoPE、GQA Attention、SwiGLU、KV Cache、decode
- Ascend 路线使用 AscendCL / ACLNN 构建 no-torch `.so` 推理路径
- 支持 HuggingFace safetensors 权重加载
- 提供 Python tokenizer + 动态库推理入口
- 当前 Ascend 路线已将 QKV、attention output projection、MLP、lm_head MatMul 搬到 ACLNN

## CUDA 编译与运行

```bash
make -f Makefile.cuda_lib lib A=sm_80

CUDA_VISIBLE_DEVICES=4 python python_infer.py \
  --model /data3/ledi/models/DeepSeek-R1-Distill-Qwen-7B \
  --lib ./build/libllm_cuda.so \
  --prompt "黑格尔的哲学思想可以概括为" \
  --max-new-tokens 128 \
  --max-seq 800
```

## 当前 CUDA 性能记录

测试模型：

```text
/data3/ledi/models/DeepSeek-R1-Distill-Qwen-7B
```

当前记录：

```text
max_seq=800
max_new_tokens=512
512 tokens = 65.6845 tok/s
max forward_ms = 16.1768
```

对应 tag：

```text
mma_max_forward_ms=16.1768_512_tokens=65.6845_tok_s
```

## 主要文件

```text
src/llm_cuda_lib.cu                  CUDA 推理核心
src/llm_ascend_lib.cpp               AscendCL / ACLNN 推理核心
python_infer.py                      Python 动态库调用入口
python_infer_ascend.py               torch_npu 参考入口
Makefile.cuda_lib                    动态库编译入口
llm_cuda_python_tokenizer_v2/         tokenizer 相关代码
log.txt                              性能记录
```

## Model Download

This repository includes `download_model.py` for downloading the HuggingFace safetensors model used by the engine. For China mainland networks, ModelScope is usually the fastest source:

```bash
pip install -U modelscope

python download_model.py \
  --source modelscope \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir /root/autodl-tmp/deepseek-r1-7b
```

For HuggingFace Hub instead:

```bash
pip install -U huggingface_hub

python download_model.py \
  --source huggingface \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir ./deepseek-r1-7b
```

## Ascend 910 torch_npu Reference

For Ascend 910 machines, the `torch_npu` entry can be used as a baseline:

```bash
export ASCEND_VISIBLE_DEVICES=4

python python_infer_ascend.py \
  --model /root/autodl-tmp/deepseek-r1-7b \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 128 \
  --max-seq 800 \
  --device npu:0 \
  --dtype float16
```

See `ASCEND.md` for setup and troubleshooting notes.

## Ascend Direct Decode Reference

The `Ascend` branch includes a CUDA-like direct AscendCL shared library path. These paths run inside `libllm_ascend.so` and do not import PyTorch.

Fast lm_head smoke test:

```bash
make -f Makefile.cuda_lib lib-ascend ASCEND_HOME=/usr/local/Ascend/cann-8.5.1

export ASCEND_VISIBLE_DEVICES=4
export ASCEND_DEVICE_ID=0
export ASCEND_LOAD_WEIGHTS=minimal
export ASCEND_RUN_EMBED=1
export ASCEND_RUN_RMSNORM=0
export ASCEND_RUN_QPROJ=0
export ASCEND_RUN_KVPROJ=0
export ASCEND_DIRECT_DECODE=lm_head_ref

python python_infer.py \
  --model ./deepseek-r1-7b \
  --lib ./build/libllm_ascend.so \
  --prompt "hello deepseek" \
  --max-new-tokens 1 \
  --max-seq 800 \
  --tokenizer-backend tokenizers
```

One-layer reference path:

```bash
export ASCEND_VISIBLE_DEVICES=4
export ASCEND_DEVICE_ID=0
export ASCEND_LOAD_WEIGHTS=layer0
export ASCEND_RUN_EMBED=1
export ASCEND_RUN_RMSNORM=0
export ASCEND_RUN_QPROJ=0
export ASCEND_RUN_KVPROJ=0
export ASCEND_DIRECT_DECODE=layer0_ref
export ASCEND_REF_CACHE_WEIGHTS=1
export ASCEND_REF_KV_CACHE=1
export ASCEND_REF_LINEAR_THREADS=16
export ASCEND_LM_HEAD_THREADS=16

python python_infer.py \
  --model ./deepseek-r1-7b \
  --lib ./build/libllm_ascend.so \
  --prompt "hello deepseek" \
  --max-new-tokens 1 \
  --max-seq 800 \
  --tokenizer-backend tokenizers
```

Complete no-torch reference path:

```bash
cd ~/LLM-inference-engine

git pull --ff-only origin Ascend

make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-ascend ASCEND_HOME=/usr/local/Ascend/cann-8.5.1

mkdir -p ~/ascend/log

export ASCEND_VISIBLE_DEVICES=4
export ASCEND_DEVICE_ID=0
export ASCEND_LOAD_WEIGHTS=all
export ASCEND_WEIGHT_LOAD_LOG=0
export ASCEND_HOST_RAW_CACHE=0
export ASCEND_RUN_EMBED=1
export ASCEND_DIRECT_DECODE=all_layers_ref
export ASCEND_REF_CACHE_WEIGHTS=1
export ASCEND_REF_CACHE_LOG=0
export ASCEND_REF_KV_CACHE=1
export ASCEND_REF_LINEAR_THREADS=16
export ASCEND_REF_ATTN_LINEAR_THREADS=8
export ASCEND_REF_MLP_THREADS=16
export ASCEND_REF_DOWN_THREADS=16
export ASCEND_LM_HEAD_THREADS=16
export ASCEND_REF_PROFILE_LAYERS=0

python python_infer.py \
  --model ./deepseek-r1-7b \
  --lib ./build/libllm_ascend.so \
  --prompt "请直接给出最终答案，用一段完整中文介绍黑格尔的哲学思想。" \
  --max-new-tokens 8 \
  --max-seq 800 \
  --tokenizer-backend tokenizers
```

## Ascend ACLNN Accelerated Inference

Current recommended direct Ascend inference command. This path keeps attention on CPU, and moves QKV, attention output projection, MLP, and lm_head MatMul to ACLNN:

```bash
cd ~/LLM-inference-engine

git pull --ff-only origin Ascend

make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-ascend ASCEND_HOME=/usr/local/Ascend/cann-8.5.1

mkdir -p ~/ascend/log

export ASCEND_VISIBLE_DEVICES=4
export ASCEND_DEVICE_ID=0

export ASCEND_LOAD_WEIGHTS=all
export ASCEND_WEIGHT_LOAD_LOG=0
export ASCEND_HOST_RAW_CACHE=0

export ASCEND_RUN_EMBED=1
export ASCEND_DIRECT_DECODE=all_layers_ref

export ASCEND_REF_CACHE_WEIGHTS=1
export ASCEND_REF_CACHE_LOG=0
export ASCEND_REF_KV_CACHE=1
export ASCEND_REF_U16_WEIGHTS=1

export ASCEND_REF_FAST_DOT=1
export ASCEND_REF_DOT4=0
export ASCEND_REF_NEON_DOT=0

export ASCEND_ATTN_BACKEND=cpu

export ASCEND_QKV_BACKEND=aclnn
export ASCEND_QKV_FALLBACK=0
export ASCEND_QKV_LOG=0

export ASCEND_ATTN_PROJ_BACKEND=aclnn
export ASCEND_ATTN_PROJ_FALLBACK=0
export ASCEND_ATTN_PROJ_LOG=0

export ASCEND_MLP_BACKEND=aclnn
export ASCEND_MLP_FALLBACK=0
export ASCEND_MLP_LOG=0

export ASCEND_LM_HEAD_BACKEND=aclnn
export ASCEND_LM_HEAD_FALLBACK=0
export ASCEND_LM_HEAD_LOG=0

export ASCEND_ACLNN_CUBE_MATH_TYPE=0

export ASCEND_REF_LINEAR_THREADS=16
export ASCEND_REF_ATTN_LINEAR_THREADS=16
export ASCEND_REF_ATTN_THREADS=16
export ASCEND_REF_ATTN_THREAD_MIN_SEQ=32
export ASCEND_REF_MLP_THREADS=24
export ASCEND_REF_DOWN_THREADS=24
export ASCEND_LM_HEAD_THREADS=16

export ASCEND_REF_PROFILE_LAYERS=0
export ASCEND_REF_PROFILE_TOKEN_LIMIT=0

python python_infer.py \
  --model ./deepseek-r1-7b \
  --lib ./build/libllm_ascend.so \
  --prompt "黑格尔的哲学思想可以概括为" \
  --max-new-tokens 128 \
  --max-seq 800 \
  --tokenizer-backend tokenizers \
  --no-chat-template
```

## Ascend 当前性能记录

在 `ascend-fused-qkv-30tok` 版本附近，DeepSeek-R1-Distill-Qwen-7B 单 batch decode 的典型速度：

```text
短上下文峰值：约 32-34 tok/s
128 token 平均：约 28-30 tok/s
长输出尾段：约 25-26 tok/s
```

当前主要瓶颈是 attention 随 `seq_len` 增长带来的 CPU 侧开销。后续如果要继续追平或超过普通 `torchrun + torch_npu`，重点方向是实现真正融合的 AscendC attention kernel，而不是把 attention 拆成大量小 ACLNN op 调度。

## 说明

这个项目偏研究和实验性质，重点是理解并优化单 batch decode 路径。CUDA 后续方向包括 CUDA Graph、decode GEMV / MLP 重写、量化和 speculative decoding；Ascend 后续方向包括 AscendC fused attention、KV cache 常驻 NPU、减少 host-device 同步和更完整的算子融合。
