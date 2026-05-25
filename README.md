# LLM CUDA Inference Engine

一个从零实现的 CUDA 大模型推理引擎，当前主要面向 DeepSeek-R1-Distill-Qwen-7B 的单 batch 推理。

项目目标是尽量少依赖外部推理框架，用 C++ / CUDA 手写核心推理路径，便于学习、实验和性能优化。

## 特点

- 不依赖 PyTorch、Transformers、vLLM、llama.cpp
- CUDA 手写 RMSNorm、RoPE、GQA Attention、SwiGLU、KV Cache、decode
- 支持 HuggingFace safetensors 权重加载
- 提供 Python tokenizer + CUDA 动态库推理入口
- 当前 `mma` 版本针对 A100 / A800 的单步 decode 做了多轮优化

## 编译与运行

```bash
make -f Makefile.cuda_lib lib A=sm_80
CUDA_VISIBLE_DEVICES=4 python python_infer.py \
  --model /data3/ledi/models/DeepSeek-R1-Distill-Qwen-7B \
  --lib ./build/libllm_cuda.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 512 \
  --max-seq 800
```

## 当前性能

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
python_infer.py                      Python 调用入口
Makefile.cuda_lib                    动态库编译入口
llm_cuda_python_tokenizer_v2/         tokenizer 版本相关代码
log.txt                              性能记录
```

## 说明

这个项目偏研究和实验性质，重点是理解并优化单 batch decode 路径。后续如果继续提高速度，主要方向是 CUDA Graph、decode GEMV / MLP 重写、量化和 speculative decoding。

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

## Ascend 910

For Ascend 910 machines, use the `torch_npu` inference entry first:

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

See `ASCEND.md` for full setup and troubleshooting notes.

The `Ascend` branch also contains a first CUDA-like direct AscendCL shared library skeleton:

```bash
make -f Makefile.cuda_lib lib-ascend ASCEND_HOME=/usr/local/Ascend/cann-8.5.1

export ASCEND_VISIBLE_DEVICES=4
export ASCEND_DEVICE_ID=0
export ASCEND_LOAD_WEIGHTS=layer0
export ASCEND_RUN_RMSNORM=1
export ASCEND_RUN_QPROJ=1
export ASCEND_QPROJ_REF_TOKENS=1

python python_infer.py \
  --model ./deepseek-r1-7b \
  --lib ./build/libllm_ascend.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 1 \
  --max-seq 800 \
  --tokenizer-backend tokenizers \
  --prefill-only
```

## Ascend Direct Decode Reference

The `Ascend` branch includes a CUDA-like direct AscendCL shared library path.
The fastest direct smoke test is `lm_head_ref`:

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

For the deeper one-layer reference path:

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

For the complete no-torch `.so` reference path:

```bash
export ASCEND_VISIBLE_DEVICES=4
export ASCEND_DEVICE_ID=0
export ASCEND_LOAD_WEIGHTS=all
export ASCEND_WEIGHT_LOAD_LOG=0
export ASCEND_HOST_RAW_CACHE=1
export ASCEND_HOST_RAW_DROP_AFTER_CONVERT=1
export ASCEND_RUN_EMBED=1
export ASCEND_DIRECT_DECODE=all_layers_ref
export ASCEND_REF_CACHE_WEIGHTS=1
export ASCEND_REF_CACHE_LOG=0
export ASCEND_REF_KV_CACHE=1
export ASCEND_REF_LINEAR_THREADS=16
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

`all_layers_ref` runs all 28 Transformer layers inside `libllm_ascend.so`.
It is a correctness/reference path and does not import PyTorch.
