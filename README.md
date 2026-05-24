# LLM CUDA Inference Engine

一个从零实现的 CUDA 大模型推理引擎，当前主要面向 DeepSeek-R1-Distill-Qwen-7B 的单 batch 推理。

项目目标是尽量少依赖外部推理框架，用 C++ / CUDA 手写核心推理路径，方便学习、实验和性能优化。

## 特点

- C++ / CUDA 实现核心推理路径
- 支持 HuggingFace safetensors 权重加载
- 手写 RMSNorm、RoPE、GQA Attention、SwiGLU、KV Cache、decode
- 提供 Python tokenizer + CUDA 动态库调用入口
- 支持 FP16 权重推理和 Jetson Orin AGX weight-only INT8 / INT4 推理

## A100 / A800 编译运行

```bash
make -f Makefile.cuda_lib lib A=sm_80
CUDA_VISIBLE_DEVICES=4 python python_infer.py \
  --model /data3/ledi/models/DeepSeek-R1-Distill-Qwen-7B \
  --lib ./build/libllm_cuda.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 512 \
  --max-seq 800
```

## Jetson Orin AGX INT8 编译运行

```bash
make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-int8 A=sm_87
CUDA_VISIBLE_DEVICES=0 python python_infer.py \
  --model /data/project/deepseek-r1-7b \
  --lib ./build/libllm_cuda.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 512 \
  --max-seq 800
```

运行时如果日志中出现 `stored=INT8(rowwise)`，说明已经启用 INT8 权重路径。

## Jetson Orin AGX INT4 编译运行

```bash
make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-int4 A=sm_87
CUDA_VISIBLE_DEVICES=0 python python_infer.py \
  --model /data/project/deepseek-r1-7b \
  --lib ./build/libllm_cuda.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 512 \
  --max-seq 800
```

运行时如果日志中出现 `stored=INT4(rowwise)`，说明已经启用 INT4 权重路径。

## Jetson Orin AGX INT4 o2-all 编译运行

`lib-int4-o2-all` 使用 INT4 weight-only + INT8 activation + DP4A，并在普通 linear、QKV、gate/up 路径中启用 2-output GEMV kernel，用于减少 kernel 调度和 activation 重复读取开销。

```bash
make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-int4-o2-all A=sm_87

CUDA_VISIBLE_DEVICES=0 python python_infer.py \
  --model /data/project/deepseek-r1-7b \
  --lib ./build/libllm_cuda.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 512 \
  --max-seq 800
```

## 当前性能

测试模型：

```text
DeepSeek-R1-Distill-Qwen-7B
max_seq=800
max_new_tokens=512
```

A100 / A800 MMA 版本：

```text
512 tokens = 65.6845 tok/s
max forward_ms = 16.1768
tag = mma_max_forward_ms=16.1768_512_tokens=65.6845_tok_s
```

Jetson Orin AGX weight-only INT8 版本：

```text
512 tokens = 14.0175 tok/s
decode forward_ms ≈ 72-74 ms
FP16 baseline ≈ 8.6 tok/s
INT8 speedup ≈ 1.63x
```

Jetson Orin AGX weight-only INT4 版本：

```text
lib-int4 baseline ≈ 20.7 tok/s
lib-int4-o2 ≈ 21.1 tok/s
lib-int4-o2-all ≈ 22.54 tok/s
decode forward_ms ≈ 45-46 ms
tag = jetson_orin_agx_int4_o2_all_forward_ms=46.4326_474_tokens=22.5382_tok_s
```

## 主要文件

```text
src/llm_cuda_lib.cu                  CUDA 推理核心
python_infer.py                      Python 调用入口
Makefile.cuda_lib                    动态库编译入口
log.txt                              性能记录
```

## 后续方向

当前 INT8 / INT4 版本是 weight-only 量化，激活仍然使用 float。继续提升 Jetson Orin AGX 速度的主要方向是 INT8 activation + DP4A GEMV、INT4 专用 GEMV、decode GEMV / MLP 重写和 CUDA Graph。

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