# 我写了一个从零实现的 CUDA 大模型推理引擎

最近我在做一个比较硬核的小项目：用 C++ / CUDA 从零实现一个大模型推理引擎。

项目地址：

[https://github.com/luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)

这个项目当前主要面向 DeepSeek-R1-Distill-Qwen-7B 的单 batch 推理。它不是在 PyTorch、Transformers、vLLM 或 llama.cpp 上套一层接口，而是尽量把推理核心路径自己写出来，直接用 CUDA 实现模型 forward 和 decode。

## 为什么做这个项目

现在大模型推理框架已经很多，vLLM、TensorRT-LLM、llama.cpp 都非常成熟。但如果想真正理解一个大模型在 GPU 上是怎么跑起来的，只会调用框架还不够。

我想做的是一个可以拆开看的推理引擎：

- 权重怎么加载
- RMSNorm 怎么算
- RoPE 怎么处理
- GQA Attention 怎么做
- KV Cache 怎么管理
- MLP / SwiGLU 怎么执行
- decode 每一步的耗时在哪里
- CUDA kernel 怎么一步一步优化

这个项目就是围绕这些问题写出来的。

## 项目特点

- 不依赖 PyTorch、Transformers、vLLM、llama.cpp
- 使用 C++ / CUDA 实现核心推理路径
- 手写 RMSNorm、RoPE、GQA Attention、SwiGLU、KV Cache、decode
- 支持 HuggingFace safetensors 权重加载
- 提供 Python tokenizer + CUDA 动态库推理入口
- 当前 `mma` 版本针对 A100 / A800 的单步 decode 做了多轮优化

## 当前性能

测试模型：

```text
DeepSeek-R1-Distill-Qwen-7B
```

当前记录：

```text
max_seq=800
max_new_tokens=512
512 tokens = 65.6845 tok/s
max forward_ms = 16.1768
```

这个速度是在单 batch、单步 decode 场景下测到的。它不是靠 batching 堆总吞吐，也不是 speculative decoding，而是比较直接地看每一步 target model forward 的速度。

## 如何运行

```bash
make -f Makefile.cuda_lib lib A=sm_80
CUDA_VISIBLE_DEVICES=4 python python_infer.py \
  --model /data3/ledi/models/DeepSeek-R1-Distill-Qwen-7B \
  --lib ./build/libllm_cuda.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 512 \
  --max-seq 800
```

## 后续方向

目前这个项目已经完成了基础推理链路和多轮 CUDA 优化。后面如果继续往上冲，主要会看几个方向：

- CUDA Graph，减少 decode 阶段的 kernel launch 开销
- 重写 decode GEMV / MLP 路径
- 更激进的 kernel fusion
- 量化推理
- speculative decoding

## 欢迎交流

这个项目更偏研究和实验性质，适合对 CUDA、大模型推理、底层性能优化感兴趣的人一起看、一起改、一起 benchmark。

如果你也对从零写推理引擎感兴趣，欢迎 star、fork 或交流：

[https://github.com/luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)
