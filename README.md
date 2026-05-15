# CUDA LLM From Scratch：不用 torch / transformers / llama.cpp 的最小推理框架

这个项目按你的要求写：

- 不用 PyTorch
- 不用 Transformers
- 不用 llama.cpp
- 不用 vLLM
- 不用 cuBLAS / cuDNN
- 只用 C++17 标准库 + CUDA Runtime
- CUDA 源码手写 RMSNorm / Linear / RoPE / GQA Attention / SwiGLU / KV Cache / greedy decode

## 文件说明

```text
cuda_llm_from_scratch/
├── Makefile
├── README.md
└── src/
    ├── toy_cuda_infer.cu
    └── deepseek7b_token_cuda_infer.cu
```

## 1. toy_cuda_infer.cu

完全自包含的小模型版本，随机权重，可以直接编译运行，用来验证推理链路。

```bash
make toy
./toy_cuda_infer
```

它实现：

```text
Embedding
RMSNorm
Q/K/V Linear
RoPE
GQA causal attention
O projection
SwiGLU MLP
Residual
Final RMSNorm
LM Head
Greedy Decode
```

## 2. deepseek7b_token_cuda_infer.cu

这个版本会直接扫描你的 HuggingFace safetensors 模型目录：

```bash
/home/lg/推理/推理引擎/deepseek-r1-7b
```

它会读取：

```text
model-00001-of-000002.safetensors
model-00002-of-000002.safetensors
```

并加载这些权重：

```text
model.embed_tokens.weight
model.layers.i.input_layernorm.weight
model.layers.i.self_attn.q_proj.weight
model.layers.i.self_attn.k_proj.weight
model.layers.i.self_attn.v_proj.weight
model.layers.i.self_attn.o_proj.weight
model.layers.i.self_attn.q_proj.bias
model.layers.i.self_attn.k_proj.bias
model.layers.i.self_attn.v_proj.bias
model.layers.i.post_attention_layernorm.weight
model.layers.i.mlp.gate_proj.weight
model.layers.i.mlp.up_proj.weight
model.layers.i.mlp.down_proj.weight
model.norm.weight
lm_head.weight
```

## 重要限制

这个版本没有实现 tokenizer。

所以它的输入不是中文文本，而是 token id：

```bash
./deepseek7b_token_cuda_infer --model  ../DeepSeek-R1-Distill-Qwen-7B --tokens 1,2,3 --steps 5 --max-seq 128

```

输出也是 token id。

为什么不直接输入中文？

因为 Qwen / DeepSeek 的 tokenizer 是 BPE / byte-level / chat-template 的组合。纯 C++ 从零实现 tokenizer 是另一个独立工程。这个项目先把“模型 forward + decode”打通。

## 编译

A800 / A100：

```bash
make deepseek7b A=sm_80
```

RTX 4090：

```bash
make deepseek7b A=sm_89
```

默认：

```bash
make deepseek7b
```

## 运行

```bash
./deepseek7b_token_cuda_infer \
  --model /home/lg/推理/推理引擎/deepseek-r1-7b \
  --tokens 1,2,3 \
  --steps 5 \
  --max-seq 128
```

## 显存说明

这个版本为了代码最简单，把 BF16 权重加载后转成 float32 放 GPU。

DeepSeek-R1-Distill-Qwen-7B 权重文件大约 15GB BF16，转成 FP32 后大约 30GB 以上。

所以：

- A800 80GB：可以试
- A100 80GB：可以试
- 4090 24GB：大概率放不下
- 3090 24GB：大概率放不下

## 性能说明

这个版本完全不用 cuBLAS，也没有 Tensor Core GEMM。

Linear 是手写 naive GEMV：

```text
一个输出维度一个线程
每个线程串行累加 input_dim
```

所以它很慢，但结构最清楚。

真正高性能版本应该逐步替换：

```text
naive Linear      -> tiled GEMM / Tensor Core
naive Attention   -> FlashAttention / FlashDecoding
float32 weights   -> half / bf16
CPU argmax        -> GPU reduce argmax
无 tokenizer       -> C++ tokenizer
```

这个项目是“从零写推理器”的第 0 版。

```
CUDA_VISIBLE_DEVICES=4 python python_infer.py --model /data3/ledi/models/DeepSeek-R1-Distill-Qwen-7B --lib ./build/libllm_cuda.so --prompt "你好 deepseek" --max-new-tokens 3 --max-seq 128
```
