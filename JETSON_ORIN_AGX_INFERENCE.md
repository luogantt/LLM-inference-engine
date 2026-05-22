# 在 Jetson AGX Orin 上运行自研 CUDA 大模型推理引擎

最近我把自己的 CUDA 大模型推理引擎跑到了 Jetson AGX Orin 上，完成了 DeepSeek-R1-Distill-Qwen-7B 的本地推理测试。

项目地址：

[https://github.com/luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)

这个项目不是基于 PyTorch、Transformers、vLLM 或 llama.cpp 的推理后端，而是用 C++ / CUDA 手写推理核心路径。当前主要支持 DeepSeek-R1-Distill-Qwen-7B 的单 batch decode。

## 测试环境

硬件：

```text
Jetson AGX Orin
RAM: 64GB
GPU: Orin Ampere GPU
CUDA arch: sm_87
```

模型：

```text
DeepSeek-R1-Distill-Qwen-7B
模型目录: /data/project/deepseek-r1-7b
```

推理框架：

```text
LLM-inference-engine
branch: jetson-orin-agx
```

## Jetson 性能模式设置

测试前先把 Jetson AGX Orin 拉到高性能模式：

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
sudo tegrastats
```

实测时的频率状态：

```text
CPU: 2201 MHz
EMC: 3199 MHz
GPU: 1300 MHz
RAM: 9271 / 62842 MB
GPU temperature: about 42C
```

说明机器没有热降频，内存也足够加载 7B FP16 模型。

## Python 依赖

这个项目只用 `transformers` 做 tokenizer，不用 PyTorch 加载模型。

如果 Jetson 上缺少依赖，可以安装：

```bash
python -m pip install transformers tokenizers safetensors jinja2
```

如果遇到权限问题，可以加 `--user`：

```bash
python -m pip install --user transformers tokenizers safetensors jinja2
```

运行时看到下面提示是正常的：

```text
PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
```

因为这里只需要 tokenizer，不需要 PyTorch 模型推理。

## 编译

Jetson AGX Orin 的 CUDA 架构是 `sm_87`，需要重新编译动态库：

```bash
cd /data/project/LLM-inference-engine

make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib A=sm_87
```

不要打开当前代码里的 `USE_WMMA_LINEAR=1`。这个 WMMA kernel 只是实验 prototype，在 Jetson 上反而会明显变慢。

推荐使用普通编译：

```bash
make -f Makefile.cuda_lib lib A=sm_87
```

## 执行命令

```bash
CUDA_VISIBLE_DEVICES=0 python python_infer.py \
  --model /data/project/deepseek-r1-7b \
  --lib ./build/libllm_cuda.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 512 \
  --max-seq 800
```

## 实测结果

在 Jetson AGX Orin 高性能模式下，DeepSeek-R1-Distill-Qwen-7B FP16 跑通后，decode 阶段实测：

```text
decode step_ms: about 117.9 ms
forward_ms: about 117.8 ms
decode_tokens_per_s: about 8.6 tok/s
```

日志片段：

```text
[C++][time] decode step_ms=117.905, sample_ms=0.046753, forward_ms=117.841, decode_tokens=385, decode_tokens_per_s=8.60732
[C++][time] decode step_ms=117.960, sample_ms=0.044001, forward_ms=117.901, decode_tokens=386, decode_tokens_per_s=8.60698
[C++][time] decode step_ms=117.873, sample_ms=0.054176, forward_ms=117.803, decode_tokens=387, decode_tokens_per_s=8.60666
```

也就是说，在 Jetson AGX Orin 上，这个自研 CUDA 推理引擎当前可以跑到大约：

```text
DeepSeek-R1-Distill-Qwen-7B FP16: 8.6 tokens/s
```

## 对比过程

测试过程中有几组不同状态：

| 状态 | 速度 | 说明 |
|---|---:|---|
| WMMA prototype | about 1.14 tok/s | 不适合当前 Jetson 单 token GEMV 路径 |
| 未确认高性能状态的普通路径 | about 3.03 tok/s | 可以跑，但频率和编译状态不够理想 |
| 高性能模式 + sm_87 普通编译 | about 8.6 tok/s | 当前推荐结果 |

最终推荐配置是：

```text
sudo nvpmodel -m 0
sudo jetson_clocks
make -f Makefile.cuda_lib lib A=sm_87
```

## 结论

Jetson AGX Orin 可以运行这个自研 CUDA 大模型推理引擎，并且可以在 FP16 模式下跑通 DeepSeek-R1-Distill-Qwen-7B。

当前速度约为：

```text
8.6 tok/s
117.8 ms/token
```

这个结果说明 Jetson AGX Orin 可以作为边缘端大模型推理实验平台。不过 7B FP16 对 Orin 来说仍然很重，每一步 decode 都需要反复扫描大量权重。

如果后续想继续提升速度，优先方向不是继续微调 FP16 kernel，而是：

- INT4 / INT8 权重量化 GEMV
- 更适合 Orin 的 decode MLP 路径
- CUDA Graph 降低 launch overhead
- 更小模型，例如 1.5B / 3B

FP16 路线预计还能小幅优化，但如果目标是 15 到 20 tok/s，量化基本是必选路线。

项目地址：

[https://github.com/luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)
