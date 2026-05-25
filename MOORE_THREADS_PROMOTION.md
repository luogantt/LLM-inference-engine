# 我把自研 LLM 推理引擎跑到了摩尔线程 MTT S4000 上：国产 GPU 路线，真的能跑起来

这次不是跑一个现成 demo，也不是套一层框架接口。

我做的是：把一个自己从零写的 LLM 推理引擎，从 NVIDIA CUDA 路径迁移到摩尔线程 MUSA 路径，让它在 MTT S4000 上真实加载 HuggingFace safetensors 模型，完成 DeepSeek-R1-Distill-Qwen-7B 的 prefill 和 decode 推理。

一句话总结：

> 这是一次从“能不能跑”到“可以实测”的国产 GPU LLM 推理适配实验。

当前结果很直接：

```text
GPU: MTT S4000 48GB
Driver: 2.7.0
MUSA Toolkit: 3.1.0
Model: DeepSeek-R1-Distill-Qwen-7B
Engine: 自研 C++ / GPU kernel 推理引擎
Path: MUSA FP16 compatibility path
Decode speed: about 19 tokens/s
Forward latency: about 51-54 ms/token
```

这不是最终优化速度，但它证明了一件更重要的事：

> 一个原本围绕 CUDA 写出来的 LLM 推理引擎，可以迁移到摩尔线程 MUSA，并完成真实大模型推理闭环。

## 为什么这件事有意义

现在大模型推理领域，最成熟的生态当然还是 NVIDIA CUDA。

但如果所有工程能力都停留在 CUDA 上，那么国产 GPU 永远只能等别人适配。

真正有价值的路线应该是：

- 不只是跑通官方样例。
- 不只是用现成框架测一个 benchmark。
- 而是把自己的推理引擎、自己的 kernel、自己的模型加载路径迁移过去。
- 然后看它哪里能跑、哪里慢、哪里需要重写。

这次摩尔线程 MTT S4000 的适配，就是沿着这个思路做的。

它不是一次“换个设备跑 Python”的实验，而是一次从底层动态库开始的推理后端迁移。

## 我的推理引擎做了什么

这个项目是一个从零实现的单 batch LLM 推理引擎，目标模型是 DeepSeek-R1-Distill-Qwen-7B。

当前支持的是 HuggingFace safetensors 模型目录，不依赖 GGUF。

核心路径包括：

- safetensors 权重加载
- tokenizer 由 Python 负责
- C++ 动态库负责模型加载和推理
- GPU kernel 负责 RMSNorm、RoPE、GQA Attention、SwiGLU、KV cache、decode
- Python 通过动态库调用 C++ 推理入口

换句话说，它不是 PyTorch 推理，也不是 vLLM，也不是 llama.cpp。

它更像一个用来研究 LLM decode 路径的“透明推理引擎”。

这类项目的价值在于：每一层开销都能看见，每一个 kernel 都能改，每一次优化都可以被日志验证。

## 从 CUDA 到 MUSA：适配摩尔线程

摩尔线程使用的是 MUSA 编程栈，不是 NVIDIA CUDA。

这意味着不能简单假设所有 CUDA 生态里的东西都能直接使用。

这次适配做了几个关键处理：

```text
CUDA build output: build/libllm_cuda.so
MUSA build output: build/libllm_musa.so
```

MUSA 路径使用：

```text
mcc -x musa -mtgpu
```

并使用摩尔线程原生头文件：

```text
musa_runtime.h
musa_fp16.h
```

同时把项目里少量 CUDA runtime 调用映射到 MUSA runtime：

```text
cudaMalloc              -> musaMalloc
cudaMemcpy              -> musaMemcpy
cudaMemset              -> musaMemset
cudaDeviceSynchronize   -> musaDeviceSynchronize
```

这样可以最大限度保留原来的 kernel 结构，同时让工程先跑进 MUSA 编译和运行路径。

## 测试环境

测试机器来自 AutoDL 的摩尔线程环境。

设备信息：

```text
GPU: MTT S4000
Memory: 49152 MiB
Driver Version: 2.7.0
MUSA Toolkit: 3.1.0
mcc: /usr/local/musa/bin/mcc
```

设备检查命令：

```bash
mthreads-gmi
musa_version_query
ls /usr/local/musa
```

看到 `MTT S4000`、MUSA toolkit 和 `mcc` 后，就可以开始构建。

## 模型下载：使用国内源

模型使用 DeepSeek-R1-Distill-Qwen-7B 的 HuggingFace safetensors 权重。

为了在国内机器上下载更稳定，我加了一个下载脚本：

```bash
pip install -U modelscope

python download_model.py \
  --source modelscope \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir /root/autodl-tmp/deepseek-r1-7b
```

它默认走 ModelScope，非常适合 AutoDL、算力云、边缘云这类国内环境。

下载完成后，目录里应该能看到：

```text
config.json
model.safetensors.index.json
model-00001-of-000002.safetensors
model-00002-of-000002.safetensors
tokenizer.json
tokenizer_config.json
```

## 编译 MUSA 动态库

在摩尔线程机器上执行：

```bash
export PATH=/usr/local/musa/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH

make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-musa
```

编译成功后会生成：

```text
build/libllm_musa.so
```

这个动态库就是 MUSA 后端版本。

## 启动推理

运行命令：

```bash
MUSA_VISIBLE_DEVICES=0 python python_infer.py \
  --model /root/autodl-tmp/deepseek-r1-7b \
  --lib ./build/libllm_musa.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 128 \
  --max-seq 800
```

同时可以在另一个终端观察 GPU：

```bash
watch -n 1 mthreads-gmi
```

如果显存占用和 GPU 利用率发生变化，说明程序确实进入了 MUSA 设备路径。

## 实测结果

一次典型日志如下：

```text
[C++][time] prefill total_ms=667.903, tokens=13, tokens_per_s=19.4639
[C++][time] decode step_ms=51.1793, sample_ms=0.158158, forward_ms=51.0141, decode_tokens=1, decode_tokens_per_s=19.5391
[C++][time] decode step_ms=52.3254, sample_ms=0.128011, forward_ms=52.1905, decode_tokens=68, decode_tokens_per_s=19.2803
[C++][time] decode step_ms=53.9144, sample_ms=0.127018, forward_ms=53.7796, decode_tokens=127, decode_tokens_per_s=19.0851
```

结果可以概括为：

```text
Prefill: about 19 tok/s
Decode: about 19 tok/s
Forward latency: about 51-54 ms/token
```

这说明当前版本已经完成了完整推理闭环：

- 能加载 DeepSeek-R1-Distill-Qwen-7B safetensors 权重
- 能在 MUSA 后端完成 prefill
- 能逐 token decode
- 能输出推理结果
- 能记录每一步 forward 和 sample 耗时

## 这个速度怎么看

19 tokens/s 不是一个可以拿来“秒杀”的数字。

但它是一个很关键的起点。

因为当前这版是兼容优先的 FP16 路径，还没有针对 MTT S4000 的体系结构做深度 kernel 重写。

这里面有几个现实限制：

- 原始代码设计主要面向 CUDA。
- MUSA 虽然语法兼容度较高，但底层指令映射和 kernel 调度不等同于 NVIDIA。
- WMMA 路径在 MUSA 构建里关闭了，因为 `nvcuda::wmma` 是 NVIDIA 专用路径。
- INT4 + DP4A 路线已经尝试过，但在当前 MUSA 路径上没有明显提速，说明不能简单照搬 Jetson/Orin 上的 CUDA 优化经验。

换句话说：

> 能跑通只是第一步，真正的优化必须面向摩尔线程自己的硬件和 MUSA 编译器特性重新设计。

## 我觉得最有价值的发现

这次实验最大的价值，不是某个单点速度，而是验证了国产 GPU LLM 适配的一条工程路径：

```text
自研推理引擎
  -> CUDA-style kernel source
  -> MUSA runtime 映射
  -> mcc 原生编译
  -> safetensors 权重加载
  -> DeepSeek 7B 推理闭环
  -> timing log 定位性能瓶颈
```

这条路径跑通后，后面就可以继续做真正有意义的事情：

- 重写 GEMV / Linear kernel
- 面向 MUSA 做 INT8 / INT4 量化 kernel
- 分析 attention 和 MLP 的耗时占比
- 尝试 MUSA 原生矩阵库
- 减少 kernel launch 和同步开销
- 做更细粒度 profiling
- 对比 llama.cpp MUSA backend

这比直接跑一个黑盒框架更适合做底层优化。

## 为什么我觉得这值得继续做

国产 GPU 要进入大模型推理，不可能只靠宣传参数。

最终要看的还是三个问题：

1. 能不能编译真实项目？
2. 能不能加载真实模型？
3. 能不能跑出可分析的真实性能日志？

这次 MTT S4000 + MUSA 的实验，至少把这三个问题都打通了。

这已经不是“理论支持”，而是“工程上跑起来了”。

下一步要追求的就不是兼容性，而是性能。

## 当前项目状态

项目地址：

[https://github.com/luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)

当前 `moore` 分支包含：

- MUSA 编译目标：`make -f Makefile.cuda_lib lib-musa`
- 模型下载脚本：`download_model.py`
- 推理入口：`python_infer.py`
- MUSA 文档：`MOORE_MUSA.md`
- README 中的摩尔线程运行说明和实测结果

如果你也想复现，可以直接按下面流程：

```bash
git clone -b moore https://github.com/luogantt/LLM-inference-engine.git
cd LLM-inference-engine

pip install -U modelscope
python download_model.py --local-dir /root/autodl-tmp/deepseek-r1-7b

export PATH=/usr/local/musa/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH

make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-musa

MUSA_VISIBLE_DEVICES=0 python python_infer.py \
  --model /root/autodl-tmp/deepseek-r1-7b \
  --lib ./build/libllm_musa.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 128 \
  --max-seq 800
```

## 最后

这篇文章想表达的不是“摩尔线程已经把 LLM 推理做到极致”。

更准确地说，是：

> 摩尔线程 MTT S4000 已经可以承载真实 LLM 推理工程的迁移实验；如果愿意深入 kernel 和后端优化，它有继续挖掘的空间。

对我来说，这就是最值得兴奋的地方。

不是因为它已经完美，而是因为它终于可以被认真优化了。
