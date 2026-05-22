# 对标 MLC：一个手写 CUDA 推理引擎在 Jetson Orin AGX 上跑到 19.6 tok/s

项目地址：[https://github.com/luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)

这是一篇关于端侧大模型推理优化的小结。

我做了一个从零实现的 C++ / CUDA LLM 推理引擎，当前主要面向 DeepSeek-R1-Distill-Qwen-7B 的单 batch decode。最近在 Jetson Orin AGX 上完成了 INT4 DP4A 路径，最终达到约 **19.6 tok/s**。

这个速度已经可以拿来和 MLC LLM 在 Jetson Orin 系列上的主流 INT4 公开结果做对照。

## 为什么对标 MLC

MLC LLM 是端侧大模型推理里非常重要的一条路线。

它的核心优势是：

- 基于 machine learning compilation
- 支持多后端部署
- 支持 CUDA、Metal、Vulkan、WebGPU 等平台
- 能覆盖桌面 GPU、Jetson、手机、浏览器等场景
- 有完整的模型编译和 runtime 体系

简单说，MLC 是“通用编译器 + 跨平台 runtime”的路线。

而我的项目是另一种路线：

- 不做通用编译器
- 不追求跨所有平台
- 直接面向 Jetson Orin AGX / CUDA
- 手写 decode 关键路径
- 专注单 batch、本地推理、低延迟

所以这不是“替代 MLC”，而是一个更轻、更专、更适合研究 CUDA decode 细节的实现。

## 测试环境

```text
Device: Jetson Orin AGX
Model: DeepSeek-R1-Distill-Qwen-7B
CUDA arch: sm_87
batch: 1
max_seq: 800
max_new_tokens: 512
```

编译命令：

```bash
make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-int4 A=sm_87
```

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 python python_infer.py \
  --model /data/project/deepseek-r1-7b \
  --lib ./build/libllm_cuda.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 512 \
  --max-seq 800
```

## 当前结果

最终 INT4 DP4A 路径的 decode 数据：

```text
forward_ms ≈ 52.7-53.3 ms
decode speed ≈ 19.6 tok/s
```

不同版本对比：

| 版本 | 计算路径 | decode speed |
|---|---|---:|
| FP16 baseline | FP16 weight + float activation | 约 8.6 tok/s |
| Weight-only INT8 | INT8 weight + float activation | 约 14.0 tok/s |
| 旧 INT4 | INT4 weight + float 解包 | 约 9.3 tok/s |
| 新 INT4 DP4A | INT4 weight + INT8 activation + DP4A | 约 19.6 tok/s |

这个结果说明，INT4 真正快起来的关键不是只把权重压到 4bit，而是让计算路径也进入整数点积。

## 和 MLC 的公开数据对比

公开资料里，MLC 在 Jetson Orin 系列上的 7B / 8B INT4 模型速度大概处在 19-22 tok/s 这个区间。

例如 NVIDIA Jetson AI Lab 的 benchmark 页面中，Orin Nano Super 上 Qwen2.5 7B 的 MLC API 结果约为 **21.75 tok/s**，Llama 3.1 8B 的 MLC API 结果约为 **19.10 tok/s**。

社区中也有 MLC 在 Jetson AGX Orin 上运行 Llama2-13B 4bit 约 **20.4 tok/s** 的记录。

把这些数据放在一起看：

| 项目 / 框架 | 设备 | 模型 | 速度 |
|---|---|---|---:|
| MLC API | Orin Nano Super | Qwen2.5 7B INT4 | 约 21.75 tok/s |
| MLC API | Orin Nano Super | Llama 3.1 8B INT4 | 约 19.10 tok/s |
| MLC LLM 社区记录 | Jetson AGX Orin | Llama2-13B 4bit | 约 20.4 tok/s |
| 本项目 | Jetson Orin AGX | DeepSeek-R1-Distill-Qwen-7B INT4 DP4A | 约 19.6 tok/s |

这不是严格的同模型、同 prompt、同上下文长度 benchmark，所以不能说“全面超过 MLC”。

但可以比较稳地说：

```text
在 Jetson Orin AGX 单 batch decode 场景下，
这个手写 CUDA INT4 DP4A 引擎已经接近 MLC 在 Jetson Orin 系列上的主流公开速度区间。
```

## 最关键的一次踩坑

最开始我也做了 INT4，但速度没有起来。

第一版 INT4 是这样的：

```text
weight: INT4 packed
activation: float
compute: 解包 int4 -> float -> fmaf
```

结果只有：

```text
约 9.3 tok/s
```

甚至明显慢于 INT8 的 14 tok/s。

原因很直接：这种 INT4 只是减少了权重带宽，但每个权重都要运行时做：

```text
shift
mask
sign extend
convert to float
float fmaf
```

这些额外操作把 INT4 节省下来的带宽吃掉了。

所以只做 weight-only INT4，并不等于真正的 INT4 高性能推理。

## 正确路线：INT4 + INT8 activation + DP4A

后面改成了新的路径：

```text
weight: INT4 packed
activation: dynamic INT8
accumulate: int32
compute: __dp4a
output: float
```

也就是：

```text
INT4 权重在寄存器中解包成 int8 lane
float activation 先动态量化成 int8
然后用 __dp4a 做 int8 x int8 -> int32 accumulate
最后乘 weight_scale * activation_scale
```

这条路径把速度从约 9.3 tok/s 提升到约 19.6 tok/s。

核心经验就是：

```text
INT4 必须配合 int8 activation + DP4A，不能走 float 解包。
```

## 我的项目和 MLC 的区别

MLC 的强项：

- 通用编译器体系
- 跨平台部署
- 支持更多模型和后端
- 工程生态完整
- 适合真实产品部署

我的项目的强项：

- 代码路径短
- 直接手写 CUDA
- 更容易理解 decode 内部细节
- 方便做单点优化实验
- 更适合研究 Jetson Orin 上单 batch decode 的瓶颈

如果说 MLC 是一套完整的“跨平台部署系统”，那这个项目更像是一把专门用来切开 Jetson CUDA decode 路径的手术刀。

## 为什么这个项目值得关注

这个项目的意义不只是跑到了 19.6 tok/s，而是展示了一条很清晰的优化过程：

```text
FP16 baseline
-> weight-only INT8
-> weight-only INT4
-> 发现 INT4 float 解包不成立
-> INT4 weight + INT8 activation + DP4A
-> Jetson Orin AGX 上约 19.6 tok/s
```

这个过程能帮助理解几个问题：

- 单 batch decode 为什么通常是 memory-bound
- 为什么 INT4 不一定天然比 INT8 快
- 为什么量化不仅是存储格式问题，也是计算路径问题
- 为什么 DP4A 对 Jetson 端侧推理很重要
- 为什么手写 kernel 仍然有研究价值

## 后续优化方向

当前版本还不是终点。

后续还可以继续做：

- 把 activation quantize 融合进 GEMV kernel，减少额外 kernel launch
- 为 QKV / MLP / down_proj 写专用 kernel
- 优化 `lm_head`
- 用 CUDA Graph 降低 decode step 调度开销
- 尝试 group-wise scale，提高 INT4 精度
- 做更完整的 benchmark，对齐 MLC / llama.cpp / TensorRT-LLM

## 总结

MLC 是端侧 LLM 推理里非常成熟的通用编译器路线。

我的项目不是要替代 MLC，而是用一个更小、更直接的手写 CUDA 引擎，在 Jetson Orin AGX 上把单 batch INT4 decode 路径打穿。

目前结果：

```text
DeepSeek-R1-Distill-Qwen-7B
Jetson Orin AGX
INT4 weight + INT8 activation + DP4A
约 19.6 tok/s
```

这个速度已经接近 MLC 在 Jetson Orin 系列上的 7B / 8B INT4 公开速度区间。

项目地址：

[https://github.com/luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)

## 参考链接

- MLC LLM 官方介绍：[https://blog.mlc.ai/2024/06/07/universal-LLM-deployment-engine-with-ML-compilation](https://blog.mlc.ai/2024/06/07/universal-LLM-deployment-engine-with-ML-compilation)
- NVIDIA Jetson AI Lab Benchmarks：[https://tokk-nv.github.io/jetson-generative-ai-playground/benchmarks.html](https://tokk-nv.github.io/jetson-generative-ai-playground/benchmarks.html)
- Jetson Orin Nano Super MLC benchmark 讨论：[https://forums.developer.nvidia.com/t/jetson-orin-nano-super-performance-test-issue/331700](https://forums.developer.nvidia.com/t/jetson-orin-nano-super-performance-test-issue/331700)
- MLC / Jetson AGX Orin 社区记录：[https://www.reddit.com/r/LocalLLaMA/comments/173lroh](https://www.reddit.com/r/LocalLLaMA/comments/173lroh)
