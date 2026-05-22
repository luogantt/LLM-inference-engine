# Jetson Orin AGX 上的 INT4 量化推理实践：从 9 tok/s 到 19.6 tok/s

项目地址：[https://github.com/luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)

这是一个从零实现的 C++ / CUDA 大模型推理引擎，当前主要面向 DeepSeek-R1-Distill-Qwen-7B 的单 batch 推理。项目不依赖 vLLM、llama.cpp 或 PyTorch 推理框架，核心 decode 路径由 CUDA 手写实现，方便研究大模型推理中的真实瓶颈。

这次重点是 Jetson Orin AGX 上的 INT4 量化推理优化。

## 核心结论

**INT4 必须配合 int8 activation + DP4A，不能走 float 解包。**

如果只是把权重量化成 INT4，然后在 kernel 里解包成 float 再做 `fmaf`，速度不会真正起来。INT4 的优势不只是权重体积小，而是要把计算路径也切到整数点积上。

## 测试环境

```text
Device: Jetson Orin AGX
Model: DeepSeek-R1-Distill-Qwen-7B
CUDA arch: sm_87
max_seq: 800
max_new_tokens: 512
batch: 1
```

## 速度结果

| 版本 | forward_ms | decode speed |
|---|---:|---:|
| FP16 baseline | 约 117.8 ms | 约 8.6 tok/s |
| Weight-only INT8 | 约 72-74 ms | 约 14.0 tok/s |
| Weight-only INT4 + float 解包 | 约 109-111 ms | 约 9.3 tok/s |
| INT4 weight + INT8 activation + DP4A | 约 52.7-53.3 ms | 约 19.6 tok/s |

INT4 DP4A 版本相对 FP16：

```text
19.6 / 8.6 ≈ 2.28x
```

相对 weight-only INT8：

```text
19.6 / 14.0 ≈ 1.40x
```

相对旧的 INT4 float 解包路径：

```text
19.6 / 9.3 ≈ 2.1x
```

## 最初的问题

一开始的 INT4 版本只做了 weight-only INT4：

```text
FP16 weight -> row-wise INT4 weight
activation 保持 float
kernel 内逐 byte 解包 int4
int4 -> float
float FMA
```

理论上，INT4 权重只有 FP16 的四分之一大小，也只有 INT8 的一半大小，应该能减少显存带宽压力。

但实际结果并不好：

```text
INT4 float 解包版本 ≈ 9.3 tok/s
INT8 版本 ≈ 14.0 tok/s
```

也就是说，INT4 不但没有超过 INT8，只是比 FP16 略快。

## 问题原因

原因在于这条路径省了带宽，但增加了大量计算开销。

旧 INT4 kernel 每个权重大致需要做这些事情：

```text
load packed byte
shift
mask
sign extend
convert to float
float fmaf
```

这些操作发生在每个权重上。对于 7B 模型 decode 阶段的大量 GEMV 来说，这个开销非常大。

所以旧 INT4 路径的问题是：

- 权重确实变小了
- 但每个权重都要运行时解包
- 解包以后又回到 float FMA
- 没有使用整数点积指令
- 节省的显存带宽被解包和类型转换吃掉了

这就是为什么 weight-only INT4 + float 解包没有真正加速。

## 正确方向

新的 INT4 DP4A 路径改成：

```text
weight: INT4 packed
activation: 动态量化为 INT8
accumulate: int32
compute: __dp4a
output: float
```

核心变化是把计算路径从 float FMA 改成整数点积。

`__dp4a` 一次可以做 4 组 int8 乘加：

```text
int8 x int8 -> int32 accumulate
```

由于 INT4 权重本身是 4-bit，需要在寄存器里把 4 个 INT4 unpack 成 4 个 int8 lane，然后和 int8 activation 做 DP4A。

这条路径才真正符合 INT4 加速的逻辑：

```text
更少的权重带宽
+ 更紧凑的权重存储
+ 整数点积计算
= 真实加速
```

## 编译运行

切换到 INT4 分支：

```bash
git switch jetson-orin-agx-int4
git pull
```

编译 INT4 DP4A 版本：

```bash
make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-int4 A=sm_87
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python python_infer.py \
  --model /data/project/deepseek-r1-7b \
  --lib ./build/libllm_cuda.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 512 \
  --max-seq 800
```

如果需要对比旧的 float 解包 INT4 路径：

```bash
make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-int4-float A=sm_87
```

## 当前实现方式

当前项目仍然使用原始 HuggingFace safetensors 模型目录。INT4 不是读取 GPTQ/AWQ 格式，而是在加载权重时做 row-wise INT4 量化：

```text
FP16/BF16 safetensors -> row-wise INT4 packed weight
```

每一行权重保存一个 scale：

```text
weight_float ≈ int4_value * weight_scale
```

activation 在每次 linear 前动态量化为 INT8：

```text
activation_float ≈ int8_value * activation_scale
```

最终输出：

```text
output_float ≈ int32_accumulate * weight_scale * activation_scale
```

## 为什么这个结果有意义

Jetson Orin AGX 是边缘端设备，显存带宽和功耗都比数据中心 GPU 更敏感。单 batch decode 又是典型的 memory-bound 场景，大量计算以 GEMV 为主。

因此，INT4 的价值不只是省显存，更重要的是让 decode 路径更适合边缘端硬件：

- 模型权重更小
- 显存带宽压力更低
- DP4A 可以利用整数点积
- 单 token decode 延迟明显下降

在这个实验中，INT4 DP4A 让 DeepSeek-R1-Distill-Qwen-7B 在 Jetson Orin AGX 上达到约 19.6 tok/s，已经明显超过 FP16 和 weight-only INT8。

## 后续优化方向

当前版本还有继续优化空间：

- 把 activation quantize 融合进 linear kernel，减少额外 kernel launch
- 针对 `HIDDEN=3584` 和 `INTERMEDIATE=18944` 做专用 GEMV kernel
- 减少 QKV、MLP 中重复的量化开销
- 对输出层 `lm_head` 做单独优化
- 尝试 group-wise scale，提高精度和速度平衡
- 结合 CUDA Graph 降低 decode step 的调度开销

## 总结

这次实验最重要的经验是：

```text
INT4 不是只把权重压到 4bit 就会快。
INT4 必须让计算路径也进入整数点积。
INT4 必须配合 int8 activation + DP4A，不能走 float 解包。
```

项目还在继续优化中，欢迎关注和交流：

[https://github.com/luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)
