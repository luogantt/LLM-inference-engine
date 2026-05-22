# Jetson Orin AGX INT4 推理优化实践：super 分支从 9 tok/s 到 24 tok/s

项目地址：[https://github.com/luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)

本文总结 `jetson-orin-agx-super` 分支上的一次端侧大模型推理优化实践。目标设备是 Jetson Orin AGX，目标模型是 DeepSeek-R1-Distill-Qwen-7B，目标场景是单 batch、单 token decode。

这次优化的核心结论很明确：

> INT4 不是只把权重压成 4 bit 就会自动变快。  
> 在 Jetson Orin AGX 上，INT4 要真正加速，必须配合 INT8 activation 和 DP4A 整数点积，不能走 float 解包。

最终保留的最快版本是 `lib-int4-o4-all`。在实际 decode 日志里，速度推进到约 **24 tokens/s**，单 token forward 延迟约 **43 ms**。

## 测试环境

```text
Device: Jetson Orin AGX
CUDA arch: sm_87
Model: DeepSeek-R1-Distill-Qwen-7B
Branch: jetson-orin-agx-super
batch: 1
max_seq: 800
max_new_tokens: 512
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

当前推荐编译命令：

```bash
make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-int4-o4-all A=sm_87
```

## 模型尺寸和 decode 的真实瓶颈

当前代码里的关键模型尺寸为：

```text
N_LAYERS     = 28
HIDDEN       = 3584
KV_DIM       = 512
INTERMEDIATE = 18944
VOCAB_SIZE   = 152064
```

在单 token decode 阶段，每一步只处理一个新的 token。此时最主要的计算不是大 batch GEMM，而是大量 GEMV：

```text
matrix weight x vector activation
```

也就是：

```math
y = W x + b
```

对第 `j` 个输出通道：

```math
y_j = \sum_{i=0}^{H-1} W_{j,i} x_i + b_j
```

这里 `H = 3584`。每个输出行都要和长度为 3584 的 hidden vector 做一次点积。

在每一层里，主要 linear 包括：

```text
Q projection:       HIDDEN x HIDDEN
K projection:       KV_DIM x HIDDEN
V projection:       KV_DIM x HIDDEN
O projection:       HIDDEN x HIDDEN
Gate projection:    INTERMEDIATE x HIDDEN
Up projection:      INTERMEDIATE x HIDDEN
Down projection:    HIDDEN x INTERMEDIATE
```

其中 MLP 的 gate/up/down 计算量很大，QKV projection 和普通 linear 也会在每个 decode step 反复出现。只优化某一个 linear，整体速度提升有限。`super` 分支里真正有效的版本，是把普通 linear、QKV、gate/up 这些主路径都切到 INT4 + INT8 activation + DP4A。

## 从 FP 线性层到 INT4 DP4A 的数学推导

原始 float 或 half 线性层为：

```math
y_j = \sum_i W_{j,i} x_i + b_j
```

如果做 weight-only INT4，通常对每个输出行保存一个 scale：

```math
W_{j,i} \approx s^W_j q^W_{j,i}
```

其中：

```math
q^W_{j,i} \in [-8, 7]
```

量化过程可以写成：

```math
q^W_{j,i} = \operatorname{clip}
\left(
  \operatorname{round}\left(\frac{W_{j,i}}{s^W_j}\right),
  -8,
  7
\right)
```

如果 activation 仍然保持 float，那么计算会变成：

```math
y_j \approx \sum_i s^W_j q^W_{j,i} x_i + b_j
```

这条路径看似使用了 INT4 权重，但每个权重在计算时仍然要：

```text
load packed int4
unpack nibble
sign extend
convert to float
float multiply-add
```

所以它只是减少了权重带宽，没有把计算本身切到整数点积。早期 INT4 版本速度不理想，根本原因就在这里。

要让 INT4 真正加速，需要把 activation 也量化成 INT8：

```math
x_i \approx s^x q^x_i
```

其中：

```math
q^x_i \in [-127, 127]
```

单 token 动态 activation 量化为：

```math
s^x = \frac{\max_i |x_i|}{127}
```

```math
q^x_i = \operatorname{clip}
\left(
  \operatorname{round}\left(\frac{x_i}{s^x}\right),
  -127,
  127
\right)
```

代回线性层：

```math
y_j
\approx
\sum_i
\left(s^W_j q^W_{j,i}\right)
\left(s^x q^x_i\right)
+ b_j
```

把 scale 提出来：

```math
y_j
\approx
s^W_j s^x
\sum_i q^W_{j,i} q^x_i
+ b_j
```

中间累加项是一个整数点积：

```math
acc_j = \sum_i q^W_{j,i} q^x_i
```

最终反量化：

```math
y_j \approx s^W_j s^x acc_j + b_j
```

这就是 `super` 分支 INT4 DP4A 路径的数学本质。

## INT32 accumulator 是否安全

对当前 hidden size：

```text
H = 3584
```

最坏情况下：

```math
|q^W_{j,i}| \le 8
```

```math
|q^x_i| \le 127
```

单项乘积最大约为：

```math
8 \times 127 = 1016
```

一个输出行的最坏累加绝对值上界为：

```math
3584 \times 1016 = 3,641,344
```

这个值远小于 int32 的范围：

```math
2^{31} - 1 = 2,147,483,647
```

所以在当前模型尺寸下，用 int32 accumulator 保存 INT4 x INT8 点积是安全的。

## DP4A 做了什么

NVIDIA GPU 的 DP4A 指令可以在一条指令中完成 4 组 int8 乘加：

```math
acc \leftarrow acc
+ a_0 b_0
+ a_1 b_1
+ a_2 b_2
+ a_3 b_3
```

其中 `a_k` 和 `b_k` 都是 int8。

对于 INT4 权重，存储时一个 byte 可以放 2 个权重，一个 `uint32_t` 可以放 8 个 INT4 权重：

```text
uint32 packed = [w7 w6 w5 w4 w3 w2 w1 w0]
```

计算时可以把 8 个 INT4 权重拆成两组 int8x4：

```text
[w0, w1, w2, w3] -> int8x4
[w4, w5, w6, w7] -> int8x4
```

activation 已经是 INT8，连续 8 个 activation 可以看作两组 int8x4：

```text
[x0, x1, x2, x3] -> int8x4
[x4, x5, x6, x7] -> int8x4
```

于是 8 个权重和 8 个 activation 的点积，可以用两次 DP4A 完成：

```math
acc \leftarrow acc + \operatorname{DP4A}(w_{0:3}, x_{0:3})
```

```math
acc \leftarrow acc + \operatorname{DP4A}(w_{4:7}, x_{4:7})
```

这条路径避免了逐元素 float 解包和 float FMA，把核心计算变成整数指令。

## 为什么 INT4 float 解包不快

INT4 的理论带宽优势很明显。以一个输出行为例，`H = 3584`：

| 权重格式 | 每个权重字节数 | 单输出行权重读取 |
|---|---:|---:|
| FP16 | 2 bytes | 7168 bytes |
| INT8 | 1 byte | 3584 bytes |
| INT4 | 0.5 byte | 1792 bytes |

INT4 相比 FP16，权重读取量变成 1/4。相比 INT8，权重读取量变成 1/2。

但如果 INT4 每个元素都走：

```text
unpack -> sign extend -> convert float -> fmaf
```

那么额外指令会吃掉带宽收益。实际日志也验证了这一点：

| 版本 | 计算路径 | 实测速度 |
|---|---|---:|
| Weight-only INT8 | INT8 weight + float/普通路径 | 约 14 tok/s |
| 早期 INT4 | INT4 weight + float 解包 | 约 9 tok/s |
| INT4 DP4A | INT4 weight + INT8 activation + DP4A | 约 20 tok/s 以上 |

所以 INT4 的关键不是“存得小”，而是“算得对”。在 Orin AGX 上，必须让 INT4 权重进入整数点积路径。

## super 分支的优化路线

这次 `jetson-orin-agx-super` 分支主要经历了几轮：

| 版本 | 核心思路 | 实测表现 |
|---|---|---:|
| 初始 INT4 | INT4 存储，但计算路径不够整数化 | 约 9 tok/s |
| INT4 + DP4A | activation INT8，权重 INT4，整数点积 | 约 20 tok/s |
| `lib-int4-o2-all` | 一个 block 同时算 2 个输出行，覆盖普通 linear、QKV、gate/up | 约 22.5 tok/s |
| `lib-int4-o4-all` | 一个 block 同时算 4 个输出行，继续提高 activation 复用 | 约 24 tok/s |
| `lib-int4-o8-all` | 一个 block 同时算 8 个输出行 | 掉到约 18 tok/s，已回滚 |

最终保留的是 `lib-int4-o4-all`。

## o2-all 和 o4-all 为什么能加速

原始做法可以理解为：一个 block 只算一个输出行。

```text
block 0 -> y0
block 1 -> y1
block 2 -> y2
...
```

每个 block 都要读取同一份 activation vector `x`，只是读取的权重 row 不同。

对 GEMV 来说，activation 是所有输出行共享的：

```math
y_j = \sum_i W_{j,i} x_i
```

这里的 `x_i` 对所有 `j` 都相同。于是可以让一个 block 同时算多个输出行：

```text
block 0 -> y0, y1, y2, y3
block 1 -> y4, y5, y6, y7
...
```

对 4-output 版本，一个 block 内维护 4 个 accumulator：

```math
acc_0 = \sum_i q^W_{0,i} q^x_i
```

```math
acc_1 = \sum_i q^W_{1,i} q^x_i
```

```math
acc_2 = \sum_i q^W_{2,i} q^x_i
```

```math
acc_3 = \sum_i q^W_{3,i} q^x_i
```

每次读取一组 activation 后，可以同时喂给 4 个权重 row：

```text
load x int8x4
load row0 int4x4 -> dp4a -> acc0
load row1 int4x4 -> dp4a -> acc1
load row2 int4x4 -> dp4a -> acc2
load row3 int4x4 -> dp4a -> acc3
```

这样做有几个好处：

1. block 数量减少，调度开销下降。
2. activation 读取被多个输出行复用。
3. 每个 block 做的工作更饱满。
4. 仍然只维护 4 个主要 accumulator，寄存器压力可控。

可以用一个简化成本模型理解：

```math
T(r)
\approx
T_{launch/block}(r)
+ T_{weight}
+ T_{activation/reuse}(r)
+ T_{reduction}(r)
+ T_{register/occupancy}(r)
```

其中 `r` 表示一个 block 同时计算的输出行数。

当 `r` 从 1 增加到 2、4：

```text
block 数量下降
activation 复用提高
整体吞吐提高
```

但当 `r` 继续增加到 8：

```text
每个线程 accumulator 变多
row pointer 和 scale pointer 变多
寄存器使用变多
shared memory reduction 变重
occupancy 下降
```

所以 `r` 不是越大越好。对 Jetson Orin AGX 和当前 hidden size 来说，`r = 4` 是这次实测中最平衡的点。

## 为什么 o8-all 失败并回滚

`lib-int4-o8-all` 的想法很自然：既然 4-output 更快，那 8-output 会不会更快？

实测结果是否定的。`o8-all` 的 decode 速度掉到了约 **18 tok/s**：

```text
forward_ms ≈ 57.6 ms
decode_tokens_per_s ≈ 18.0 tok/s
```

这说明瓶颈已经从 block 调度和 activation 复用，转移到了寄存器压力、occupancy 和 reduction 成本。

8-output kernel 里每个线程需要同时维护：

```text
8 个 accumulator
8 个 row pointer
更多 scale/local/output 指针
更多写回分支
更多 shared memory reduction 数据
```

这些都会降低 SM 上可同时驻留的 block 数量。对 Orin AGX 这种端侧 GPU 来说，occupancy 一旦下降，整数 DP4A 指令也喂不满，最后性能反而下降。

所以 `o8-all` 被回滚，当前 `super` 分支保留 `o4-all` 作为推荐路径。

## 实测结果

`lib-int4-o2-all` 的一次记录：

```text
forward_ms ≈ 46.4326
decode_tokens = 474
decode_tokens_per_s ≈ 22.5382
```

`lib-int4-o4-all` 的一次记录：

```text
forward_ms ≈ 43.7584
decode_tokens = 474
decode_tokens_per_s ≈ 23.9898
```

`lib-int4-o8-all` 的一次记录：

```text
forward_ms ≈ 57.5994
decode_tokens = 474
decode_tokens_per_s ≈ 18.0096
```

对比可以看到：

```math
\frac{23.99}{22.54} \approx 1.064
```

`o4-all` 相比 `o2-all` 继续提升约 6.4%。

而 `o8-all` 相比 `o4-all`：

```math
\frac{18.01}{23.99} \approx 0.751
```

也就是掉了约 25%。这说明 `o4-all` 已经接近当前 kernel 结构下的甜点区间。

## 与主流端侧推理引擎的关系

MLC、llama.cpp、TensorRT-LLM 等主流推理引擎都有更完整的工程体系，例如模型转换、图优化、跨平台 runtime、更多量化格式和更成熟的算子调度。

这个项目的目标不是替代它们，而是做一条更透明、更直接的 CUDA decode 优化路线：

```text
不依赖 PyTorch 推理
不依赖大型 runtime
直接手写 C++ / CUDA decode 路径
针对 Jetson Orin AGX 的单 batch 场景优化
```

这次 `super` 分支的意义在于，它证明了一个小型手写 CUDA 推理引擎，只要抓住端侧 decode 的真实瓶颈，也可以把 7B 模型推到和主流端侧引擎同量级的速度区间。

更重要的是，这个过程把 INT4 加速的关键讲清楚了：

```text
INT4 weight-only compression 只解决存储和带宽问题
INT8 activation quantization 让计算进入整数域
DP4A 让整数点积真正被硬件高效执行
4-output GEMV layout 在复用和 occupancy 之间取得平衡
```

## 当前推荐使用方式

切换到 `jetson-orin-agx-super` 分支后：

```bash
git pull origin jetson-orin-agx-super
```

编译：

```bash
make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-int4-o4-all A=sm_87
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

## 后续还能继续优化什么

当前 `o4-all` 已经是这轮实验里最好的版本，但后面仍然有一些方向可以继续尝试。

### 1. 更精细的 kernel fusion

现在已经优化了多个 linear 的 INT4 DP4A 路径，但 RMSNorm、量化、linear、SwiGLU、residual 之间仍然存在 kernel 边界。后续可以继续研究是否能减少中间写回。

### 2. activation quantization 优化

当前 activation 每步动态量化：

```math
s^x = \frac{\max_i |x_i|}{127}
```

这一步需要先求 max，再写出 int8 activation。后续可以研究更快的归约、近似 scale、或者和前一个算子融合。

### 3. KV cache 访存优化

decode 越往后，attention 对 KV cache 的读取越重。当前 `max_seq=800` 下，后段 token 的 forward_ms 会逐步上升，说明 KV cache 和 attention 访存仍然值得优化。

### 4. 针对固定尺寸生成专用 kernel

当前模型尺寸固定：

```text
HIDDEN = 3584
INTERMEDIATE = 18944
KV_DIM = 512
```

可以为这些尺寸生成更激进的专用 kernel，减少通用分支和边界判断。

### 5. 更严格的同条件 benchmark

后续如果要和 MLC、llama.cpp 等引擎对标，需要统一：

```text
同一模型
同一量化方式
同一 prompt
同一 max_seq
同一 max_new_tokens
同一 Jetson 电源模式和频率设置
同一 prefill/decode 统计口径
```

只有这样，速度对比才足够严谨。

## 总结

`jetson-orin-agx-super` 分支这次实践说明：

1. INT4 不等于自动加速。
2. INT4 如果走 float 解包，会浪费掉 4 bit 权重的优势。
3. 真正有效的路径是 INT4 权重、INT8 activation、DP4A 整数点积。
4. 单 token decode 的核心瓶颈是 GEMV，不是大 batch GEMM。
5. 一个 block 同时算 4 个输出行，是当前 Jetson Orin AGX 上更合理的平衡点。
6. 更激进的 8-output kernel 会因为寄存器压力和 occupancy 下降而变慢。

最终，`lib-int4-o4-all` 把 DeepSeek-R1-Distill-Qwen-7B 在 Jetson Orin AGX 上的 decode 速度推进到约 **24 tokens/s**。

这不是靠框架黑盒得到的结果，而是从线性层数学公式、量化公式、DP4A 指令，到 GEMV kernel layout 一步步压出来的结果。

这也是这个项目最有价值的地方：它把端侧 LLM 推理的性能问题拆开，让每一次优化都能被解释、被验证、被继续推进。
