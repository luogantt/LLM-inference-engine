# Ascend 47-50 tok/s 优化原因总结

本文总结此前在 Ascend 芯片上把 DeepSeek-R1-Distill-Qwen-7B 推理速度优化到约 47-50 tok/s 的主要原因。

## 一句话结论

Ascend 47-50 tok/s 的核心原因不是芯片天然更强，而是当时的场景非常适合专用优化：

```text
DeepSeek-R1-Distill-Qwen-7B dense 模型
+ 单 batch decode
+ 自研 direct runtime
+ KV cache
+ 权重常驻
+ QKV / MLP / LM Head 融合
+ 低框架调度开销
```

它和现在 A800 跑 DeepSeek-V4-Flash 不能直接横向比较，因为后者是大 MoE + FP4/FP8 低比特模型，而 A800 没有原生 FP4/FP8 Tensor Core 路径。

## 主要直接优化

### 1. 绕过 torch_npu / Transformers 高层路径

当时使用自研 direct `.so` 跑：

```bash
ASCEND_DIRECT_DECODE=all_layers_ref
```

这样减少了 Python、Transformers、torch_npu 框架调度开销。单 token decode 最怕每层都经过高层框架调度，这一项收益最大。

### 2. KV Cache 生效

```bash
ASCEND_REF_KV_CACHE=1
```

decode 每一步只处理新 token，不重复计算历史上下文。

### 3. 权重常驻和权重缓存

```bash
ASCEND_LOAD_WEIGHTS=all
ASCEND_REF_CACHE_WEIGHTS=1
```

权重预加载并缓存，避免 decode 时反复查找、转换、搬运。7B 单 batch decode 很多时候是权重访问瓶颈。

### 4. QKV 融合

```bash
ASCEND_QKV_BACKEND=aclnn
ASCEND_QKV_FUSE_WEIGHTS=1
```

把 Q/K/V 三路线性层融合成一次矩阵计算，减少 kernel 调用和权重读取。

### 5. MLP gate/up 融合

```bash
ASCEND_MLP_BACKEND=aclnn
ASCEND_MLP_FUSE_GATE_UP=1
```

把 MLP 的 gate/up 两路投影合并，减少一次大线性层调度和访存。

### 6. Attention projection 和 LM Head 上 ACLNN

```bash
ASCEND_ATTN_PROJ_BACKEND=aclnn
ASCEND_LM_HEAD_BACKEND=aclnn
```

尤其 LM Head 收益明显，后期日志里 LM Head 大约可以压到 1.1ms 左右。

### 7. CPU reference 小 batch 路径线程优化

小 batch decode 下，部分 reference dot / attention 路径走 CPU 更稳定，因此用了：

```bash
ASCEND_REF_FAST_DOT=1
ASCEND_REF_NEON_DOT=1
ASCEND_REF_LINEAR_THREADS=16
ASCEND_REF_ATTN_LINEAR_THREADS=16
ASCEND_REF_MLP_THREADS=24
ASCEND_REF_DOWN_THREADS=24
```

### 8. 关闭 fallback、profile 和详细日志

```bash
ASCEND_QKV_FALLBACK=0
ASCEND_MLP_FALLBACK=0
ASCEND_TIME_LOG_FILE=0
ASCEND_REF_PROFILE_LAYERS=0
```

避免性能测试时被日志、profile、异常 fallback 路径拖慢。

## 其他隐性原因

### 1. 模型结构简单很多

当时是 7B dense Qwen 结构，没有 MoE routing、expert dispatch、top-k expert、跨 expert 累加这些碎操作。dense 模型 decode 热路径更直，更容易融合。

### 2. 参数规模小很多

7B 每 token 需要读取的权重远少于 DeepSeek-V4-Flash。单 batch decode 常常是权重带宽瓶颈，模型越小越容易跑快。

### 3. 通信路径更短

Ascend 那次主要是更短的单设备或少通信路径，不像现在 4 卡 A800 TP 路径有 all-reduce、all-gather、expert 分片同步等开销。

### 4. 没有 FP4 unpack 税

7B 权重路径更直接，而 DeepSeek-V4-Flash 在 A800 上需要：

```text
packed FP4 -> unpack -> scale -> BF16/FP16 -> GEMM/GEMV
```

A800 没有原生 FP4 Tensor Core，这一步会带来额外访存和转换成本。

### 5. max_seq 更小

Ascend 测试常用 `max_seq=800`，现在 DeepSeek-V4-Flash 常用 `max_seq_len=4096`。即使有 KV cache，attention/cache 管理和状态维护仍然更重。

### 6. batch=1 专用化更彻底

Ascend direct path 很多地方就是围绕 batch=1、greedy decode 写死优化；DeepSeek-V4-Flash 当前还保留不少官方通用 batch、MoE、distributed 框架逻辑。

### 7. 权重布局更适合当前 kernel

7B 权重可以按自研 kernel 访问方式预加载和缓存；DeepSeek-V4-Flash 的 converted shard、FP4 expert、FP8 dense 布局更复杂，不一定适合 A800 访存模式。

### 8. 测的是 decode 稳态

Ascend 47-50 tok/s 主要看 decode 稳态，不把模型加载、转换、首轮编译、复杂 warmup 算进去。DeepSeek-V4-Flash 测速更容易被初始化、TileLang 编译、首次 cache 影响。

### 9. 自研算子覆盖比例更高

7B 的核心大头就是 attention、MLP、LM Head，刚好都被优化覆盖。DeepSeek-V4-Flash 还有 MoE gate、hash routing、shared expert、FP4 expert、compressed attention/indexer 等，尚未完全进入自研 `.so`。

### 10. 目标更窄

Ascend 那次目标是把一个固定 7B dense 模型跑快；现在 A800 DeepSeek-V4-Flash 是让一个面向更高低比特硬件的模型在 A800 上兼容且尽量快，难度更高。

## 和 A800 DeepSeek-V4-Flash 的区别

| 项目 | Ascend 7B 优化 | A800 DeepSeek-V4-Flash |
| --- | --- | --- |
| 模型 | DeepSeek-R1-Distill-Qwen-7B | DeepSeek-V4-Flash |
| 结构 | Dense | MoE |
| 参数规模 | 7B | 284B total / 13B activated |
| 低比特 | 相对直接 | FP4 experts + FP8 dense |
| 硬件匹配 | 可用 ACLNN + 自研 direct path | A800 无原生 FP4/FP8 Tensor Core |
| 通信 | 更少 | 4 卡 TP + MoE 通信 |
| 优化状态 | direct path 覆盖比例高 | 仍有不少官方 PyTorch/TileLang 路径 |

## 最终总结

Ascend 47-50 tok/s 的原因可以概括为：

```text
小 dense 模型
+ 单 batch decode
+ KV cache
+ 权重常驻
+ QKV / MLP / LM Head 融合
+ direct runtime
+ 更少通信
+ 更少低比特解包成本
```

而 A800 DeepSeek-V4-Flash 当前慢，核心原因是：

```text
大 MoE 模型
+ FP4/FP8 低比特权重
+ A800 无原生 FP4/FP8 Tensor Core
+ unpack/dequant 成本
+ 多卡通信
+ expert routing 调度碎片化
+ 自研 CUDA 覆盖还不完整
```

所以这两个结果不能直接用 tok/s 横比。Ascend 那次是固定 7B dense 模型的高度专用化优化；A800 V4-Flash 是在不完全匹配低比特硬件能力的芯片上跑复杂 MoE 模型。
