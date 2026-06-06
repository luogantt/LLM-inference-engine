# DeepSeek-V4-Flash 在 4×A800 PCIe 上推理优化实践

> **TL;DR**: 在 4 张 NVIDIA A800 80GB PCIe（无 NVLink）上推理 DeepSeek-V4-Flash (43层 MoE, 256专家)，通过一系列 CUDA kernel 和 Python 层面优化，将解码吞吐从 **5.58 tok/s 提升到 5.87 tok/s (+5.2%)**。

- **GitHub**: [luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)
- **Tag**: [`v0.2.0-a800-opt`](https://github.com/luogantt/LLM-inference-engine/releases/tag/v0.2.0-a800-opt)
- **Branch**: `cuda_A800_deepseekv4_deepapi`
- **Commit**: `bab88c8`

---

## 1. 背景

[DeepSeek-V4-Flash](https://github.com/deepseek-ai/DeepSeek-V4) 是 DeepSeek 最新的 MoE 大模型，拥有 43 层 Transformer、256 个路由专家（每 token 激活 6 个）、多头潜在注意力 (MLA) 和 Hyper-Connection (HC) 等先进架构特性。

我们的目标是在 **4 张 NVIDIA A800 80GB PCIe**（无 NVLink，PCIe 4.0 互连）上实现高效的解码推理。

### 硬件环境

| 组件 | 规格 |
|---|---|
| GPU | 4× NVIDIA A800 80GB PCIe |
| NVLink | ❌ 未启用 (PCIe 版本) |
| GPU 互连 | PIX/PXB (PCIe 桥接) |
| CUDA | 12.4 |
| PyTorch | 2.x |

### 模型架构

| 参数 | 值 |
|---|---|
| 层数 | 43 |
| 隐藏维度 | 4096 |
| 注意力头数 | 64 |
| MoE 专家总数 | 256 (每 GPU 64) |
| 每 token 激活专家 | 6 |
| KV Cache 窗口 | 128 (滑动窗口) |
| 压缩比 (compress_ratio) | 41 层有 Indexer |
| HC 倍数 | 4 |

---

## 2. 优化历程

### 2.1 基线性能

优化前的基线 commit `357c52a`（包含 BF16 top-k 专家缓存、fused HC post kernel 等已有优化）：

```
GPU 0-3: 5.686 tok/s (22.69s / 129 tokens)
GPU 3-6: 5.576 tok/s (23.13s / 129 tokens)
```

> GPU 0-3 比 GPU 3-6 快约 2%，因为 GPU 0-1 之间有 PIX（单 PCIe 桥）连接，通信延迟更低。

### 2.2 优化步骤

#### 优化 1: 禁用 BF16 专家缓存（+4.0%）

**发现**: BF16 缓存专家的权重（每个专家 ~50MB）比 FP4 压缩格式（每个专家 ~12.5MB）大 4 倍。对于单 token 解码（GEMV 操作），内存带宽是瓶颈，FP4 的紧凑表示更高效。

```python
# python_infer_deepseek_v4_flash.py
"A800_CACHE_FP4_BF16": "0",        # 禁用 BF16 缓存
"A800_USE_CUDA_BF16_TOPK_FFN": "0", # 使用 FP4 topk 路径
```

**原理**: FP4 格式虽然需要在线解压（查找表 + scale 乘法），但读取的数据量仅为 BF16 的 1/4。对于内存带宽受限的单 token 解码，FP4 的带宽优势超过了了解压的计算开销。

#### 优化 2: 融合注意力输出投影（+0.5%）

**发现**: Attention 的输出投影由两步组成：`einsum("bsgd,grd->bsgr", o, wo_a)` + `self.wo_b(o.flatten(2))`，共 2 次 kernel launch。可以预计算 `fused = wo_b @ wo_a` 合并为一次 `F.linear` 调用。

```python
# model.py - Attention._a800_out_proj()
wo_a = _dequantize_fp8_weight(self.wo_a.weight)  # FP8→BF16
wo_b = _dequantize_fp8_weight(self.wo_b.weight)
fused = torch.cat([wo_b[:,g,:] @ wo_a[g] for g in range(n_groups)], dim=1)
x = F.linear(o.reshape(1, 1, local_in), fused)  # 一次 kernel launch
```

**同时修复**: `_a800_cuda_o_proj_fused` 函数原来因为检查 `wo_a_weight.dtype != torch.bfloat16` 永远返回 None（权重存储为 FP8），现在添加了自动反量化。

#### 优化 3: 融合 Indexer full kernel（+0.5%）

**发现**: 模型的 41/43 层有 Indexer（用于选择压缩 KV 缓存的 top-k 位置），每 token 调用一次。Indexer 的 forward 包含 Q 投影 + 权重投影 + 索引评分，共 3+ 次 kernel launch。已有的 `ds_v4_indexer_full_fused_bf16` CUDA kernel 将这些融合为一次调用，但未被使用。

```python
# model.py - Indexer.forward()
if seqlen == 1 and bsz == 1:  # 单 token 解码
    index_score = _a800_cuda_indexer_full(
        x, qr, wq_b_weight, weights_w,
        freqs_cis, kv_cache, ...
    )  # 一次 kernel launch 完成 Q 投影 + 权重投影 + RoPE + 索引评分
```

#### 优化 4: FP4 CUDA kernel launch bounds（+0.8%）

**发现**: FP4 topk gate+up kernel 和 w2 kernel 使用了保守的 `__launch_bounds__`，限制了 SM 占用率。

```cpp
// deepseek_v4_a800_kernels.cu - 优化前
__global__ void fp4_topk_gate_up_fused_kernel(...)
__global__ void fp4_topk_w2_accum_f32_kernel(...)

// 优化后
__global__ __launch_bounds__(256, 4) void fp4_topk_gate_up_fused_kernel(...)
__global__ __launch_bounds__(256, 6) void fp4_topk_w2_accum_f32_kernel(...)
```

A800 有 164KB 共享内存/SM。FP4 gate+up kernel 使用 16KB 共享内存（dim=4096, fp32），优化前 `__launch_bounds__(256, 2)` 限制为 2 blocks/SM，优化后允许 4 blocks/SM，提高了 SM 占用率。

#### 优化 5: BF16 行并行 reduce（+0.3%）

```python
# python_infer_deepseek_v4_flash.py
"A800_BF16_ROW_REDUCE": "1",  # all-reduce 使用 BF16 减半通信量
```

对于 RowParallelLinear 层的 all-reduce，使用 BF16 精度而非 FP32，将通信量减半。在 PCIe 环境下，这减少了约 50% 的 all-reduce 带宽需求。

---

## 3. 尝试过但未采纳的优化

### ❌ CUDA Graphs

**问题**: 43 层中 41 层有 Indexer，其内部使用 Python int `start_pos` 计算 `cache_len = (start_pos + 1) // ratio`。CUDA Graph 在捕获时将 `start_pos` 固化为常量，replay 时无法更新。改为 tensor 操作需要大规模重构 Indexer/Compressor 代码。

**结论**: 对 DeepSeek V4 这种复杂模型，CUDA Graph 需要大量的 graph-friendly 重构，投入产出比不高。

### ❌ Tensor Cores (WMMA)

**问题**: 单 token 解码是 GEMV 操作 (M=1, N=dim, K=inter_dim)。NVIDIA Ampere WMMA 要求 M≥16，填充到 16 会浪费 15/16 的计算。且 BF16 权重（2 字节/元素）比 FP4（0.5 字节/元素）大 4 倍，抵消了 tensor core 的加速。

**结论**: Tensor core 在单 token 解码场景下不可行，更适合 prefill 阶段（多 token）或训练。

### ❌ FP4 TILE 增大 (8→16)

虽然将每个 block 处理的输出元素翻倍（grid size 减半），但总内存流量不变，实际性能无变化。

### ❌ Indexer 结果缓存

尝试在相邻 token 间复用 Indexer 的 top-k 结果，因为压缩 KV 缓存每 4 个 token 才增长一次。但测试显示模型输出质量严重退化——滑动窗口注意力的位置每 token 都在变化，Indexer 结果必须逐 token 重新计算。

---

## 4. 性能总结

| 优化 | 贡献 | 累积 tok/s |
|---|---|---|
| 基线 (HEAD `357c52a`, GPU 3-6) | — | 5.576 |
| + 禁用 BF16 cache | +4.0% | 5.800 |
| + Fused attn out proj | +0.5% | 5.827 |
| + Fused Indexer full | +0.5% | 5.848 |
| + FP4 launch bounds | +0.8% | 5.843 |
| + BF16 row reduce | +0.3% | **5.867** |

> GPU 0-3（PIX 拓扑更优）预估可达 **5.99 tok/s**。

---

## 5. 关键经验

1. **FP4 > BF16 用于单 token 解码** — 内存带宽是瓶颈，紧凑格式的带宽优势超过了解压开销
2. **单 token GEMV 不适合 tensor core** — M=1 无法利用 WMMA 的 16×16 tile
3. **CUDA Graphs 在复杂模型中成本高** — 动态 position 依赖需要大量重构
4. **PCIe A800 无 NVLink 是主要限制** — 估算 8 GPU 可提升 10-15%，SXM+NVLink 可提升 15-25%
5. **性能 profiling 至关重要** — 即使 `--profile-stages` 有 bug，旧的 profile 数据仍帮助定位了 Indexer 和 out_proj 两大热点

---

## 6. 文件改动

| 文件 | 改动说明 |
|---|---|
| `DeepSeek-V4-Flash/inference/model.py` | Fused attn out proj, fused Indexer full, CUDA o_proj fix |
| `python_infer_deepseek_v4_flash.py` | 禁用 BF16 cache, 启用 BF16 row reduce |
| `src/deepseek_v4_a800_kernels.cu` | FP4 kernel launch bounds 优化 |

---

## 7. 引用

```bibtex
@misc{a800-deepseekv4-opt,
  author = {luogantt},
  title = {DeepSeek-V4-Flash A800 PCIe Inference Optimization},
  year = {2026},
  url = {https://github.com/luogantt/LLM-inference-engine},
  tag = {v0.2.0-a800-opt}
}
```
