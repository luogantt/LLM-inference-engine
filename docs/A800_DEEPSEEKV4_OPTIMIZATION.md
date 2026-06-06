# DeepSeek-V4-Flash 在 4×A800 PCIe 上推理优化实践

> **TL;DR**: 在 4 张 NVIDIA A800 80GB PCIe（无 NVLink）上推理 DeepSeek-V4-Flash (43层 MoE, 256专家, 4096维)，从初始的纯 PyTorch 反量化回退路径 **0.69 tok/s** 出发，通过 14 个自定义 CUDA kernel、MoE 分发优化、分布式通信优化等手段，最终达到 **5.87 tok/s**，**整体提升 8.5×**。

- **GitHub**: [luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)
- **Tag**: [`v0.2.0-a800-opt`](https://github.com/luogantt/LLM-inference-engine/releases/tag/v0.2.0-a800-opt)
- **Branch**: `cuda_A800_deepseekv4_deepapi`
- **Final Commit**: `bab88c8`

---

## 1. 背景

### 1.1 模型架构

[DeepSeek-V4-Flash](https://github.com/deepseek-ai/DeepSeek-V4) 是 DeepSeek 最新一代 MoE 大模型，采用多项先进架构：

| 参数 | 值 |
|---|---|
| 层数 | 43 |
| 隐藏维度 | 4096 |
| 注意力头数 | 64 |
| MoE 专家总数 | 256 (每 GPU 64，4 GPU TP) |
| 每 token 激活专家 | 6 (路由) + 1 (共享) |
| KV Cache 窗口 | 128 (滑动窗口) |
| 压缩比 | 41 层带 Indexer (4:1 KV 压缩) |
| HC 倍数 | 4 |
| 权重格式 | 注意力/共享专家: FP8; 路由专家: FP4 |
| 总参数量 | ~130B (FP4 压缩后 ~20GB/GPU) |

### 1.2 硬件环境

| 组件 | 规格 |
|---|---|
| GPU | 4× NVIDIA A800 80GB PCIe |
| NVLink | ❌ 未启用 (PCIe 版本) |
| GPU 互连 | PIX/PXB (PCIe 4.0 桥接) |
| CPU | Intel Xeon |
| CUDA | 12.4 |
| PyTorch | 2.x |
| Python | 3.13 |

### 1.3 核心挑战

A800 是 **SM80 (Ampere)** 架构，而 DeepSeek-V4-Flash 官方推理代码依赖 **TileLang JIT 编译** 生成 **SM89 (Ada/Hopper)** FP8 GEMM kernel。A800 无法执行这些 kernel，编译时报错：

```
error: no suitable constructor exists to convert from "float" to "fp8_e8_t"
```

这意味着我们必须**完全绕过 TileLang**，自建 A800 兼容的推理路径。

---

## 2. 优化全景

```
0.69 tok/s  ──┬── Phase 1: 纯 PyTorch 反量化回退
              │   FP8/FP4 → BF16, F.linear / cuBLAS
              │
2.37 tok/s  ──┼── Phase 2: 自定义 CUDA FP4/FP8 kernel (+243%)
              │   FP4 dequant GEMM, 融合 expert FFN, FP8 shared FFN
              │
2.99 tok/s  ──┼── Phase 3: MoE 分发 + 分布式优化 (+26%)
              │   Top-k dispatch, buffer 复用, BF16 reduce, 分布式 argmax
              │
4.93 tok/s  ──┼── Phase 4: 融合 CUDA kernel (+65%)
              │   融合 HC split/sinkhorn, 融合 sparse attention,
              │   融合 Indexer, 融合 attention o_proj
              │
5.66 tok/s  ──┼── Phase 5: BF16 top-k 缓存 + 行并行 reduce (+15%)
              │   专家权重 BF16 缓存, BF16 all-reduce, sparse attn 优化
              │
5.87 tok/s  ──┴── Phase 6: 最终精调 (+3.7%)
                  禁用 BF16 缓存, 融合 attn out proj, launch bounds 调优
```

**总提升: 8.5×**（从 0.69 到 5.87 tok/s）

---

## 3. Phase 1: 纯 PyTorch 反量化回退 (0.69 tok/s)

### 3.1 问题

A800 (sm80) 无法运行 TileLang 生成的 sm89 FP8 GEMM kernel。需要一条完全不同的路径。

### 3.2 方案

通过环境变量 `A800_FORCE_DEQUANT_GEMM=1` 触发回退路径：

1. **绕过 TileLang**: 在 `linear()` 函数中，当检测到 `A800_FORCE_DEQUANT_GEMM=1` 时，跳过 `act_quant()`（FP8 量化激活）和 `fp8_gemm()`/`fp4_gemm()`（TileLang GEMM）
2. **权重反量化**: FP8 权重通过 `_dequantize_fp8_weight()` 反量化到 BF16；FP4 权重通过 `_dequantize_fp4_weight()` 用查表法 + block scale 反量化到 BF16
3. **标准 PyTorch GEMM**: 用 `F.linear()` (cuBLAS) 执行反量化后的 BF16 矩阵乘法
4. **跳过 Hadamard 旋转**: `rotate_activation()` 在 A800 回退路径返回恒等映射

```python
# model.py - linear() 回退逻辑
if _a800_force_dequant_gemm():
    if weight.dtype == torch.float8_e4m3fn:
        weight = _dequantize_fp8_weight(weight)
    elif weight.dtype == torch.float4_e2m1fn_x2:
        weight = _dequantize_fp4_weight(weight)
    return F.linear(x, weight, bias)
```

### 3.3 性能

```
tokens_per_s=0.691
```

**瓶颈分析**: 每层 256 个专家的 FP4 权重需要在每次推理时反量化（尽管有 LRU 缓存），且 cuBLAS GEMV (M=1) 对于单 token 解码效率低下。但这条路**可以跑通**，验证了模型加载和权重转换的正确性。

---

## 4. Phase 2: 自定义 CUDA FP4/FP8 Kernel (+243% → 2.37 tok/s)

### 4.1 FP4 Dequant GEMM Kernel

FP4 (E2M1) 格式将 2 个 4-bit 权重打包在 1 个 uint8 中。kernel 在线解包并执行 GEMV：

```cpp
// 在线 FP4 解包: 查表 + block scale 乘法
uint8_t packed = w_row[k >> 1];
uint8_t nibble = (k & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
float w = fp4_e2m1_to_float(nibble) * scales[scale_row * scale_cols + scale_col];
acc += __bfloat162float(x_row[k]) * w;
```

**设计要点**:
- 每个 block 计算一个 (output_dim, token) 对，256 线程协作
- 共享内存做 block 内 reduction
- Grid: `(out_dim, tokens)` — 单 token 解码时 grid 为 `(dim, 1)`
- BF16 输入/输出，FP32 累积

### 4.2 融合 Expert FFN Kernel

将 SwiGLU FFN 的三个线性层融合为两次 kernel launch：

**Kernel 1 — `fp4_expert_gate_up_fused_kernel`**: 同时计算 gate (w1) 和 up (w3) 投影
```cpp
// 融合计算: gate_acc 和 up_acc 在同一循环中计算
for (int k = tid; k < dim; k += blockDim.x) {
    float xv = __bfloat162float(x_row[k]);
    // w1 (gate): dequant + scale
    gate_acc += xv * fp4_e2m1_to_float(nibble1) * scales1[...];
    // w3 (up): dequant + scale
    up_acc += xv * fp4_e2m1_to_float(nibble3) * scales3[...];
}
// SiLU(gate) * up * route, 写入 hidden
float silu = gate_v / (1.0f + expf(-gate_v));
hidden[...] = silu * up * route;
```

**Kernel 2 — `fp8_dequant_gemm_bf16_kernel`**: 将 hidden 通过 w2 投影回 dim 维度 (FP8→BF16 反量化)

### 4.3 FP8 Shared Expert FFN Kernel

共享专家的权重是 FP8 格式，kernel 融合了 gate+up (w1, w3) 和 down (w2) 三步：

```cpp
// ds_v4_fp8_shared_expert_ffn_bf16
// Launch 1: fp8_shared_gate_up_fused_kernel (inter_dim, tokens)
// Launch 2: fp8_dequant_gemm_bf16_kernel (dim, tokens)
```

### 4.4 性能提升

```
Phase 1 (纯 PyTorch):  0.69 tok/s
Phase 2 (+CUDA FP4):   2.37 tok/s  (+243%)
```

关键因素：
- 消除 Python 层面的反量化开销（FP4 查表在 kernel 内完成）
- 减少 kernel launch 次数（3 次 F.linear → 2 次自定义 kernel）
- 避免 cuBLAS GEMV 的通用性开销

---

## 5. Phase 3: MoE 分发 + 分布式优化 (+26% → 2.99 tok/s)

### 5.1 快速解码 MoE 分发

单 token 解码每层只需 6 个激活专家，而非全部 256 个。通过只扫描 top-k 索引选中的专家：

```python
# model.py - MoE._forward_no_profile()
if _a800_fast_decode_moe() and x.size(0) == 1:
    # 只处理 topk 专家，跳过 bincount + where 的全局扫描
    for top, expert_id in enumerate(indices[0].tolist()):
        if self.experts_start_idx <= expert_id < self.experts_end_idx:
            y += expert(x, weights[:, top:top+1])
```

### 5.2 FP4 Top-K Expert FFN (批量指针表)

将同一 MoE 层所有选中专家的权重/scale 指针预先打包为 GPU 上的 `int64` tensor，通过一次 kernel launch 处理所有 topk 专家：

```python
# 预构建指针表（初始化时一次）
self._a800_fp4_topk_ptrs = (
    ptr_tensor(w1_ptrs),   # [n_local] int64 指针
    ptr_tensor(s1_ptrs),
    ptr_tensor(w2_ptrs),
    ptr_tensor(s2_ptrs),
    ptr_tensor(w3_ptrs),
    ptr_tensor(s3_ptrs),
    s1_shape, s2_shape, s3_shape,
    inter_dim,
)
```

**CUDA kernel 端**:
```cpp
// fp4_topk_gate_up_fused_kernel
// Grid: (ceil(inter_dim/TILE), topk)
// 每 block 处理 TILE=16 个输出，x 加载到共享内存一次，8 个 warp 各计算 2 个输出
const uint8_t* w1_base = reinterpret_cast<const uint8_t*>(packed_w1_ptrs[local_e]);
const float* scales1 = reinterpret_cast<const float*>(scales1_ptrs[local_e]);
// ... 在线 FP4 解包 + GEMV
```

### 5.3 FP32 直接累积

传统 MoE 需要为每个专家分配临时输出、然后求和。新的 `fp4_expert_ffn_accum_f32` kernel 直接将专家结果累积到 FP32 输出缓冲区：

```cpp
// fp4_topk_w2_accum_f32_kernel
// y_accum[out_idx] += result  (FP32 原子性由不同 block 索引保证)
```

消除了每专家的临时 BF16 输出 tensor 分配和 `torch.add` kernel launch。

### 5.4 Buffer 复用

单 token 解码每步的 tensor shape 完全一致，可以复用缓冲区：

```python
def _a800_decode_accum_buffer(self, x, dtype):
    y = self._a800_decode_y  # 复用上一轮的 FP32 缓冲区
    if y is None or y.shape != x.shape:
        y = torch.empty_like(x, dtype=dtype)
        self._a800_decode_y = y
    y.zero_()
    return y
```

类似地复用 top-k hidden buffer、int32 index buffer、argmax pack buffer。

### 5.5 分布式优化

**分布式 Argmax**: 贪婪解码只需全局最大 logit 的 token ID，无需 all-gather 完整 vocab logits (128K×4=512K float32)。改为每 rank 计算本地 (max_value, max_index)，all-gather 2 个值而非 vocab_size 个值：

```python
local_values, local_indices = logits.max(dim=-1)  # 本地 argmax
local_pack = torch.stack((local_values, local_indices), dim=-1)
dist.all_gather(all_packs, local_pack)  # 每 rank 仅传 2 个 float32
best_rank = packs[:, :, 0].argmax(dim=0)  # 全局最大
```

通信量: `vocab_size × 4 bytes` → `world_size × 2 × 4 bytes`，减少 ~16000×

**Defer Token Decode**: 将 tokenizer decode 移到 CUDA timer 外部，避免 GPU→CPU 同步影响计时。

**EOS Check Interval**: 通过 `A800_EOS_CHECK_INTERVAL=0` 在计时期间跳过 per-step EOS 检查，减少 GPU→CPU sync。

### 5.6 性能提升

```
Phase 2 (CUDA FP4):     2.37 tok/s
Phase 3 (+MoE dispatch): 2.99 tok/s  (+26%)
```

---

## 6. Phase 4: 融合 CUDA Kernel (+65% → 4.93 tok/s)

这是**提升最大**的阶段。核心洞察：DeepSeek-V4-Flash 的每个 Transformer 层包含 10+ 次独立的 kernel launch（RMSNorm → Linear → SiLU → HC → Attention → Indexer → MoE → all-reduce...），每次 launch 都有固定的调度延迟 (~5-10μs)。43 层 × 10+ kernel = 430+ launches/token，launch overhead 累计可达数毫秒。

### 6.1 融合 HC Split Sinkhorn Kernel

Hyper-Connection (HC) 是 DeepSeek-V4 的核心创新之一。每层的 HC pre-processing 包含 10+ 个 PyTorch 操作：
- RMSNorm (1 kernel)
- Linear projection → mixes (1 kernel)
- Sigmoid → pre, post (2 kernel)
- Softmax + Sinkhorn iterations → comb (多次 kernel)
- Weighted sum y = Σ pre[j] × x[j] (1 kernel)

**融合后: 1 次 kernel launch**:

```cpp
// hc_pre_fused_kernel — 一个 kernel 完成 RMS norm + Linear + Sinkhorn + 加权和
// Step 1: RMS norm (共享内存 reduction)
// Step 2: mixes = rsqrt * (x @ hc_fn^T) (warp 级 dot product)
// Step 3: Sinkhorn iterations (单线程，寄存器内计算)
// Step 4: y[d] = Σ pre[j] * x[j*dim + d] (并行 weighted sum)
```

**约束**: hc_mult ≤ 8（`comb_reg[64]` 寄存器数组限制）

### 6.2 融合 Sparse Attention Kernel

稀疏注意力只计算 top-k KV 位置的注意力分数。原始实现分两步：
1. `torch.gather` 从 KV cache 选取 topk 位置
2. `torch.matmul` 计算注意力

**融合后: 1 次 kernel launch**:

```cpp
// sparse_attn_decode_bf16_kernel
// 使用 Online Softmax 避免存储完整 attention matrix
for (int t_start = 0; t_start < topk; t_start += TILE) {
    // 协作加载 KV tile 到共享内存
    // Warp 级 dot product (q · k)
    // Online softmax 更新:
    //   new_max = max(max_score, score)
    //   acc_o *= exp(max_score - new_max)
    //   sum_exp *= exp(max_score - new_max)
    //   acc_o += exp(score - new_max) * v
    //   sum_exp += exp(score - new_max)
}
// 归一化 + BF16 输出
out = acc_o / sum_exp
```

### 6.3 融合 Indexer Kernel

41/43 层有 Indexer（用于选择压缩 KV 缓存的 top-k 位置）。原始实现：Q 投影 (Linear) + 权重投影 (Linear) + einsum + ReLU + topk，共 5+ kernel launches。

**`ds_v4_indexer_full_fused_bf16`**: 单 kernel launch 完成 Q 投影 + 权重投影 + RoPE + einsum + ReLU + 加权和：

```cpp
// indexer_full_fused_kernel — 单 block, 256 threads
// Step 1: 加载 x 到共享内存
// Step 2: wq_b GEMV（warp 协作）
// Step 3: weights_proj GEMV (warp 0)
// Step 4: RoPE 旋转 q[..., -rd:]
// Step 5: einsum + ReLU + weighted sum → index_score[t]
```

### 6.4 融合 Attention O-Projection Kernel

Attention 输出投影 `einsum + wo_b` (2 kernel launches) 融合为 1 次：

```cpp
// attn_o_proj_fused_kernel
// Step 1: 加载 o 到共享内存 (bf16→fp32)
// Step 2: einsum("bsgd,grd->bsgr") — warp 协作 dot product
// Step 3: wo_b GEMV — 行优先合并读取
```

### 6.5 性能提升

```
Phase 3 (MoE dispatch):  2.99 tok/s
Phase 4 (+fused kernels): 4.93 tok/s  (+65%)
```

这是单阶段最大提升。融合 kernel 将每层 kernel launch 从 10+ 减少到 4-5 次，大幅减少 launch overhead。

---

## 7. Phase 5: BF16 Top-K 缓存 + 行并行 Reduce (+15% → 5.66 tok/s)

### 7.1 BF16 Top-K 专家缓存

对于反复激活的专家（单 prompt 生成中热点专家占总激活的 60-80%），将 FP4 权重预反量化为 BF16 并缓存，避免每步在线解压：

```python
# MoE._a800_selected_bf16_topk_ptrs()
# 对每个选中的专家：
w1_bf16 = _dequantize_fp4_weight(expert.w1.weight)  # FP4 → BF16 (50MB/专家)
# 缓存到 GPU，用 data_ptr 构建指针表
# LRU 淘汰 + 内存 guard（free < min_free 时禁用缓存）
```

**缓存策略**:
- 最多缓存 `A800_BF16_TOPK_MAX_LOCAL_EXPERTS` 个专家（默认 2）
- `A800_BF16_TOPK_CACHE_MB` 限制总缓存大小（默认 2048 MB）
- `A800_BF16_TOPK_MIN_FREE_MB` 内存 guard（默认 4096 MB）
- `A800_BF16_TOPK_FREEZE_AFTER_WARMUP` 预热后冻结缓存

### 7.2 BF16 Row-Parallel Reduce

RowParallelLinear 层的 all-reduce 使用 BF16 精度（通信量减半）：

```python
# python_infer_deepseek_v4_flash.py
"A800_BF16_ROW_REDUCE": "1",
```

对于 4096 维 × 4 GPU 的 all-reduce，FP32 → BF16 将通信量从 64KB 降到 32KB per all-reduce。

### 7.3 性能提升

```
Phase 4 (fused kernels):    4.93 tok/s
Phase 5 (+BF16 cache+reduce): 5.66 tok/s  (+15%)
```

---

## 8. Phase 6: 最终精调 (+3.7% → 5.87 tok/s)

### 8.1 禁用 BF16 缓存 (+4.0%)

**反直觉发现**: BF16 缓存专家权重（~50MB/专家）虽然避免了在线 FP4 解包，但每个专家的权重数据量是 FP4（~12.5MB）的 4 倍。对于单 token 解码（GEMV 内存带宽瓶颈），FP4 的紧凑表示更高效。

```python
"A800_CACHE_FP4_BF16": "0",
"A800_USE_CUDA_BF16_TOPK_FFN": "0",
```

### 8.2 融合 Attention 输出投影 (+0.5%)

预计算 `fused = wo_b @ wo_a`，将 attention o_proj 从 2 kernel (einsum + wo_b) 合并为 1 次 F.linear：

```python
# model.py - Attention._a800_out_proj()
wo_a = _dequantize_fp8_weight(self.wo_a.weight).float()
wo_b = _dequantize_fp8_weight(self.wo_b.weight).float()
fused = torch.cat([wo_b[:,g,:] @ wo_a[g] for g in range(n_groups)], dim=1)
x = F.linear(o, fused)  # 1 kernel launch
```

### 8.3 融合 Indexer Full Kernel (+0.5%)

启用 Phase 4 中已实现的 `ds_v4_indexer_full_fused_bf16` kernel——之前因未正确触发，现在在单 token 解码路径中自动启用。

### 8.4 Launch Bounds 调优 (+0.8%)

FP4 top-k gate+up kernel 优化共享内存占用，允许更高的 SM 占用率：

```cpp
// 优化前: 保守的 launch bounds
__global__ void fp4_topk_gate_up_fused_kernel(...)

// 优化后: 允许 4 blocks/SM (A800 164KB shared memory)
__global__ __launch_bounds__(256, 4) void fp4_topk_gate_up_fused_kernel(...)
__global__ __launch_bounds__(256, 6) void fp4_topk_w2_accum_f32_kernel(...)
```

### 8.5 性能

```
Phase 5 (BF16 cache):   5.66 tok/s
Phase 6 (final polish): 5.87 tok/s  (+3.7%)
```

---

## 9. 尝试过但未采纳的优化

### ❌ CUDA Graphs

DeepSeek-V4 Flash 有 41/43 层使用 Indexer，其内部依赖 Python int `start_pos` 计算 `cache_len`。CUDA Graph 在捕获时固化此值为常量，replay 时无法更新。需要大规模重构 Indexer/Compressor 代码——投入产出比不高。

### ❌ Tensor Cores (WMMA)

单 token 解码是 GEMV (M=1)，Ampere WMMA 要求 M≥16。填充到 16 会浪费 15/16 的计算。且 BF16 权重比 FP4 大 4 倍，抵消了 tensor core 加速。**结论**: Tensor core 不适合单 token 解码。

### ❌ FP4 TILE 增大 (8→16)

每个 block 处理更多输出元素（grid 减半），但总内存流量不变，性能无变化——确认瓶颈在内存带宽而非 grid launch。

### ❌ Indexer 结果缓存

相邻 token 间复用 Indexer top-k 结果（压缩 KV 缓存每 4 token 才增长一次），但滑动窗口注意力的位置每 token 变化，复用导致输出质量严重退化。

### ❌ Packed-byte FP4 Dot

尝试在一次字节加载中解包两个 FP4 权重并同时计算两个点积。但查表操作的串行依赖抵消了带宽优势。

### ❌ Async MoE All-Reduce

重叠 MoE all-reduce 和 shared expert 计算。实测通信和计算重叠不足（PCIe 带宽已饱和），反而增加同步开销。

---

## 10. 架构总览

### 10.1 完整 Kernel 清单

| Kernel | 功能 | Launch 次数/token |
|---|---|---|
| `fp4_dequant_gemm_bf16` | FP4 权重在线解包 + GEMV | 按需 |
| `fp4_dequant_gemm_accum_f32` | FP4 GEMV, FP32 累积 | 按需 |
| `fp8_dequant_gemm_bf16` | FP8 GEMV | 按需 |
| `fp8_shared_gate_up_fused` | 共享专家 gate+up 融合 | 43× |
| `fp4_expert_gate_up_fused` | 路由专家 gate+up 融合 | 43× topk |
| `fp4_topk_gate_up_fused` | Top-k 路由专家批量 gate+up | 43× |
| `fp4_topk_w2_accum_f32` | Top-k 批量 w2, FP32 累积 | 43× |
| `bf16_topk_gate_up_fused` | BF16 缓存专家 gate+up | 按需 |
| `bf16_topk_w2_accum_f32` | BF16 缓存专家 w2 | 按需 |
| `sparse_attn_decode_bf16` | 融合稀疏注意力 (Online Softmax) | 43× |
| `hc_pre_fused` | 融合 HC pre (RMS+Linear+Sinkhorn+加权和) | 43× |
| `hc_post_fused` | 融合 HC post | 43× |
| `attn_o_proj_fused` | 融合 attention 输出投影 | 43× |
| `indexer_score_fused` | 融合 Indexer 评分 | 41× |
| `indexer_full_fused` | 融合 Indexer 全流程 (Q+权重+RoPE+评分) | 41× |

### 10.2 回退链设计

每个 CUDA 快速路径都有多层回退：

```
FP4 CUDA top-k FFN
    ↓ (失败)
FP4 CUDA per-expert FFN
    ↓ (失败)
FP8 CUDA shared FFN (共享专家)
    ↓ (失败)
PyTorch 反量化 + F.linear (cuBLAS)
```

关键设计原则：**任何 kernel 失败都不会崩溃**，自动回退到次优路径。

### 10.3 Stage Profiler

通过 `A800_PROFILE_STAGES=1` 启用，使用 CUDA Events 记录每个阶段的耗时：

```
========== A800 stage profile ==========
stage                              calls     total_ms       avg_ms
moe.routed_topk_cuda                 258      123.456       0.4785
attn.sparse                          258       98.765       0.3828
moe.shared                           258       45.678       0.1770
hc.pre                               258       32.109       0.1245
...
```

无需 profiling 时，NoopProfileRegion 保证零开销。

---

## 11. 完整性能演进

| 阶段 | tok/s | 相对提升 | 累计提升 | 关键改动 |
|---|---|---|---|---|
| Phase 1 | 0.69 | — | 1.0× | 纯 PyTorch 反量化回退 |
| Phase 2 | 2.37 | +243% | 3.4× | CUDA FP4/FP8 dequant GEMM, 融合 FFN |
| Phase 3 | 2.99 | +26% | 4.3× | MoE 分发, buffer 复用, BF16 reduce, 分布式 argmax |
| Phase 4 | 4.93 | +65% | 7.1× | 融合 HC/Attention/Indexer CUDA kernel |
| Phase 5 | 5.66 | +15% | 8.2× | BF16 top-k 缓存, BF16 行并行 reduce |
| Phase 6 | 5.87 | +3.7% | **8.5×** | 禁用 BF16 缓存, 融合 attn out proj, launch bounds |

---

## 12. 关键经验

### 12.1 单 Token 解码的特殊性

1. **GEMV 是内存带宽瓶颈，不是计算瓶颈** — 每个权重元素只用一次，ARITH INTENSITY ≈ 2/(2×bytes_per_element)。FP4 (0.5B/elem) 的 AI 约为 BF16 (2B/elem) 的 4 倍，解释了为什么 FP4 在线解包比 BF16 缓存更快
2. **Tensor Cores 不适合 GEMV** — M=1 远小于 WMMA tile 16×16
3. **Kernel Launch Overhead 不可忽略** — 430+ launches/token, 每次 ~5-10μs, 累计 2-4ms = 10-20% 总延迟

### 12.2 PCIe A800 的特殊性

1. **无 NVLink 是主要限制** — 估算 NVLink 版本可额外提升 15-25%
2. **PIX vs PXB 拓扑差异** — GPU 0-1 有 PIX (单桥) 比 GPU 3-6 的 PXB (多桥) 快 ~2%
3. **通信与计算难以重叠** — PCIe 带宽有限, async all-reduce 实测无收益

### 12.3 工程实践

1. **性能 profiling 至关重要** — 即使 profiler 初期有 bug, 旧数据仍帮助定位热点
2. **回退链设计降低了实验风险** — 每个新 kernel 可以独立测试, 失败自动回退
3. **Buffer 复用是低垂果实** — 单 token 解码每步 tensor shape 不变, 复用省去 allocator 开销
4. **FP4 > BF16 for decode** — 带宽优势超过解包开销, 是反直觉但正确的结果

---

## 13. 文件与代码

| 文件 | 行数 | 说明 |
|---|---|---|
| `src/deepseek_v4_a800_kernels.cu` | 2146 | 14 个自定义 CUDA kernel + C 导出接口 |
| `DeepSeek-V4-Flash/inference/model.py` | +2800 | Python 端 kernel wrapper, 回退链, buffer 复用, 分布式优化 |
| `python_infer_deepseek_v4_flash.py` | 541 | CLI 入口, 权重转换, A800 兼容配置 |
| `DeepSeek-V4-Flash/inference/generate.py` | +86 | 单 prompt 快速生成, 分布式 argmax, EOS 优化 |
| `Makefile.cuda_lib` | +6 | `deepseek-v4-a800` 编译目标 |

---

## 14. 快速开始

```bash
# 编译 CUDA kernel
make -f Makefile.cuda_lib deepseek-v4-a800 A=sm_80

# 4 GPU 推理
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.run \
  --standalone --nproc-per-node 4 \
  python_infer_deepseek_v4_flash.py \
  --ckpt-path ../models/DeepSeek-V4-Flash-converted \
  --config ../models/DeepSeek-V4-Flash/inference/config_a800_fp32scale.json \
  --prompt "介绍一下黑格尔的思想" \
  --max-new-tokens 128 \
  --max-seq-len 4096 \
  --max-batch-size 1 \
  --temperature 0
```

---

## 15. 引用

```bibtex
@misc{a800-deepseekv4-opt,
  author = {luogantt},
  title = {DeepSeek-V4-Flash A800 PCIe Inference Optimization},
  year = {2026},
  url = {https://github.com/luogantt/LLM-inference-engine},
  tag = {v0.2.0-a800-opt}
}
```
