#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cmath>

static thread_local char g_last_error[256] = {0};

static void set_last_error(const char* msg) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", msg ? msg : "");
}

static void set_last_cuda_error(cudaError_t err) {
    std::snprintf(g_last_error, sizeof(g_last_error), "CUDA: %s", cudaGetErrorString(err));
}

extern "C" const char* ds_v4_a800_last_error() {
    return g_last_error;
}

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t v) {
    switch (v & 0x0F) {
        case 0x0: return 0.0f;
        case 0x1: return 0.5f;
        case 0x2: return 1.0f;
        case 0x3: return 1.5f;
        case 0x4: return 2.0f;
        case 0x5: return 3.0f;
        case 0x6: return 4.0f;
        case 0x7: return 6.0f;
        case 0x8: return 0.0f;
        case 0x9: return -0.5f;
        case 0xA: return -1.0f;
        case 0xB: return -1.5f;
        case 0xC: return -2.0f;
        case 0xD: return -3.0f;
        case 0xE: return -4.0f;
        default: return -6.0f;
    }
}

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t v) {
    if ((v & 0x7F) == 0) {
        return (v & 0x80) ? -0.0f : 0.0f;
    }

    float sign = (v & 0x80) ? -1.0f : 1.0f;
    int exp = (v >> 3) & 0x0F;
    int mant = v & 0x07;

    if (exp == 0) {
        return sign * ldexpf(static_cast<float>(mant), -9);
    }
    if (exp == 0x0F && mant == 0x07) {
        return sign * 448.0f;
    }
    return sign * ldexpf(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
}

__global__ void fp4_dequant_gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ y,
    int tokens,
    int in_dim,
    int out_dim,
    int scale_rows,
    int scale_cols,
    int group_size
) {
    int out_idx = blockIdx.x;
    int token_idx = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float acc = 0.0f;

    const __nv_bfloat16* x_row = x + token_idx * in_dim;
    const uint8_t* w_row = packed_w + out_idx * (in_dim / 2);

    int scale_row = out_idx;
    if (scale_rows != out_dim) {
        scale_row = out_idx / group_size;
        if (scale_row >= scale_rows) {
            scale_row = scale_rows - 1;
        }
    }

    for (int k = tid; k < in_dim; k += blockDim.x) {
        uint8_t packed = w_row[k >> 1];
        uint8_t nibble = (k & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
        int scale_col = k / group_size;
        if (scale_col >= scale_cols) {
            scale_col = scale_cols - 1;
        }
        float w = fp4_e2m1_to_float(nibble) * scales[scale_row * scale_cols + scale_col];
        acc += __bfloat162float(x_row[k]) * w;
    }

    smem[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y[token_idx * out_dim + out_idx] = __float2bfloat16(smem[0]);
    }
}

__global__ void fp4_dequant_gemm_accum_f32_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ scales,
    float* __restrict__ y_accum,
    int tokens,
    int in_dim,
    int out_dim,
    int scale_rows,
    int scale_cols,
    int group_size
) {
    int out_idx = blockIdx.x;
    int token_idx = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float acc = 0.0f;

    const __nv_bfloat16* x_row = x + token_idx * in_dim;
    const uint8_t* w_row = packed_w + out_idx * (in_dim / 2);

    int scale_row = out_idx;
    if (scale_rows != out_dim) {
        scale_row = out_idx / group_size;
        if (scale_row >= scale_rows) {
            scale_row = scale_rows - 1;
        }
    }

    for (int k = tid; k < in_dim; k += blockDim.x) {
        uint8_t packed = w_row[k >> 1];
        uint8_t nibble = (k & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
        int scale_col = k / group_size;
        if (scale_col >= scale_cols) {
            scale_col = scale_cols - 1;
        }
        float w = fp4_e2m1_to_float(nibble) * scales[scale_row * scale_cols + scale_col];
        acc += __bfloat162float(x_row[k]) * w;
    }

    smem[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float out = __bfloat162float(__float2bfloat16(smem[0]));
        y_accum[token_idx * out_dim + out_idx] += out;
    }
}

__global__ void fp8_dequant_gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ y,
    int tokens,
    int in_dim,
    int out_dim,
    int scale_rows,
    int scale_cols,
    int group_size
) {
    int out_idx = blockIdx.x;
    int token_idx = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float acc = 0.0f;

    const __nv_bfloat16* x_row = x + token_idx * in_dim;
    const uint8_t* w_row = w + out_idx * in_dim;

    int scale_row = out_idx / group_size;
    if (scale_row >= scale_rows) {
        scale_row = scale_rows - 1;
    }
    const float* scale_row_ptr = scales + scale_row * scale_cols;

    for (int k = tid; k < in_dim; k += blockDim.x) {
        int scale_col = k / group_size;
        if (scale_col >= scale_cols) {
            scale_col = scale_cols - 1;
        }
        float wv = fp8_e4m3fn_to_float(w_row[k]) * scale_row_ptr[scale_col];
        acc += __bfloat162float(x_row[k]) * wv;
    }

    smem[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y[token_idx * out_dim + out_idx] = __float2bfloat16(smem[0]);
    }
}

__global__ void fp8_shared_gate_up_fused_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w1,
    const float* __restrict__ scales1,
    const uint8_t* __restrict__ w3,
    const float* __restrict__ scales3,
    __nv_bfloat16* __restrict__ hidden,
    int tokens,
    int dim,
    int inter_dim,
    int scale1_rows,
    int scale1_cols,
    int scale3_rows,
    int scale3_cols,
    int group_size,
    float swiglu_limit
) {
    int inter_idx = blockIdx.x;
    int token_idx = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* smem_gate = smem;
    float* smem_up = smem + blockDim.x;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    const __nv_bfloat16* x_row = x + token_idx * dim;
    const uint8_t* w1_row = w1 + inter_idx * dim;
    const uint8_t* w3_row = w3 + inter_idx * dim;

    int scale1_row = inter_idx / group_size;
    if (scale1_row >= scale1_rows) {
        scale1_row = scale1_rows - 1;
    }
    int scale3_row = inter_idx / group_size;
    if (scale3_row >= scale3_rows) {
        scale3_row = scale3_rows - 1;
    }
    const float* scale1_row_ptr = scales1 + scale1_row * scale1_cols;
    const float* scale3_row_ptr = scales3 + scale3_row * scale3_cols;

    for (int k = tid; k < dim; k += blockDim.x) {
        float xv = __bfloat162float(x_row[k]);
        int scale1_col = k / group_size;
        if (scale1_col >= scale1_cols) {
            scale1_col = scale1_cols - 1;
        }
        int scale3_col = k / group_size;
        if (scale3_col >= scale3_cols) {
            scale3_col = scale3_cols - 1;
        }
        float w1v = fp8_e4m3fn_to_float(w1_row[k]) * scale1_row_ptr[scale1_col];
        float w3v = fp8_e4m3fn_to_float(w3_row[k]) * scale3_row_ptr[scale3_col];
        gate_acc += xv * w1v;
        up_acc += xv * w3v;
    }

    smem_gate[tid] = gate_acc;
    smem_up[tid] = up_acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_gate[tid] += smem_gate[tid + stride];
            smem_up[tid] += smem_up[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float gate_v = __bfloat162float(__float2bfloat16(smem_gate[0]));
        float up = __bfloat162float(__float2bfloat16(smem_up[0]));
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate_v = fminf(gate_v, swiglu_limit);
        }
        float silu = gate_v / (1.0f + expf(-gate_v));
        hidden[token_idx * inter_dim + inter_idx] = __float2bfloat16(silu * up);
    }
}

__global__ void fp4_expert_gate_up_fused_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ route,
    const uint8_t* __restrict__ packed_w1,
    const float* __restrict__ scales1,
    const uint8_t* __restrict__ packed_w3,
    const float* __restrict__ scales3,
    __nv_bfloat16* __restrict__ hidden,
    int tokens,
    int dim,
    int inter_dim,
    int scale1_rows,
    int scale1_cols,
    int scale3_rows,
    int scale3_cols,
    int group_size,
    float swiglu_limit
) {
    int inter_idx = blockIdx.x;
    int token_idx = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* smem_gate = smem;
    float* smem_up = smem + blockDim.x;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    const __nv_bfloat16* x_row = x + token_idx * dim;
    const uint8_t* w1_row = packed_w1 + inter_idx * (dim / 2);
    const uint8_t* w3_row = packed_w3 + inter_idx * (dim / 2);

    int scale1_row = inter_idx;
    if (scale1_rows != inter_dim) {
        scale1_row = inter_idx / group_size;
        if (scale1_row >= scale1_rows) {
            scale1_row = scale1_rows - 1;
        }
    }

    int scale3_row = inter_idx;
    if (scale3_rows != inter_dim) {
        scale3_row = inter_idx / group_size;
        if (scale3_row >= scale3_rows) {
            scale3_row = scale3_rows - 1;
        }
    }

    for (int k = tid; k < dim; k += blockDim.x) {
        float xv = __bfloat162float(x_row[k]);
        int scale_col = k / group_size;

        int scale1_col = scale_col < scale1_cols ? scale_col : scale1_cols - 1;
        uint8_t packed1 = w1_row[k >> 1];
        uint8_t nibble1 = (k & 1) ? ((packed1 >> 4) & 0x0F) : (packed1 & 0x0F);
        float w1 = fp4_e2m1_to_float(nibble1) * scales1[scale1_row * scale1_cols + scale1_col];
        gate_acc += xv * w1;

        int scale3_col = scale_col < scale3_cols ? scale_col : scale3_cols - 1;
        uint8_t packed3 = w3_row[k >> 1];
        uint8_t nibble3 = (k & 1) ? ((packed3 >> 4) & 0x0F) : (packed3 & 0x0F);
        float w3 = fp4_e2m1_to_float(nibble3) * scales3[scale3_row * scale3_cols + scale3_col];
        up_acc += xv * w3;
    }

    smem_gate[tid] = gate_acc;
    smem_up[tid] = up_acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_gate[tid] += smem_gate[tid + stride];
            smem_up[tid] += smem_up[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float gate_v = __bfloat162float(__float2bfloat16(smem_gate[0]));
        float up = __bfloat162float(__float2bfloat16(smem_up[0]));
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate_v = fminf(gate_v, swiglu_limit);
        }
        float silu = gate_v / (1.0f + expf(-gate_v));
        float routed = route ? route[token_idx] : 1.0f;
        hidden[token_idx * inter_dim + inter_idx] = __float2bfloat16(silu * up * routed);
    }
}

// Tiled FP4 topk gate+up kernel: each block handles FP4_TOPK_GU_TILE=8 outputs
// for one expert. x loaded into shared memory once per block (not once per output!).
// 8 warps cooperatively compute 8 output elements simultaneously via warp shuffle.
// Grid: (ceil(inter_dim / 8), topk) — 8x fewer blocks than original.
#define FP4_TOPK_GU_TILE 8

__global__ __launch_bounds__(256, 2) void fp4_topk_gate_up_fused_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ routes,
    const int32_t* __restrict__ indices,
    const uintptr_t* __restrict__ packed_w1_ptrs,
    const uintptr_t* __restrict__ scales1_ptrs,
    const uintptr_t* __restrict__ packed_w3_ptrs,
    const uintptr_t* __restrict__ scales3_ptrs,
    __nv_bfloat16* __restrict__ hidden,
    int topk,
    int local_start,
    int n_local,
    int dim,
    int inter_dim,
    int scale1_rows,
    int scale1_cols,
    int scale3_rows,
    int scale3_cols,
    int group_size,
    float swiglu_limit
) {
    int inter_base = blockIdx.x * FP4_TOPK_GU_TILE;
    int top_idx = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    constexpr int WARP_SIZE = 32;

    int expert_id = indices[top_idx];
    int local_e = expert_id - local_start;
    if (local_e < 0 || local_e >= n_local) {
        for (int i = tid; i < FP4_TOPK_GU_TILE; i += blockDim.x) {
            int idx = inter_base + i;
            if (idx < inter_dim) {
                hidden[top_idx * inter_dim + idx] = __float2bfloat16(0.0f);
            }
        }
        return;
    }

    extern __shared__ float smem[];
    float* x_s = smem;  // [dim] fp32

    // Cooperative load of x into shared memory (once per block!)
    for (int k = tid; k < dim; k += blockDim.x) {
        x_s[k] = __bfloat162float(x[k]);
    }
    __syncthreads();

    const uint8_t* w1_base = reinterpret_cast<const uint8_t*>(packed_w1_ptrs[local_e]);
    const uint8_t* w3_base = reinterpret_cast<const uint8_t*>(packed_w3_ptrs[local_e]);
    const float* scales1 = reinterpret_cast<const float*>(scales1_ptrs[local_e]);
    const float* scales3 = reinterpret_cast<const float*>(scales3_ptrs[local_e]);
    const int packed_dim = dim / 2;
    float route_val = routes ? routes[top_idx] : 1.0f;

    // Each warp handles one output element (inter_idx = inter_base + warp_id)
    int inter_idx = inter_base + warp_id;
    if (inter_idx < inter_dim) {
        const uint8_t* w1_row = w1_base + inter_idx * packed_dim;
        const uint8_t* w3_row = w3_base + inter_idx * packed_dim;
        int s1_row = (scale1_rows == inter_dim) ? inter_idx :
            min(inter_idx / group_size, scale1_rows - 1);
        int s3_row = (scale3_rows == inter_dim) ? inter_idx :
            min(inter_idx / group_size, scale3_rows - 1);

        float gate_acc = 0.0f, up_acc = 0.0f;
        for (int k = lane; k < dim; k += WARP_SIZE) {
            float xv = x_s[k];
            int sc = min(k / group_size, scale1_cols - 1);

            // w1 (gate): dequant FP4 nibble + scale
            uint8_t p1 = w1_row[k >> 1];
            uint8_t n1 = (k & 1) ? (p1 >> 4) : (p1 & 0x0F);
            gate_acc += xv * fp4_e2m1_to_float(n1) *
                scales1[s1_row * scale1_cols + sc];

            // w3 (up)
            int s3c = min(sc, scale3_cols - 1);
            uint8_t p3 = w3_row[k >> 1];
            uint8_t n3 = (k & 1) ? (p3 >> 4) : (p3 & 0x0F);
            up_acc += xv * fp4_e2m1_to_float(n3) *
                scales3[s3_row * scale3_cols + s3c];
        }

        // Warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xFFFFFFFF, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xFFFFFFFF, up_acc, offset);
        }

        if (lane == 0) {
            float gate = gate_acc, up = up_acc;
            if (swiglu_limit > 0.0f) {
                up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
                gate = fminf(gate, swiglu_limit);
            }
            float silu = gate / (1.0f + expf(-gate));
            hidden[top_idx * inter_dim + inter_idx] =
                __float2bfloat16(silu * up * route_val);
        }
    }
}


__global__ void fp4_topk_w2_accum_f32_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const int32_t* __restrict__ indices,
    const uintptr_t* __restrict__ packed_w2_ptrs,
    const uintptr_t* __restrict__ scales2_ptrs,
    float* __restrict__ y_accum,
    int topk,
    int local_start,
    int n_local,
    int dim,
    int inter_dim,
    int scale2_rows,
    int scale2_cols,
    int group_size
) {
    int out_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float acc = 0.0f;

    const int packed_inter_dim = inter_dim / 2;

    for (int top_idx = 0; top_idx < topk; ++top_idx) {
        int expert_id = indices[top_idx];
        int local_e = expert_id - local_start;
        if (local_e < 0 || local_e >= n_local) {
            continue;
        }

        const __nv_bfloat16* hidden_row = hidden + top_idx * inter_dim;
        const uint8_t* w2_base = reinterpret_cast<const uint8_t*>(packed_w2_ptrs[local_e]);
        const uint8_t* w2_row = w2_base + out_idx * packed_inter_dim;

        int scale2_row = out_idx;
        if (scale2_rows != dim) {
            scale2_row = out_idx / group_size;
            if (scale2_row >= scale2_rows) {
                scale2_row = scale2_rows - 1;
            }
        }
        const float* scales2 = reinterpret_cast<const float*>(scales2_ptrs[local_e]);

        for (int k = tid; k < inter_dim; k += blockDim.x) {
            uint8_t packed = w2_row[k >> 1];
            uint8_t nibble = (k & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
            int scale_col = k / group_size;
            if (scale_col >= scale2_cols) {
                scale_col = scale2_cols - 1;
            }
            float w = fp4_e2m1_to_float(nibble) * scales2[scale2_row * scale2_cols + scale_col];
            acc += __bfloat162float(hidden_row[k]) * w;
        }
    }

    smem[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y_accum[out_idx] += __bfloat162float(__float2bfloat16(smem[0]));
    }
}

__global__ __launch_bounds__(256, 2) void bf16_topk_gate_up_fused_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ routes,
    const int32_t* __restrict__ indices,
    const uintptr_t* __restrict__ w1_ptrs,
    const uintptr_t* __restrict__ w3_ptrs,
    __nv_bfloat16* __restrict__ hidden,
    int topk,
    int local_start,
    int n_local,
    int dim,
    int inter_dim,
    float swiglu_limit
) {
    int inter_base = blockIdx.x * FP4_TOPK_GU_TILE;
    int top_idx = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    constexpr int WARP_SIZE = 32;

    int expert_id = indices[top_idx];
    int local_e = expert_id - local_start;
    uintptr_t w1_addr = 0;
    uintptr_t w3_addr = 0;
    if (local_e >= 0 && local_e < n_local) {
        w1_addr = w1_ptrs[local_e];
        w3_addr = w3_ptrs[local_e];
    }
    if (local_e < 0 || local_e >= n_local || w1_addr == 0 || w3_addr == 0) {
        for (int i = tid; i < FP4_TOPK_GU_TILE; i += blockDim.x) {
            int idx = inter_base + i;
            if (idx < inter_dim) {
                hidden[top_idx * inter_dim + idx] = __float2bfloat16(0.0f);
            }
        }
        return;
    }

    extern __shared__ float smem[];
    float* x_s = smem;
    for (int k = tid; k < dim; k += blockDim.x) {
        x_s[k] = __bfloat162float(x[k]);
    }
    __syncthreads();

    const __nv_bfloat16* w1_base = reinterpret_cast<const __nv_bfloat16*>(w1_addr);
    const __nv_bfloat16* w3_base = reinterpret_cast<const __nv_bfloat16*>(w3_addr);
    float route_val = routes ? routes[top_idx] : 1.0f;

    int inter_idx = inter_base + warp_id;
    if (inter_idx < inter_dim) {
        const __nv_bfloat16* w1_row = w1_base + inter_idx * dim;
        const __nv_bfloat16* w3_row = w3_base + inter_idx * dim;

        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (int k = lane; k < dim; k += WARP_SIZE) {
            float xv = x_s[k];
            gate_acc += xv * __bfloat162float(w1_row[k]);
            up_acc += xv * __bfloat162float(w3_row[k]);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_acc += __shfl_xor_sync(0xFFFFFFFF, gate_acc, offset);
            up_acc += __shfl_xor_sync(0xFFFFFFFF, up_acc, offset);
        }

        if (lane == 0) {
            float gate = gate_acc;
            float up = up_acc;
            if (swiglu_limit > 0.0f) {
                up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
                gate = fminf(gate, swiglu_limit);
            }
            float silu = gate / (1.0f + expf(-gate));
            hidden[top_idx * inter_dim + inter_idx] =
                __float2bfloat16(silu * up * route_val);
        }
    }
}

__global__ void bf16_topk_w2_accum_f32_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const int32_t* __restrict__ indices,
    const uintptr_t* __restrict__ w2_ptrs,
    float* __restrict__ y_accum,
    int topk,
    int local_start,
    int n_local,
    int dim,
    int inter_dim
) {
    int out_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float acc = 0.0f;

    for (int top_idx = 0; top_idx < topk; ++top_idx) {
        int expert_id = indices[top_idx];
        int local_e = expert_id - local_start;
        if (local_e < 0 || local_e >= n_local) {
            continue;
        }
        uintptr_t w2_addr = w2_ptrs[local_e];
        if (w2_addr == 0) {
            continue;
        }

        const __nv_bfloat16* hidden_row = hidden + top_idx * inter_dim;
        const __nv_bfloat16* w2_row =
            reinterpret_cast<const __nv_bfloat16*>(w2_addr) + out_idx * inter_dim;

        for (int k = tid; k < inter_dim; k += blockDim.x) {
            acc += __bfloat162float(hidden_row[k]) * __bfloat162float(w2_row[k]);
        }
    }

    smem[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y_accum[out_idx] += __bfloat162float(__float2bfloat16(smem[0]));
    }
}

// Sparse attention for single-token decode: computes attention over topk KV positions.
// Uses online softmax within each head for numerical stability and minimal memory.
// One CUDA block per head; processes topk positions in tiles to keep KV in shared memory.
// Constraints: head_dim <= 512 (256 threads × 2 elements/thread), TILE=32.
// The caller must synchronize the stream before reading the output.
__global__ __launch_bounds__(256, 2) void sparse_attn_decode_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,         // [n_heads, head_dim]
    const __nv_bfloat16* __restrict__ kv_cache,  // [cache_size, head_dim]
    const int32_t* __restrict__ topk_idx,        // [topk]
    const float* __restrict__ attn_sink,         // [n_heads]
    float softmax_scale,
    __nv_bfloat16* __restrict__ out,             // [n_heads, head_dim]
    int n_heads,
    int head_dim,
    int topk,
    int cache_size
) {
    int h = blockIdx.x;  // one block per head
    if (h >= n_heads) return;

    int tid = threadIdx.x;
    constexpr int TILE = 32;
    constexpr int THREADS = 256;
    constexpr int THREADS_PER_DOT = THREADS / TILE;  // 8 threads per dot product

    extern __shared__ char smem_raw[];
    float* q_s = reinterpret_cast<float*>(smem_raw);
    __nv_bfloat16* kv_tile = reinterpret_cast<__nv_bfloat16*>(smem_raw + head_dim * sizeof(float));
    // Remaining shared memory used for per-thread partial scores
    float* partial = reinterpret_cast<float*>(smem_raw + head_dim * sizeof(float) + TILE * head_dim * sizeof(__nv_bfloat16));

    // Load q for this head into shared memory, pre-applying softmax_scale
    for (int i = tid; i < head_dim; i += THREADS) {
        q_s[i] = __bfloat162float(q[h * head_dim + i]) * softmax_scale;
    }
    __syncthreads();

    // Online softmax state in registers
    float acc_o[2];  // each thread accumulates 2 elements of output (head_dim <= 512, 256 threads)
    float max_score = -1e30f;
    float sum_exp = 0.0f;
    #pragma unroll
    for (int j = 0; j < 2; j++) acc_o[j] = 0.0f;

    // Process topk in tiles
    for (int t_start = 0; t_start < topk; t_start += TILE) {
        int t_end = min(t_start + TILE, topk);
        int t_count = t_end - t_start;

        // Cooperative load of kv tile into shared memory
        for (int i = tid; i < t_count * head_dim; i += THREADS) {
            int t_idx = i / head_dim;
            int d_idx = i % head_dim;
            int kv_idx = topk_idx[t_start + t_idx];
            if (kv_idx >= 0 && kv_idx < cache_size) {
                kv_tile[t_idx * head_dim + d_idx] = kv_cache[kv_idx * head_dim + d_idx];
            } else {
                kv_tile[t_idx * head_dim + d_idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Each group of THREADS_PER_DOT threads computes one dot product
        int dot_idx = tid / THREADS_PER_DOT;  // which tile element this thread works on
        int lane = tid % THREADS_PER_DOT;      // position within the group

        if (dot_idx < t_count) {
            int kv_idx = topk_idx[t_start + dot_idx];
            float score = 0.0f;

            if (kv_idx >= 0 && kv_idx < cache_size) {
                for (int d = lane; d < head_dim; d += THREADS_PER_DOT) {
                    score += q_s[d] * __bfloat162float(kv_tile[dot_idx * head_dim + d]);
                }
            }

            // Warp shuffle reduction within the group — mask only the 8 lanes of this group
            int group_in_warp = dot_idx % (32 / THREADS_PER_DOT);
            unsigned active_mask = 0xFFu << (group_in_warp * THREADS_PER_DOT);
            #pragma unroll
            for (int offset = THREADS_PER_DOT / 2; offset > 0; offset >>= 1) {
                score += __shfl_xor_sync(active_mask, score, offset);
            }

            // Write score to shared memory (only the first lane in each group)
            // softmax_scale was already baked into q_s, so no extra multiply here
            if (lane == 0) {
                partial[dot_idx] = score;
            }
        }
        __syncthreads();

        // Online softmax update — all threads participate
        for (int t = 0; t < t_count; t++) {
            float score = partial[t];
            int kv_idx = topk_idx[t_start + t];

            if (kv_idx < 0 || kv_idx >= cache_size) continue;

            float new_max = fmaxf(max_score, score);
            float scale = expf(max_score - new_max);

            // Rescale accumulators
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                acc_o[j] *= scale;
            }
            sum_exp *= scale;
            max_score = new_max;

            float exp_score = expf(score - max_score);

            // Accumulate weighted KV — each thread owns 2 output dims
            for (int j = 0; j < 2; j++) {
                int d = tid * 2 + j;
                if (d < head_dim) {
                    acc_o[j] += exp_score * __bfloat162float(kv_tile[t * head_dim + d]);
                }
            }
            sum_exp += exp_score;
        }
        __syncthreads();
    }

    // Include attn_sink — per-head scalar that absorbs excess probability mass
    {
        float sink_val = attn_sink[h];
        if (sink_val > max_score) {
            float scale = expf(max_score - sink_val);
            #pragma unroll
            for (int j = 0; j < 2; j++) acc_o[j] *= scale;
            sum_exp *= scale;
            max_score = sink_val;
        }
        sum_exp += expf(sink_val - max_score);
    }

    // Normalize and write output
    float inv_sum = 1.0f / fmaxf(sum_exp, 1e-10f);
    for (int j = 0; j < 2; j++) {
        int d = tid * 2 + j;
        if (d < head_dim) {
            out[h * head_dim + d] = __float2bfloat16(acc_o[j] * inv_sum);
        }
    }
}

extern "C" int ds_v4_sparse_attn_decode_bf16(
    const void* q_bf16,
    const void* kv_cache_bf16,
    const void* topk_idx_i32,
    const void* attn_sink_f32,
    float softmax_scale,
    void* out_bf16,
    int n_heads,
    int head_dim,
    int topk,
    int cache_size,
    void* stream_ptr
) {
    set_last_error("");

    if (!q_bf16 || !kv_cache_bf16 || !topk_idx_i32 || !attn_sink_f32 || !out_bf16) {
        set_last_error("null pointer");
        return 1;
    }
    if (n_heads <= 0 || head_dim <= 0 || topk <= 0 || cache_size <= 0) {
        set_last_error("invalid shape");
        return 2;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 256;
    constexpr int TILE = 32;
    size_t shared_bytes = head_dim * sizeof(float)           // q_s
                        + TILE * head_dim * sizeof(__nv_bfloat16)  // kv_tile
                        + TILE * sizeof(float);               // partial scores

    sparse_attn_decode_bf16_kernel<<<n_heads, threads, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_bf16),
        reinterpret_cast<const __nv_bfloat16*>(kv_cache_bf16),
        reinterpret_cast<const int32_t*>(topk_idx_i32),
        reinterpret_cast<const float*>(attn_sink_f32),
        softmax_scale,
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        n_heads,
        head_dim,
        topk,
        cache_size
    );

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        set_last_cuda_error(launch_err);
        return 3;
    }
    // No cudaStreamSynchronize here — the kernel runs on the caller's stream
    // and downstream kernels on the same stream will implicitly wait for it.
    // The caller is responsible for syncing before reading the output on the CPU.
    return 0;
}

extern "C" int ds_v4_fp4_dequant_gemm_bf16(
    const void* x_bf16,
    const void* packed_w_fp4,
    const void* scales_fp32,
    void* y_bf16,
    int tokens,
    int in_dim,
    int out_dim,
    int scale_rows,
    int scale_cols,
    int group_size,
    void* stream_ptr
) {
    set_last_error("");

    if (!x_bf16 || !packed_w_fp4 || !scales_fp32 || !y_bf16) {
        set_last_error("null pointer");
        return 1;
    }
    if (tokens <= 0 || in_dim <= 0 || out_dim <= 0 || scale_rows <= 0 || scale_cols <= 0 || group_size <= 0) {
        set_last_error("invalid shape");
        return 2;
    }
    if ((in_dim & 1) != 0) {
        set_last_error("in_dim must be even for packed fp4");
        return 3;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    dim3 grid(out_dim, tokens);
    constexpr int threads = 256;
    size_t shared_bytes = threads * sizeof(float);

    fp4_dequant_gemm_bf16_kernel<<<grid, threads, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<const uint8_t*>(packed_w_fp4),
        reinterpret_cast<const float*>(scales_fp32),
        reinterpret_cast<__nv_bfloat16*>(y_bf16),
        tokens,
        in_dim,
        out_dim,
        scale_rows,
        scale_cols,
        group_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_cuda_error(err);
        return 4;
    }
    return 0;
}

extern "C" int ds_v4_fp8_shared_expert_ffn_bf16(
    const void* x_bf16,
    const void* w1_fp8,
    const void* s1_fp32,
    const void* w2_fp8,
    const void* s2_fp32,
    const void* w3_fp8,
    const void* s3_fp32,
    void* hidden_bf16,
    void* y_bf16,
    int tokens,
    int dim,
    int inter_dim,
    int s1_rows,
    int s1_cols,
    int s2_rows,
    int s2_cols,
    int s3_rows,
    int s3_cols,
    int group_size,
    float swiglu_limit,
    void* stream_ptr
) {
    set_last_error("");

    if (!x_bf16 || !w1_fp8 || !s1_fp32 || !w2_fp8 || !s2_fp32 ||
        !w3_fp8 || !s3_fp32 || !hidden_bf16 || !y_bf16) {
        set_last_error("null pointer");
        return 1;
    }
    if (tokens <= 0 || dim <= 0 || inter_dim <= 0 || group_size <= 0) {
        set_last_error("invalid shape");
        return 2;
    }
    if (s1_rows <= 0 || s1_cols <= 0 || s2_rows <= 0 || s2_cols <= 0 || s3_rows <= 0 || s3_cols <= 0) {
        set_last_error("invalid scale shape");
        return 3;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 256;
    size_t shared_bytes = threads * sizeof(float);
    size_t shared_pair_bytes = threads * 2 * sizeof(float);

    fp8_shared_gate_up_fused_kernel<<<dim3(inter_dim, tokens), threads, shared_pair_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<const uint8_t*>(w1_fp8),
        reinterpret_cast<const float*>(s1_fp32),
        reinterpret_cast<const uint8_t*>(w3_fp8),
        reinterpret_cast<const float*>(s3_fp32),
        reinterpret_cast<__nv_bfloat16*>(hidden_bf16),
        tokens,
        dim,
        inter_dim,
        s1_rows,
        s1_cols,
        s3_rows,
        s3_cols,
        group_size,
        swiglu_limit
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_cuda_error(err);
        return 4;
    }

    fp8_dequant_gemm_bf16_kernel<<<dim3(dim, tokens), threads, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_bf16),
        reinterpret_cast<const uint8_t*>(w2_fp8),
        reinterpret_cast<const float*>(s2_fp32),
        reinterpret_cast<__nv_bfloat16*>(y_bf16),
        tokens,
        inter_dim,
        dim,
        s2_rows,
        s2_cols,
        group_size
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_cuda_error(err);
        return 5;
    }
    return 0;
}

extern "C" int ds_v4_fp4_expert_ffn_bf16(
    const void* x_bf16,
    const void* route_fp32,
    const void* w1_fp4,
    const void* s1_fp32,
    const void* w2_fp4,
    const void* s2_fp32,
    const void* w3_fp4,
    const void* s3_fp32,
    void* gate_f32,
    void* hidden_bf16,
    void* y_bf16,
    int tokens,
    int dim,
    int inter_dim,
    int s1_rows,
    int s1_cols,
    int s2_rows,
    int s2_cols,
    int s3_rows,
    int s3_cols,
    int group_size,
    float swiglu_limit,
    void* stream_ptr
) {
    set_last_error("");
    (void)gate_f32;

    if (!x_bf16 || !route_fp32 || !w1_fp4 || !s1_fp32 || !w2_fp4 || !s2_fp32 ||
        !w3_fp4 || !s3_fp32 || !hidden_bf16 || !y_bf16) {
        set_last_error("null pointer");
        return 1;
    }
    if (tokens <= 0 || dim <= 0 || inter_dim <= 0 || group_size <= 0) {
        set_last_error("invalid shape");
        return 2;
    }
    if ((dim & 1) != 0 || (inter_dim & 1) != 0) {
        set_last_error("dim and inter_dim must be even for packed fp4");
        return 3;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 256;
    size_t shared_bytes = threads * sizeof(float);
    size_t shared_pair_bytes = threads * 2 * sizeof(float);

    fp4_expert_gate_up_fused_kernel<<<dim3(inter_dim, tokens), threads, shared_pair_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<const float*>(route_fp32),
        reinterpret_cast<const uint8_t*>(w1_fp4),
        reinterpret_cast<const float*>(s1_fp32),
        reinterpret_cast<const uint8_t*>(w3_fp4),
        reinterpret_cast<const float*>(s3_fp32),
        reinterpret_cast<__nv_bfloat16*>(hidden_bf16),
        tokens,
        dim,
        inter_dim,
        s1_rows,
        s1_cols,
        s3_rows,
        s3_cols,
        group_size,
        swiglu_limit
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_cuda_error(err);
        return 4;
    }

    fp4_dequant_gemm_bf16_kernel<<<dim3(dim, tokens), threads, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_bf16),
        reinterpret_cast<const uint8_t*>(w2_fp4),
        reinterpret_cast<const float*>(s2_fp32),
        reinterpret_cast<__nv_bfloat16*>(y_bf16),
        tokens,
        inter_dim,
        dim,
        s2_rows,
        s2_cols,
        group_size
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_cuda_error(err);
        return 5;
    }
    return 0;
}

extern "C" int ds_v4_fp4_expert_ffn_accum_f32(
    const void* x_bf16,
    const void* route_fp32,
    const void* w1_fp4,
    const void* s1_fp32,
    const void* w2_fp4,
    const void* s2_fp32,
    const void* w3_fp4,
    const void* s3_fp32,
    void* hidden_bf16,
    void* y_accum_f32,
    int tokens,
    int dim,
    int inter_dim,
    int s1_rows,
    int s1_cols,
    int s2_rows,
    int s2_cols,
    int s3_rows,
    int s3_cols,
    int group_size,
    float swiglu_limit,
    void* stream_ptr
) {
    set_last_error("");

    if (!x_bf16 || !route_fp32 || !w1_fp4 || !s1_fp32 || !w2_fp4 || !s2_fp32 ||
        !w3_fp4 || !s3_fp32 || !hidden_bf16 || !y_accum_f32) {
        set_last_error("null pointer");
        return 1;
    }
    if (tokens <= 0 || dim <= 0 || inter_dim <= 0 || group_size <= 0) {
        set_last_error("invalid shape");
        return 2;
    }
    if ((dim & 1) != 0 || (inter_dim & 1) != 0) {
        set_last_error("dim and inter_dim must be even for packed fp4");
        return 3;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 256;
    size_t shared_bytes = threads * sizeof(float);
    size_t shared_pair_bytes = threads * 2 * sizeof(float);

    fp4_expert_gate_up_fused_kernel<<<dim3(inter_dim, tokens), threads, shared_pair_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<const float*>(route_fp32),
        reinterpret_cast<const uint8_t*>(w1_fp4),
        reinterpret_cast<const float*>(s1_fp32),
        reinterpret_cast<const uint8_t*>(w3_fp4),
        reinterpret_cast<const float*>(s3_fp32),
        reinterpret_cast<__nv_bfloat16*>(hidden_bf16),
        tokens,
        dim,
        inter_dim,
        s1_rows,
        s1_cols,
        s3_rows,
        s3_cols,
        group_size,
        swiglu_limit
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_cuda_error(err);
        return 4;
    }

    fp4_dequant_gemm_accum_f32_kernel<<<dim3(dim, tokens), threads, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_bf16),
        reinterpret_cast<const uint8_t*>(w2_fp4),
        reinterpret_cast<const float*>(s2_fp32),
        reinterpret_cast<float*>(y_accum_f32),
        tokens,
        inter_dim,
        dim,
        s2_rows,
        s2_cols,
        group_size
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_cuda_error(err);
        return 5;
    }
    return 0;
}

extern "C" int ds_v4_fp4_topk_expert_ffn_accum_f32(
    const void* x_bf16,
    const void* routes_fp32,
    const void* indices_i32,
    const void* w1_ptrs_i64,
    const void* s1_ptrs_i64,
    const void* w2_ptrs_i64,
    const void* s2_ptrs_i64,
    const void* w3_ptrs_i64,
    const void* s3_ptrs_i64,
    void* hidden_bf16,
    void* y_accum_f32,
    int topk,
    int local_start,
    int n_local,
    int dim,
    int inter_dim,
    int s1_rows,
    int s1_cols,
    int s2_rows,
    int s2_cols,
    int s3_rows,
    int s3_cols,
    int group_size,
    float swiglu_limit,
    void* stream_ptr
) {
    set_last_error("");

    if (!x_bf16 || !routes_fp32 || !indices_i32 || !w1_ptrs_i64 || !s1_ptrs_i64 ||
        !w2_ptrs_i64 || !s2_ptrs_i64 || !w3_ptrs_i64 || !s3_ptrs_i64 ||
        !hidden_bf16 || !y_accum_f32) {
        set_last_error("null pointer");
        return 1;
    }
    if (topk <= 0 || n_local <= 0 || dim <= 0 || inter_dim <= 0 || group_size <= 0) {
        set_last_error("invalid shape");
        return 2;
    }
    if ((dim & 1) != 0 || (inter_dim & 1) != 0) {
        set_last_error("dim and inter_dim must be even for packed fp4");
        return 3;
    }
    if (s1_rows <= 0 || s1_cols <= 0 || s2_rows <= 0 || s2_cols <= 0 || s3_rows <= 0 || s3_cols <= 0) {
        set_last_error("invalid scale shape");
        return 4;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 256;
    size_t shared_bytes = threads * sizeof(float);
    size_t shared_pair_bytes = threads * 2 * sizeof(float);
    int gate_up_grid_x = (inter_dim + FP4_TOPK_GU_TILE - 1) / FP4_TOPK_GU_TILE;
    size_t gate_up_smem = dim * sizeof(float);  // x_s in shared memory

    fp4_topk_gate_up_fused_kernel<<<dim3(gate_up_grid_x, topk), threads, gate_up_smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<const float*>(routes_fp32),
        reinterpret_cast<const int32_t*>(indices_i32),
        reinterpret_cast<const uintptr_t*>(w1_ptrs_i64),
        reinterpret_cast<const uintptr_t*>(s1_ptrs_i64),
        reinterpret_cast<const uintptr_t*>(w3_ptrs_i64),
        reinterpret_cast<const uintptr_t*>(s3_ptrs_i64),
        reinterpret_cast<__nv_bfloat16*>(hidden_bf16),
        topk,
        local_start,
        n_local,
        dim,
        inter_dim,
        s1_rows,
        s1_cols,
        s3_rows,
        s3_cols,
        group_size,
        swiglu_limit
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_cuda_error(err);
        return 5;
    }

    fp4_topk_w2_accum_f32_kernel<<<dim3(dim), threads, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_bf16),
        reinterpret_cast<const int32_t*>(indices_i32),
        reinterpret_cast<const uintptr_t*>(w2_ptrs_i64),
        reinterpret_cast<const uintptr_t*>(s2_ptrs_i64),
        reinterpret_cast<float*>(y_accum_f32),
        topk,
        local_start,
        n_local,
        dim,
        inter_dim,
        s2_rows,
        s2_cols,
        group_size
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_cuda_error(err);
        return 6;
    }
    return 0;
}

extern "C" int ds_v4_bf16_topk_expert_ffn_accum_f32(
    const void* x_bf16,
    const void* routes_fp32,
    const void* indices_i32,
    const void* w1_ptrs_i64,
    const void* w2_ptrs_i64,
    const void* w3_ptrs_i64,
    void* hidden_bf16,
    void* y_accum_f32,
    int topk,
    int local_start,
    int n_local,
    int dim,
    int inter_dim,
    float swiglu_limit,
    void* stream_ptr
) {
    set_last_error("");

    if (!x_bf16 || !routes_fp32 || !indices_i32 || !w1_ptrs_i64 ||
        !w2_ptrs_i64 || !w3_ptrs_i64 || !hidden_bf16 || !y_accum_f32) {
        set_last_error("null pointer");
        return 1;
    }
    if (topk <= 0 || n_local <= 0 || dim <= 0 || inter_dim <= 0) {
        set_last_error("invalid shape");
        return 2;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 256;
    int gate_up_grid_x = (inter_dim + FP4_TOPK_GU_TILE - 1) / FP4_TOPK_GU_TILE;
    size_t gate_up_smem = dim * sizeof(float);

    bf16_topk_gate_up_fused_kernel<<<dim3(gate_up_grid_x, topk), threads, gate_up_smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<const float*>(routes_fp32),
        reinterpret_cast<const int32_t*>(indices_i32),
        reinterpret_cast<const uintptr_t*>(w1_ptrs_i64),
        reinterpret_cast<const uintptr_t*>(w3_ptrs_i64),
        reinterpret_cast<__nv_bfloat16*>(hidden_bf16),
        topk,
        local_start,
        n_local,
        dim,
        inter_dim,
        swiglu_limit
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_cuda_error(err);
        return 3;
    }

    size_t shared_bytes = threads * sizeof(float);
    bf16_topk_w2_accum_f32_kernel<<<dim3(dim), threads, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_bf16),
        reinterpret_cast<const int32_t*>(indices_i32),
        reinterpret_cast<const uintptr_t*>(w2_ptrs_i64),
        reinterpret_cast<float*>(y_accum_f32),
        topk,
        local_start,
        n_local,
        dim,
        inter_dim
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_last_cuda_error(err);
        return 4;
    }
    return 0;
}

// Fused HC pre-processing: RMS norm + linear projection + Sinkhorn + weighted sum.
// Replaces ~10 PyTorch/TileLang kernel launches with a single kernel.
// One block per batch*seq element; N = bsz * seqlen (typically 1 for decode).
// Constraints: hc_mult <= 8, dim <= 8192.
__global__ __launch_bounds__(256, 2) void hc_pre_fused_kernel(
    const __nv_bfloat16* __restrict__ x,     // [N, hc_mult * dim]
    const float* __restrict__ hc_fn,         // [mix_hc, hc_mult * dim]
    const float* __restrict__ hc_scale,      // [3]
    const float* __restrict__ hc_base,       // [mix_hc]
    float eps,
    int N,                                    // batch * seq
    int hc_mult,
    int dim,
    int mix_hc,
    int sinkhorn_iters,
    __nv_bfloat16* __restrict__ y,           // [N, dim]
    float* __restrict__ pre_out,             // [N, hc_mult]
    float* __restrict__ post_out,            // [N, hc_mult]
    float* __restrict__ comb_out             // [N, hc_mult, hc_mult]
) {
    int n = blockIdx.x;
    if (n >= N) return;

    int tid = threadIdx.x;
    constexpr int THREADS = 256;
    int hc_dim = hc_mult * dim;

    const __nv_bfloat16* x_n = x + n * hc_dim;
    __nv_bfloat16* y_n = y + n * dim;
    float* pre_n = pre_out + n * hc_mult;
    float* post_n = post_out + n * hc_mult;
    float* comb_n = comb_out + n * hc_mult * hc_mult;

    // Shared memory: reduction buffer for RMS + mixes buffer
    extern __shared__ float smem[];
    float* reduce_buf = smem;        // [THREADS]
    float* mixes = smem + THREADS;    // [mix_hc]

    // Step 1: compute sum of squares for RMS norm
    float sum_sq = 0.0f;
    for (int i = tid; i < hc_dim; i += THREADS) {
        float v = __bfloat162float(x_n[i]);
        sum_sq += v * v;
    }
    reduce_buf[tid] = sum_sq;
    __syncthreads();

    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float rsqrt_val = rsqrtf(reduce_buf[0] / float(hc_dim) + eps);

    // Step 2: compute mixes = rsqrt * (x @ hc_fn^T), warp-level dot products
    int warp_id = tid / 32;
    int lane = tid % 32;
    int num_warps = THREADS / 32;
    int dots_per_warp = (mix_hc + num_warps - 1) / num_warps;

    for (int d = 0; d < dots_per_warp; d++) {
        int row = warp_id * dots_per_warp + d;
        float dot = 0.0f;
        if (row < mix_hc) {
            const float* hc_row = hc_fn + row * hc_dim;
            for (int i = lane; i < hc_dim; i += 32) {
                dot += __bfloat162float(x_n[i]) * hc_row[i];
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, offset);
            }
            if (lane == 0) {
                mixes[row] = dot * rsqrt_val;
            }
        }
    }
    __syncthreads();

    // Step 3: Sinkhorn iterations on mixes → pre, post, comb (thread 0 only)
    if (tid == 0) {
        // pre[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps
        for (int j = 0; j < hc_mult; j++) {
            float z = mixes[j] * hc_scale[0] + hc_base[j];
            // Numerically stable sigmoid
            float sig = (z >= 0.0f)
                ? 1.0f / (1.0f + expf(-z))
                : expf(z) / (1.0f + expf(z));
            pre_n[j] = sig + eps;
        }
        // post[j] = 2 * sigmoid(mixes[j+hc_mult] * scale[1] + base[j+hc_mult])
        for (int j = 0; j < hc_mult; j++) {
            float z = mixes[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult];
            float sig = (z >= 0.0f)
                ? 1.0f / (1.0f + expf(-z))
                : expf(z) / (1.0f + expf(z));
            post_n[j] = 2.0f * sig;
        }

        // comb = softmax(logits, dim=-1) + eps
        // logits[j,k] = mixes[2*hc_mult + j*hc_mult + k] * scale[2] + base[2*hc_mult + j*hc_mult + k]
        float comb_reg[64]; // max 8*8
        for (int j = 0; j < hc_mult; j++) {
            float row_max = -1e30f;
            int base_idx = 2 * hc_mult + j * hc_mult;
            for (int k = 0; k < hc_mult; k++) {
                float v = mixes[base_idx + k] * hc_scale[2] + hc_base[base_idx + k];
                if (v > row_max) row_max = v;
                comb_reg[j * hc_mult + k] = v;
            }
            float row_sum = 0.0f;
            for (int k = 0; k < hc_mult; k++) {
                float v = expf(comb_reg[j * hc_mult + k] - row_max);
                comb_reg[j * hc_mult + k] = v;
                row_sum += v;
            }
            float inv_sum = 1.0f / row_sum;
            for (int k = 0; k < hc_mult; k++) {
                comb_reg[j * hc_mult + k] = comb_reg[j * hc_mult + k] * inv_sum + eps;
            }
        }

        // Column normalization: comb /= (sum(comb, dim=-2) + eps)
        for (int k = 0; k < hc_mult; k++) {
            float col_sum = 0.0f;
            for (int j = 0; j < hc_mult; j++) {
                col_sum += comb_reg[j * hc_mult + k];
            }
            float inv_sum = 1.0f / (col_sum + eps);
            for (int j = 0; j < hc_mult; j++) {
                comb_reg[j * hc_mult + k] *= inv_sum;
            }
        }

        // Sinkhorn iterations
        for (int it = 0; it < sinkhorn_iters - 1; it++) {
            // Row normalization
            for (int j = 0; j < hc_mult; j++) {
                float row_sum = 0.0f;
                for (int k = 0; k < hc_mult; k++) {
                    row_sum += comb_reg[j * hc_mult + k];
                }
                float inv_sum = 1.0f / (row_sum + eps);
                for (int k = 0; k < hc_mult; k++) {
                    comb_reg[j * hc_mult + k] *= inv_sum;
                }
            }
            // Column normalization
            for (int k = 0; k < hc_mult; k++) {
                float col_sum = 0.0f;
                for (int j = 0; j < hc_mult; j++) {
                    col_sum += comb_reg[j * hc_mult + k];
                }
                float inv_sum = 1.0f / (col_sum + eps);
                for (int j = 0; j < hc_mult; j++) {
                    comb_reg[j * hc_mult + k] *= inv_sum;
                }
            }
        }

        // Write comb to global memory
        for (int i = 0; i < hc_mult * hc_mult; i++) {
            comb_n[i] = comb_reg[i];
        }
    }
    __syncthreads();

    // Step 4: Weighted sum y[d] = sum_j pre[j] * x[j*dim + d], write as bf16
    for (int d = tid; d < dim; d += THREADS) {
        float acc = 0.0f;
        for (int j = 0; j < hc_mult; j++) {
            acc += pre_n[j] * __bfloat162float(x_n[j * dim + d]);
        }
        y_n[d] = __float2bfloat16(acc);
    }
}

extern "C" int ds_v4_hc_pre_fused_bf16(
    const void* x_bf16,
    const void* hc_fn_fp32,
    const void* hc_scale_fp32,
    const void* hc_base_fp32,
    float eps,
    int N,
    int hc_mult,
    int dim,
    int sinkhorn_iters,
    void* y_bf16,
    void* pre_out_fp32,
    void* post_out_fp32,
    void* comb_out_fp32,
    void* stream_ptr
) {
    set_last_error("");

    if (!x_bf16 || !hc_fn_fp32 || !hc_scale_fp32 || !hc_base_fp32 ||
        !y_bf16 || !pre_out_fp32 || !post_out_fp32 || !comb_out_fp32) {
        set_last_error("null pointer");
        return 1;
    }
    if (N <= 0 || hc_mult <= 0 || hc_mult > 8 || dim <= 0 || sinkhorn_iters < 0) {
        set_last_error("invalid shape");
        return 2;
    }

    int mix_hc = (2 + hc_mult) * hc_mult;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    constexpr int threads = 256;
    size_t shared_bytes = threads * sizeof(float)      // reduce_buf
                        + mix_hc * sizeof(float);       // mixes

    hc_pre_fused_kernel<<<N, threads, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<const float*>(hc_fn_fp32),
        reinterpret_cast<const float*>(hc_scale_fp32),
        reinterpret_cast<const float*>(hc_base_fp32),
        eps,
        N,
        hc_mult,
        dim,
        mix_hc,
        sinkhorn_iters,
        reinterpret_cast<__nv_bfloat16*>(y_bf16),
        reinterpret_cast<float*>(pre_out_fp32),
        reinterpret_cast<float*>(post_out_fp32),
        reinterpret_cast<float*>(comb_out_fp32)
    );

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        set_last_cuda_error(launch_err);
        return 3;
    }
    return 0;
}

__global__ __launch_bounds__(256, 2) void hc_post_fused_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    __nv_bfloat16* __restrict__ y,
    int total,
    int hc_mult,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d = idx % dim;
    int h = (idx / dim) % hc_mult;
    int n = idx / (dim * hc_mult);

    const __nv_bfloat16* residual_n = residual + n * hc_mult * dim;
    const float* post_n = post + n * hc_mult;
    const float* comb_n = comb + n * hc_mult * hc_mult;
    const __nv_bfloat16* x_n = x + n * dim;

    float acc = post_n[h] * __bfloat162float(x_n[d]);
    for (int in_h = 0; in_h < hc_mult; ++in_h) {
        acc += comb_n[in_h * hc_mult + h] *
            __bfloat162float(residual_n[in_h * dim + d]);
    }
    y[idx] = __float2bfloat16(acc);
}

extern "C" int ds_v4_hc_post_fused_bf16(
    const void* x_bf16,
    const void* residual_bf16,
    const void* post_fp32,
    const void* comb_fp32,
    void* y_bf16,
    int N,
    int hc_mult,
    int dim,
    void* stream_ptr
) {
    set_last_error("");

    if (!x_bf16 || !residual_bf16 || !post_fp32 || !comb_fp32 || !y_bf16) {
        set_last_error("null pointer");
        return 1;
    }
    if (N <= 0 || hc_mult <= 0 || hc_mult > 8 || dim <= 0) {
        set_last_error("invalid shape");
        return 2;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 256;
    int total = N * hc_mult * dim;
    int blocks = (total + threads - 1) / threads;

    hc_post_fused_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<const __nv_bfloat16*>(residual_bf16),
        reinterpret_cast<const float*>(post_fp32),
        reinterpret_cast<const float*>(comb_fp32),
        reinterpret_cast<__nv_bfloat16*>(y_bf16),
        total,
        hc_mult,
        dim
    );

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        set_last_cuda_error(launch_err);
        return 3;
    }
    return 0;
}

// Fused attention output projection: einsum("bsgd,grd->bsgr") + RowParallelLinear wo_b.
// Replaces two kernel launches (einsum + F.linear) with one.
// One block per batch*seq element; designed for single-token decode (N=1).
// Uses warp-level cooperation for coalesced global memory reads.
__global__ __launch_bounds__(256, 1) void attn_o_proj_fused_kernel(
    const __nv_bfloat16* __restrict__ o,       // [N, n_groups, group_dim] — attention output
    const __nv_bfloat16* __restrict__ wo_a,    // [n_groups, lora_rank, group_dim] — LoRA-A weight
    const __nv_bfloat16* __restrict__ wo_b,    // [n_groups * lora_rank, dim] — LoRA-B weight
    int N,
    int n_groups,
    int lora_rank,
    int group_dim,
    int dim,
    __nv_bfloat16* __restrict__ y              // [N, dim]
) {
    int n = blockIdx.x;
    if (n >= N) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    constexpr int THREADS = 256;
    constexpr int WARP_SIZE = 32;
    int num_warps = THREADS / WARP_SIZE;
    int mid_dim = n_groups * lora_rank;

    const __nv_bfloat16* o_n = o + n * n_groups * group_dim;
    __nv_bfloat16* y_n = y + n * dim;

    extern __shared__ float smem[];
    float* o_fp32 = smem;                            // n_groups * group_dim
    float* mid_fp32 = smem + n_groups * group_dim;   // mid_dim

    // Step 1: Load o into shared memory as fp32
    int o_elems = n_groups * group_dim;
    for (int i = tid; i < o_elems; i += THREADS) {
        o_fp32[i] = __bfloat162float(o_n[i]);
    }
    __syncthreads();

    // Step 2: einsum("bsgd,grd->bsgr") — warp-cooperative dot products
    // Each warp processes a chunk of (g,r) pairs. Within a warp, 32 threads
    // cooperatively read wo_a[g,r,:] (coalesced) and compute the dot product.
    int pairs_per_warp = (mid_dim + num_warps - 1) / num_warps;
    for (int p = 0; p < pairs_per_warp; p++) {
        int idx = warp_id * pairs_per_warp + p;
        if (idx >= mid_dim) break;
        int g = idx / lora_rank;
        int r = idx % lora_rank;

        // Cooperative coalesced read of wo_a[g, r, :] by this warp
        const __nv_bfloat16* wa_row = wo_a + g * lora_rank * group_dim + r * group_dim;
        const float* o_g = o_fp32 + g * group_dim;

        float dot = 0.0f;
        for (int d = lane; d < group_dim; d += WARP_SIZE) {
            dot += o_g[d] * __bfloat162float(wa_row[d]);
        }
        // Warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, offset);
        }
        if (lane == 0) {
            mid_fp32[idx] = dot;
        }
    }
    __syncthreads();

    // Step 3: wo_b GEMV — row-based approach for coalesced reads
    // Each thread owns dim/THREADS output elements and reads rows i=tid,
    // i=tid+THREADS, ... of wo_b (contiguous, coalesced).
    int outputs_per_thread = dim / THREADS;
    int j_start = tid * outputs_per_thread;

    float partial[32]; // up to 32 outputs per thread (dim <= 8192)
    for (int k = 0; k < outputs_per_thread; k++) {
        partial[k] = 0.0f;
    }

    for (int i = tid; i < mid_dim; i += THREADS) {
        float m = mid_fp32[i];
        const __nv_bfloat16* wo_b_row = wo_b + i * dim + j_start;
        for (int k = 0; k < outputs_per_thread; k++) {
            partial[k] += m * __bfloat162float(wo_b_row[k]);
        }
    }

    for (int k = 0; k < outputs_per_thread; k++) {
        y_n[j_start + k] = __float2bfloat16(partial[k]);
    }
}

extern "C" int ds_v4_attn_o_proj_fused_bf16(
    const void* o_bf16,
    const void* wo_a_bf16,
    const void* wo_b_bf16,
    int N,
    int n_groups,
    int lora_rank,
    int group_dim,
    int dim,
    void* y_bf16,
    void* stream_ptr
) {
    set_last_error("");

    if (!o_bf16 || !wo_a_bf16 || !wo_b_bf16 || !y_bf16) {
        set_last_error("null pointer");
        return 1;
    }
    if (N <= 0 || n_groups <= 0 || lora_rank <= 0 || group_dim <= 0 || dim <= 0) {
        set_last_error("invalid shape");
        return 2;
    }

    int mid_dim = n_groups * lora_rank;
    int o_elems = n_groups * group_dim;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    constexpr int threads = 256;
    size_t shared_bytes = o_elems * sizeof(float)      // o_fp32
                        + mid_dim * sizeof(float);       // mid_fp32

    attn_o_proj_fused_kernel<<<N, threads, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(o_bf16),
        reinterpret_cast<const __nv_bfloat16*>(wo_a_bf16),
        reinterpret_cast<const __nv_bfloat16*>(wo_b_bf16),
        N,
        n_groups,
        lora_rank,
        group_dim,
        dim,
        reinterpret_cast<__nv_bfloat16*>(y_bf16)
    );

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        set_last_cuda_error(launch_err);
        return 3;
    }
    return 0;
}

// Fused Indexer scoring: einsum("hd,td->ht") + ReLU + weighted sum over heads.
// Computes index_score[t] = sum_h relu(dot(q[h], kv[t])) * weights[h].
// Single block, 256 threads; all cache positions processed cooperatively.
__global__ __launch_bounds__(256, 1) void indexer_score_fused_kernel(
    const __nv_bfloat16* __restrict__ q,        // [n_heads, head_dim]
    const __nv_bfloat16* __restrict__ kv_cache, // [cache_len, head_dim]
    const __nv_bfloat16* __restrict__ weights,  // [n_heads]
    int n_heads,
    int head_dim,
    int cache_len,
    float* __restrict__ index_score             // [cache_len]
) {
    int tid = threadIdx.x;
    constexpr int THREADS = 256;

    // Shared memory: q (fp32) + weights (fp32)
    extern __shared__ float smem[];
    float* q_s = smem;                      // n_heads * head_dim
    float* w_s = smem + n_heads * head_dim; // n_heads

    for (int i = tid; i < n_heads * head_dim; i += THREADS) {
        q_s[i] = __bfloat162float(q[i]);
    }
    if (tid < n_heads) {
        w_s[tid] = __bfloat162float(weights[tid]);
    }
    __syncthreads();

    // Each thread handles a subset of cache positions
    for (int t = tid; t < cache_len; t += THREADS) {
        float score = 0.0f;
        const __nv_bfloat16* kv_t = kv_cache + t * head_dim;
        for (int h = 0; h < n_heads; h++) {
            const float* q_h = q_s + h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_h[d] * __bfloat162float(kv_t[d]);
            }
            // ReLU + weighted sum
            if (dot > 0.0f) {
                score += dot * w_s[h];
            }
        }
        index_score[t] = score;
    }
}

extern "C" int ds_v4_indexer_score_fused_bf16(
    const void* q_bf16,
    const void* kv_cache_bf16,
    const void* weights_bf16,
    int n_heads,
    int head_dim,
    int cache_len,
    void* index_score_fp32,
    void* stream_ptr
) {
    set_last_error("");

    if (!q_bf16 || !kv_cache_bf16 || !weights_bf16 || !index_score_fp32) {
        set_last_error("null pointer");
        return 1;
    }
    if (n_heads <= 0 || head_dim <= 0 || cache_len <= 0) {
        set_last_error("invalid shape");
        return 2;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 256;
    size_t shared_bytes = n_heads * head_dim * sizeof(float)    // q_s
                        + n_heads * sizeof(float);               // w_s

    indexer_score_fused_kernel<<<1, threads, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_bf16),
        reinterpret_cast<const __nv_bfloat16*>(kv_cache_bf16),
        reinterpret_cast<const __nv_bfloat16*>(weights_bf16),
        n_heads,
        head_dim,
        cache_len,
        reinterpret_cast<float*>(index_score_fp32)
    );

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        set_last_cuda_error(launch_err);
        return 3;
    }
    return 0;
}

// Fully fused Indexer forward for single-token decode.
// Replaces ~7 kernel launches: wq_b GEMV + RoPE + weights_proj GEMV +
// einsum + ReLU + weighted_sum → single kernel launch.
__global__ __launch_bounds__(256, 1) void indexer_full_fused_kernel(
    const __nv_bfloat16* __restrict__ x,            // [dim]
    const __nv_bfloat16* __restrict__ qr,           // [q_lora_rank]
    const __nv_bfloat16* __restrict__ wq_b_w,       // [n_heads*head_dim, q_lora_rank]
    const __nv_bfloat16* __restrict__ weights_w,    // [n_heads, dim]
    const float* __restrict__ freqs_cis,            // [rd] cos/sin interleaved
    const __nv_bfloat16* __restrict__ kv_cache,     // [cache_len, head_dim]
    int dim,                // 4096
    int q_lora_rank,        // 1024
    int n_heads,            // 16
    int head_dim,           // 128
    int rd,                 // 64
    int cache_len,
    float weight_scale,     // softmax_scale * n_heads ** -0.5
    float* __restrict__ index_score  // [cache_len] fp32
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    constexpr int THREADS = 256;
    constexpr int WARP_SIZE = 32;
    int num_warps = THREADS / WARP_SIZE;

    extern __shared__ float smem[];
    float* q_s = smem;                              // n_heads * head_dim
    float* w_s = smem + n_heads * head_dim;          // n_heads
    float* x_s = smem + n_heads * head_dim + n_heads; // dim

    // Step 1: Load x into shared memory
    for (int i = tid; i < dim; i += THREADS) {
        x_s[i] = __bfloat162float(x[i]);
    }
    // Step 2: wq_b GEMV — q = qr @ wq_b^T, warps cooperate on outputs
    int q_elems = n_heads * head_dim;
    int q_per_warp = (q_elems + num_warps - 1) / num_warps;
    for (int p = 0; p < q_per_warp; p++) {
        int idx = warp_id * q_per_warp + p;
        float dot = 0.0f;
        if (idx < q_elems) {
            const __nv_bfloat16* w_row = wq_b_w + idx * q_lora_rank;
            for (int k = lane; k < q_lora_rank; k += WARP_SIZE) {
                dot += __bfloat162float(qr[k]) * __bfloat162float(w_row[k]);
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, offset);
        }
        if (lane == 0 && idx < q_elems) {
            q_s[idx] = dot;
        }
    }

    // Step 3: weights_proj GEMV — w = x @ weights_w^T, only n_heads outputs
    if (warp_id == 0) {
        for (int h = lane; h < n_heads; h += WARP_SIZE) {
            const __nv_bfloat16* w_row = weights_w + h * dim;
            float dot = 0.0f;
            for (int d = 0; d < dim; d++) {
                dot += x_s[d] * __bfloat162float(w_row[d]);
            }
            w_s[h] = dot * weight_scale;
        }
    }
    __syncthreads();

    // Step 4: RoPE on q[..., -rd:]
    for (int h = warp_id; h < n_heads; h += num_warps) {
        int q_off = h * head_dim + (head_dim - rd);
        for (int i = lane; i < rd / 2; i += WARP_SIZE) {
            float a = q_s[q_off + 2 * i];
            float b = q_s[q_off + 2 * i + 1];
            float c = freqs_cis[2 * i];
            float s = freqs_cis[2 * i + 1];
            q_s[q_off + 2 * i]     = a * c - b * s;
            q_s[q_off + 2 * i + 1] = a * s + b * c;
        }
    }
    __syncthreads();

    // Step 5: einsum + ReLU + weighted sum → index_score[t]
    for (int t = tid; t < cache_len; t += THREADS) {
        float score = 0.0f;
        const __nv_bfloat16* kv_t = kv_cache + t * head_dim;
        for (int h = 0; h < n_heads; h++) {
            const float* q_h = q_s + h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_h[d] * __bfloat162float(kv_t[d]);
            }
            if (dot > 0.0f) {
                score += dot * w_s[h];
            }
        }
        index_score[t] = score;
    }
}

extern "C" int ds_v4_indexer_full_fused_bf16(
    const void* x_bf16,
    const void* qr_bf16,
    const void* wq_b_w_bf16,
    const void* weights_w_bf16,
    const void* freqs_cis_fp32,
    const void* kv_cache_bf16,
    int dim,
    int q_lora_rank,
    int n_heads,
    int head_dim,
    int rd,
    int cache_len,
    float weight_scale,
    void* index_score_fp32,
    void* stream_ptr
) {
    set_last_error("");

    if (!x_bf16 || !qr_bf16 || !wq_b_w_bf16 || !weights_w_bf16 ||
        !freqs_cis_fp32 || !kv_cache_bf16 || !index_score_fp32) {
        set_last_error("null pointer");
        return 1;
    }
    if (dim <= 0 || q_lora_rank <= 0 || n_heads <= 0 || head_dim <= 0 ||
        rd <= 0 || cache_len <= 0) {
        set_last_error("invalid shape");
        return 2;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 256;
    size_t shared_bytes = n_heads * head_dim * sizeof(float)    // q_s
                        + n_heads * sizeof(float)                // w_s
                        + dim * sizeof(float);                   // x_s

    indexer_full_fused_kernel<<<1, threads, shared_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<const __nv_bfloat16*>(qr_bf16),
        reinterpret_cast<const __nv_bfloat16*>(wq_b_w_bf16),
        reinterpret_cast<const __nv_bfloat16*>(weights_w_bf16),
        reinterpret_cast<const float*>(freqs_cis_fp32),
        reinterpret_cast<const __nv_bfloat16*>(kv_cache_bf16),
        dim,
        q_lora_rank,
        n_heads,
        head_dim,
        rd,
        cache_len,
        weight_scale,
        reinterpret_cast<float*>(index_score_fp32)
    );

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        set_last_cuda_error(launch_err);
        return 3;
    }
    return 0;
}
