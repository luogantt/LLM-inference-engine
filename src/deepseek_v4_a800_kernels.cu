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
