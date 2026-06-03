#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

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
