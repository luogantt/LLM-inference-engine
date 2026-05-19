#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#define CK(call)                                                        \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)      \
                      << " at " << __FILE__ << ":" << __LINE__          \
                      << std::endl;                                     \
            std::exit(1);                                               \
        }                                                               \
    } while (0)

using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static void reset_time_log() {
    std::ofstream f("log.txt", std::ios::trunc);
    if (f) f << "[time] log reset\n";
}

static void time_log(const std::string& line) {
    std::cout << line << "\n";
    std::ofstream f("log.txt", std::ios::app);
    if (f) f << line << "\n";
}

constexpr int VOCAB_SIZE = 128;
constexpr int MAX_SEQ = 64;
constexpr int N_LAYERS = 2;
constexpr int HIDDEN = 64;
constexpr int N_HEADS = 4;
constexpr int N_KV_HEADS = 1;
constexpr int HEAD_DIM = HIDDEN / N_HEADS;
constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
constexpr int INTERMEDIATE = 128;
constexpr float ROPE_THETA = 10000.0f;

static_assert(HIDDEN == N_HEADS * HEAD_DIM, "bad hidden size");

__global__ void embedding_kernel(const int token, const float* embed, float* x) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < HIDDEN) {
        x[d] = embed[token * HIDDEN + d];
    }
}

__global__ void rmsnorm_kernel(const float* x, const float* weight, float* y, int D, float eps) {
    extern __shared__ float sh[];
    int tid = threadIdx.x;

    float s = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = x[i];
        s += v * v;
    }

    sh[tid] = s;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh[tid] += sh[tid + stride];
        }
        __syncthreads();
    }

    float inv = rsqrtf(sh[0] / D + eps);

    for (int i = tid; i < D; i += blockDim.x) {
        y[i] = x[i] * inv * weight[i];
    }
}

// y[out] = W[out, in] @ x[in] + bias[out]
__global__ void linear_kernel(const float* x, const float* W, const float* bias, float* y, int IN, int OUT) {
    __shared__ float sh[256];
    int o = blockIdx.x;
    int tid = threadIdx.x;
    if (o >= OUT) return;

    float sum = 0.0f;
    const float* row = W + o * IN;

    for (int i = tid; i < IN; i += blockDim.x) {
        sum += row[i] * x[i];
    }

    sh[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh[tid] += sh[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias) sh[0] += bias[o];
        y[o] = sh[0];
    }
}

__global__ void add_kernel(float* x, const float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] += y[i];
}

__global__ void silu_mul_kernel(const float* gate, const float* up, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__global__ void rope_kernel(float* x, int n_heads, int pos) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = n_heads * (HEAD_DIM / 2);
    if (pair >= total_pairs) return;

    int h = pair / (HEAD_DIM / 2);
    int p = pair % (HEAD_DIM / 2);

    int d0 = p * 2;
    int d1 = d0 + 1;

    float inv_freq = powf(ROPE_THETA, -static_cast<float>(d0) / HEAD_DIM);
    float angle = pos * inv_freq;
    float c = cosf(angle);
    float s = sinf(angle);

    int base = h * HEAD_DIM;

    float a = x[base + d0];
    float b = x[base + d1];

    x[base + d0] = a * c - b * s;
    x[base + d1] = a * s + b * c;
}

__global__ void store_kv_kernel(float* cache, const float* x, int pos, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        cache[pos * dim + i] = x[i];
    }
}

__global__ void attention_kernel(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    float* ctx,
    int pos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HIDDEN) return;

    int d = idx % HEAD_DIM;
    int h = idx / HEAD_DIM;

    int group_size = N_HEADS / N_KV_HEADS;
    int kv_h = h / group_size;

    float max_score = -1e30f;

    for (int t = 0; t <= pos; ++t) {
        float dot = 0.0f;

        for (int r = 0; r < HEAD_DIM; ++r) {
            float qv = q[h * HEAD_DIM + r];
            float kv = k_cache[t * KV_DIM + kv_h * HEAD_DIM + r];
            dot += qv * kv;
        }

        float score = dot / sqrtf(static_cast<float>(HEAD_DIM));
        max_score = fmaxf(max_score, score);
    }

    float denom = 0.0f;
    float out = 0.0f;

    for (int t = 0; t <= pos; ++t) {
        float dot = 0.0f;

        for (int r = 0; r < HEAD_DIM; ++r) {
            float qv = q[h * HEAD_DIM + r];
            float kv = k_cache[t * KV_DIM + kv_h * HEAD_DIM + r];
            dot += qv * kv;
        }

        float score = dot / sqrtf(static_cast<float>(HEAD_DIM));
        float e = expf(score - max_score);
        denom += e;

        float vv = v_cache[t * KV_DIM + kv_h * HEAD_DIM + d];
        out += e * vv;
    }

    ctx[idx] = out / denom;
}

float* to_gpu(const std::vector<float>& h) {
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(float)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

std::vector<float> randn(int n, float scale = 0.02f) {
    static std::mt19937 gen(123);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
}

std::vector<float> ones(int n) {
    return std::vector<float>(n, 1.0f);
}

std::vector<float> zeros(int n) {
    return std::vector<float>(n, 0.0f);
}

struct Layer {
    float* ln1;
    float* ln2;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* bq;
    float* bk;
    float* bv;
    float* wgate;
    float* wup;
    float* wdown;
    float* k_cache;
    float* v_cache;
};

struct Model {
    float* embed;
    float* norm;
    float* lm_head;
    Layer layer[N_LAYERS];
};

struct Work {
    float* x;
    float* n;
    float* q;
    float* k;
    float* v;
    float* ctx;
    float* attn_out;
    float* gate;
    float* up;
    float* mid;
    float* mlp_out;
    float* logits;
};

Model make_model() {
    Model m{};

    m.embed = to_gpu(randn(VOCAB_SIZE * HIDDEN));
    m.norm = to_gpu(ones(HIDDEN));
    m.lm_head = to_gpu(randn(VOCAB_SIZE * HIDDEN));

    for (int l = 0; l < N_LAYERS; ++l) {
        auto& w = m.layer[l];

        w.ln1 = to_gpu(ones(HIDDEN));
        w.ln2 = to_gpu(ones(HIDDEN));

        w.wq = to_gpu(randn(HIDDEN * HIDDEN));
        w.wk = to_gpu(randn(KV_DIM * HIDDEN));
        w.wv = to_gpu(randn(KV_DIM * HIDDEN));
        w.wo = to_gpu(randn(HIDDEN * HIDDEN));

        w.bq = to_gpu(zeros(HIDDEN));
        w.bk = to_gpu(zeros(KV_DIM));
        w.bv = to_gpu(zeros(KV_DIM));

        w.wgate = to_gpu(randn(INTERMEDIATE * HIDDEN));
        w.wup = to_gpu(randn(INTERMEDIATE * HIDDEN));
        w.wdown = to_gpu(randn(HIDDEN * INTERMEDIATE));

        CK(cudaMalloc(&w.k_cache, MAX_SEQ * KV_DIM * sizeof(float)));
        CK(cudaMalloc(&w.v_cache, MAX_SEQ * KV_DIM * sizeof(float)));
    }

    return m;
}

Work make_work() {
    Work w{};

    CK(cudaMalloc(&w.x, HIDDEN * sizeof(float)));
    CK(cudaMalloc(&w.n, HIDDEN * sizeof(float)));
    CK(cudaMalloc(&w.q, HIDDEN * sizeof(float)));
    CK(cudaMalloc(&w.k, KV_DIM * sizeof(float)));
    CK(cudaMalloc(&w.v, KV_DIM * sizeof(float)));
    CK(cudaMalloc(&w.ctx, HIDDEN * sizeof(float)));
    CK(cudaMalloc(&w.attn_out, HIDDEN * sizeof(float)));
    CK(cudaMalloc(&w.gate, INTERMEDIATE * sizeof(float)));
    CK(cudaMalloc(&w.up, INTERMEDIATE * sizeof(float)));
    CK(cudaMalloc(&w.mid, INTERMEDIATE * sizeof(float)));
    CK(cudaMalloc(&w.mlp_out, HIDDEN * sizeof(float)));
    CK(cudaMalloc(&w.logits, VOCAB_SIZE * sizeof(float)));

    return w;
}

void forward_token(const Model& m, Work& w, int token, int pos) {
    int block = 256;

    embedding_kernel<<<(HIDDEN + block - 1) / block, block>>>(token, m.embed, w.x);

    for (int l = 0; l < N_LAYERS; ++l) {
        const Layer& layer = m.layer[l];

        rmsnorm_kernel<<<1, block, block * sizeof(float)>>>(w.x, layer.ln1, w.n, HIDDEN, 1e-6f);

        linear_kernel<<<HIDDEN, block>>>(w.n, layer.wq, layer.bq, w.q, HIDDEN, HIDDEN);
        linear_kernel<<<KV_DIM, block>>>(w.n, layer.wk, layer.bk, w.k, HIDDEN, KV_DIM);
        linear_kernel<<<KV_DIM, block>>>(w.n, layer.wv, layer.bv, w.v, HIDDEN, KV_DIM);

        rope_kernel<<<(N_HEADS * (HEAD_DIM / 2) + block - 1) / block, block>>>(w.q, N_HEADS, pos);
        rope_kernel<<<(N_KV_HEADS * (HEAD_DIM / 2) + block - 1) / block, block>>>(w.k, N_KV_HEADS, pos);

        store_kv_kernel<<<(KV_DIM + block - 1) / block, block>>>(layer.k_cache, w.k, pos, KV_DIM);
        store_kv_kernel<<<(KV_DIM + block - 1) / block, block>>>(layer.v_cache, w.v, pos, KV_DIM);

        attention_kernel<<<(HIDDEN + block - 1) / block, block>>>(w.q, layer.k_cache, layer.v_cache, w.ctx, pos);

        linear_kernel<<<HIDDEN, block>>>(w.ctx, layer.wo, nullptr, w.attn_out, HIDDEN, HIDDEN);
        add_kernel<<<(HIDDEN + block - 1) / block, block>>>(w.x, w.attn_out, HIDDEN);

        rmsnorm_kernel<<<1, block, block * sizeof(float)>>>(w.x, layer.ln2, w.n, HIDDEN, 1e-6f);

        linear_kernel<<<INTERMEDIATE, block>>>(w.n, layer.wgate, nullptr, w.gate, HIDDEN, INTERMEDIATE);
        linear_kernel<<<INTERMEDIATE, block>>>(w.n, layer.wup, nullptr, w.up, HIDDEN, INTERMEDIATE);

        silu_mul_kernel<<<(INTERMEDIATE + block - 1) / block, block>>>(w.gate, w.up, w.mid, INTERMEDIATE);

        linear_kernel<<<HIDDEN, block>>>(w.mid, layer.wdown, nullptr, w.mlp_out, INTERMEDIATE, HIDDEN);
        add_kernel<<<(HIDDEN + block - 1) / block, block>>>(w.x, w.mlp_out, HIDDEN);
    }

    rmsnorm_kernel<<<1, block, block * sizeof(float)>>>(w.x, m.norm, w.n, HIDDEN, 1e-6f);

    linear_kernel<<<VOCAB_SIZE, block>>>(w.n, m.lm_head, nullptr, w.logits, HIDDEN, VOCAB_SIZE);

    CK(cudaDeviceSynchronize());
}

int argmax_cpu(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

int main() {
    reset_time_log();
    Model m = make_model();
    Work w = make_work();

    std::vector<int> tokens = {1, 2, 3};
    std::vector<float> logits(VOCAB_SIZE);

    std::cout << "toy cuda infer, random weights\n";
    std::cout << "initial tokens: ";
    for (int t : tokens) std::cout << t << " ";
    std::cout << "\n";

    int steps = 16;

    double prefill_forward_ms = 0.0;
    auto prefill_start = Clock::now();
    for (int pos = 0; pos < static_cast<int>(tokens.size()); ++pos) {
        auto forward_start = Clock::now();
        forward_token(m, w, tokens[pos], pos);
        double forward_ms = elapsed_ms(forward_start, Clock::now());
        prefill_forward_ms += forward_ms;
        {
            std::ostringstream os;
            os << "[time] prefill token " << pos << " forward_ms=" << forward_ms;
            time_log(os.str());
        }
    }
    double prefill_ms = elapsed_ms(prefill_start, Clock::now());
    {
        std::ostringstream os;
        os << "[time] prefill total_ms=" << prefill_ms
           << ", forward_ms=" << prefill_forward_ms
           << ", tokens=" << tokens.size()
           << ", tokens_per_s=" << (prefill_ms > 0.0 ? 1000.0 * tokens.size() / prefill_ms : 0.0);
        time_log(os.str());
    }

    double decode_ms_total = 0.0;
    double sample_ms_total = 0.0;
    double decode_forward_ms_total = 0.0;
    int decode_tokens = 0;
    for (int i = 0; i < steps; ++i) {
        auto decode_start = Clock::now();
        auto sample_start = Clock::now();
        CK(cudaMemcpy(logits.data(), w.logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        int next = argmax_cpu(logits);
        double sample_ms = elapsed_ms(sample_start, Clock::now());
        sample_ms_total += sample_ms;
        int pos = static_cast<int>(tokens.size());
        tokens.push_back(next);
        decode_tokens++;

        std::cout << "step " << i << ", next token = " << next << "\n";

        if (pos >= MAX_SEQ) break;
        auto forward_start = Clock::now();
        forward_token(m, w, next, pos);
        double forward_ms = elapsed_ms(forward_start, Clock::now());
        decode_forward_ms_total += forward_ms;
        double decode_ms = elapsed_ms(decode_start, Clock::now());
        decode_ms_total += decode_ms;
        {
            std::ostringstream os;
            os << "[time] decode token " << i
               << " step_ms=" << decode_ms
               << ", sample_ms=" << sample_ms
               << ", forward_ms=" << forward_ms;
            time_log(os.str());
        }
    }
    {
        std::ostringstream os;
        os << "[time] decode total_ms=" << decode_ms_total
           << ", sample_ms=" << sample_ms_total
           << ", forward_ms=" << decode_forward_ms_total
           << ", tokens=" << decode_tokens
           << ", tokens_per_s=" << (decode_ms_total > 0.0 ? 1000.0 * decode_tokens / decode_ms_total : 0.0);
        time_log(os.str());
    }

    std::cout << "all tokens: ";
    for (int t : tokens) std::cout << t << " ";
    std::cout << "\n";

    return 0;
}
