#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
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

// DeepSeek-R1-Distill-Qwen-7B / Qwen2ForCausalLM 常见结构参数
constexpr int N_LAYERS = 28;
constexpr int HIDDEN = 3584;
constexpr int N_HEADS = 28;
constexpr int N_KV_HEADS = 4;
constexpr int HEAD_DIM = 128;
constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
constexpr int INTERMEDIATE = 18944;
constexpr int VOCAB_SIZE = 152064;
constexpr float DEFAULT_RMS_NORM_EPS = 1e-6f;
constexpr float DEFAULT_ROPE_THETA = 1000000.0f;

struct ModelConfig {
    int n_layers = N_LAYERS;
    int hidden = HIDDEN;
    int n_heads = N_HEADS;
    int n_kv_heads = N_KV_HEADS;
    int intermediate = INTERMEDIATE;
    int vocab_size = VOCAB_SIZE;
    float rms_norm_eps = DEFAULT_RMS_NORM_EPS;
    float rope_theta = DEFAULT_ROPE_THETA;
};

// Qwen2 长上下文模型常见 rope_theta 是 1000000。
// 如果你的 config.json 里不是这个值，需要改这里。
constexpr float ROPE_THETA = 1000000.0f;

struct TensorMeta {
    std::string file;
    std::string dtype;
    std::vector<size_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
    uint64_t data_base = 0;
};

static bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

static std::string read_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static int json_int_or(const std::string& json, const std::string& key, int fallback) {
    std::regex re("\"" + key + "\"\\s*:\\s*(-?\\d+)");
    std::smatch m;
    return std::regex_search(json, m, re) ? std::stoi(m[1].str()) : fallback;
}

static float json_float_or(const std::string& json, const std::string& key, float fallback) {
    std::regex re("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch m;
    return std::regex_search(json, m, re) ? std::stof(m[1].str()) : fallback;
}

static void require_config_value(const std::string& name, int actual, int expected) {
    if (actual != expected) {
        std::cerr << "unsupported config " << name << "=" << actual
                  << ", this binary was compiled for " << expected << "\n";
        std::exit(1);
    }
}

static ModelConfig load_config(const std::string& model_dir) {
    ModelConfig c;
    std::string json = read_text_file(join_path(model_dir, "config.json"));
    if (json.empty()) {
        std::cout << "config.json not found, using compiled defaults\n";
        return c;
    }

    c.n_layers = json_int_or(json, "num_hidden_layers", c.n_layers);
    c.hidden = json_int_or(json, "hidden_size", c.hidden);
    c.n_heads = json_int_or(json, "num_attention_heads", c.n_heads);
    c.n_kv_heads = json_int_or(json, "num_key_value_heads", c.n_kv_heads);
    c.intermediate = json_int_or(json, "intermediate_size", c.intermediate);
    c.vocab_size = json_int_or(json, "vocab_size", c.vocab_size);
    c.rms_norm_eps = json_float_or(json, "rms_norm_eps", c.rms_norm_eps);
    c.rope_theta = json_float_or(json, "rope_theta", c.rope_theta);

    require_config_value("num_hidden_layers", c.n_layers, N_LAYERS);
    require_config_value("hidden_size", c.hidden, HIDDEN);
    require_config_value("num_attention_heads", c.n_heads, N_HEADS);
    require_config_value("num_key_value_heads", c.n_kv_heads, N_KV_HEADS);
    require_config_value("intermediate_size", c.intermediate, INTERMEDIATE);
    require_config_value("vocab_size", c.vocab_size, VOCAB_SIZE);

    std::cout << "config loaded: rms_norm_eps=" << c.rms_norm_eps
              << ", rope_theta=" << c.rope_theta << "\n";
    return c;
}

static std::vector<std::string> list_safetensors(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp = opendir(dir.c_str());
    if (!dp) {
        std::cerr << "cannot open dir: " << dir << "\n";
        std::exit(1);
    }

    while (auto* ent = readdir(dp)) {
        std::string name = ent->d_name;
        if (ends_with(name, ".safetensors")) {
            files.push_back(join_path(dir, name));
        }
    }

    closedir(dp);
    std::sort(files.begin(), files.end());
    return files;
}

static uint64_t read_u64_le(std::ifstream& f) {
    uint8_t b[8];
    f.read(reinterpret_cast<char*>(b), 8);

    uint64_t x = 0;
    for (int i = 0; i < 8; ++i) {
        x |= static_cast<uint64_t>(b[i]) << (8 * i);
    }
    return x;
}

static std::vector<size_t> parse_shape(const std::string& s) {
    std::vector<size_t> shape;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ',')) {
        std::string t;
        for (char c : item) {
            if (!std::isspace(static_cast<unsigned char>(c))) t.push_back(c);
        }
        if (!t.empty()) {
            shape.push_back(static_cast<size_t>(std::stoull(t)));
        }
    }
    return shape;
}

static size_t numel(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (size_t x : shape) n *= x;
    return n;
}

static std::unordered_map<std::string, TensorMeta> scan_safetensors(const std::string& dir) {
    std::unordered_map<std::string, TensorMeta> map;

    auto files = list_safetensors(dir);
    if (files.empty()) {
        std::cerr << "no .safetensors files found in " << dir << "\n";
        std::exit(1);
    }

    std::regex item_re(
        "\"([^\"]+)\"\\s*:\\s*\\{[^\\}]*?"
        "\"dtype\"\\s*:\\s*\"([^\"]+)\"[^\\}]*?"
        "\"shape\"\\s*:\\s*\\[([^\\]]*)\\][^\\}]*?"
        "\"data_offsets\"\\s*:\\s*\\[\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\]"
    );

    for (const auto& path : files) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::cerr << "cannot open file: " << path << "\n";
            std::exit(1);
        }

        uint64_t header_len = read_u64_le(f);
        std::string header(header_len, '\0');
        f.read(&header[0], header_len);

        auto begin = std::sregex_iterator(header.begin(), header.end(), item_re);
        auto end = std::sregex_iterator();

        int count = 0;

        for (auto it = begin; it != end; ++it) {
            std::smatch m = *it;

            TensorMeta meta;
            std::string name = m[1].str();
            meta.file = path;
            meta.dtype = m[2].str();
            meta.shape = parse_shape(m[3].str());
            meta.begin = std::stoull(m[4].str());
            meta.end = std::stoull(m[5].str());
            meta.data_base = 8 + header_len;

            map[name] = meta;
            count++;
        }

        std::cout << "scanned " << path << ", tensors = " << count << "\n";
    }

    return map;
}

static float half_to_float(uint16_t h) {
    uint16_t h_exp = (h & 0x7C00u);
    uint16_t h_sig = (h & 0x03FFu);
    uint32_t f_sgn = static_cast<uint32_t>(h & 0x8000u) << 16;

    uint32_t f_exp;
    uint32_t f_sig;

    if (h_exp == 0) {
        if (h_sig == 0) {
            uint32_t f = f_sgn;
            float out;
            std::memcpy(&out, &f, 4);
            return out;
        }

        int shift = 0;
        while ((h_sig & 0x0400u) == 0) {
            h_sig <<= 1;
            shift++;
        }

        h_sig &= 0x03FFu;
        f_exp = static_cast<uint32_t>(127 - 15 - shift) << 23;
        f_sig = static_cast<uint32_t>(h_sig) << 13;
    } else if (h_exp == 0x7C00u) {
        f_exp = 0xFFu << 23;
        f_sig = static_cast<uint32_t>(h_sig) << 13;
    } else {
        f_exp = static_cast<uint32_t>((h_exp >> 10) + (127 - 15)) << 23;
        f_sig = static_cast<uint32_t>(h_sig) << 13;
    }

    uint32_t f = f_sgn | f_exp | f_sig;
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

static float bf16_to_float(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

static std::vector<unsigned char> read_tensor_bytes(const TensorMeta& meta) {
    std::ifstream f(meta.file, std::ios::binary);
    if (!f) {
        std::cerr << "cannot open tensor file: " << meta.file << "\n";
        std::exit(1);
    }

    uint64_t absolute_begin = meta.data_base + meta.begin;
    uint64_t bytes = meta.end - meta.begin;

    f.seekg(static_cast<std::streamoff>(absolute_begin), std::ios::beg);

    std::vector<unsigned char> raw(bytes);
    f.read(reinterpret_cast<char*>(raw.data()), bytes);

    if (static_cast<uint64_t>(f.gcount()) != bytes) {
        std::cerr << "failed to read tensor bytes from " << meta.file << "\n";
        std::exit(1);
    }

    return raw;
}

static float* load_tensor_gpu(
    const std::unordered_map<std::string, TensorMeta>& metas,
    const std::string& name
) {
    auto it = metas.find(name);
    if (it == metas.end()) {
        std::cerr << "missing tensor: " << name << "\n";
        std::exit(1);
    }

    const TensorMeta& meta = it->second;
    size_t n = numel(meta.shape);

    auto raw = read_tensor_bytes(meta);

    std::vector<float> h(n);

    if (meta.dtype == "BF16") {
        if (raw.size() != n * 2) {
            std::cerr << "bad BF16 size for " << name << "\n";
            std::exit(1);
        }

        const uint16_t* p = reinterpret_cast<const uint16_t*>(raw.data());
        for (size_t i = 0; i < n; ++i) {
            h[i] = bf16_to_float(p[i]);
        }
    } else if (meta.dtype == "F16") {
        if (raw.size() != n * 2) {
            std::cerr << "bad F16 size for " << name << "\n";
            std::exit(1);
        }

        const uint16_t* p = reinterpret_cast<const uint16_t*>(raw.data());
        for (size_t i = 0; i < n; ++i) {
            h[i] = half_to_float(p[i]);
        }
    } else if (meta.dtype == "F32") {
        if (raw.size() != n * 4) {
            std::cerr << "bad F32 size for " << name << "\n";
            std::exit(1);
        }

        std::memcpy(h.data(), raw.data(), n * 4);
    } else {
        std::cerr << "unsupported dtype " << meta.dtype << " for " << name << "\n";
        std::exit(1);
    }

    float* d = nullptr;
    CK(cudaMalloc(&d, n * sizeof(float)));
    CK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "loaded " << name << ", dtype=" << meta.dtype << ", numel=" << n << "\n";

    return d;
}

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

// PyTorch Linear 权重一般是 [out_features, in_features]
// y[out] = W[out, in] @ x[in] + bias[out]
__global__ void linear_kernel(const float* x, const float* W, const float* bias, float* y, int IN, int OUT) {
    __shared__ float sh[256];
    int o = blockIdx.x;
    int tid = threadIdx.x;
    if (o >= OUT) return;

    float sum = 0.0f;
    const float* row = W + static_cast<size_t>(o) * IN;

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

__global__ void rope_kernel(float* x, int n_heads, int pos, float rope_theta) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = n_heads * (HEAD_DIM / 2);
    if (pair >= total_pairs) return;

    int h = pair / (HEAD_DIM / 2);
    int p = pair % (HEAD_DIM / 2);

    int d0 = p;
    int d1 = p + (HEAD_DIM / 2);

    float inv_freq = powf(rope_theta, -static_cast<float>(2 * p) / HEAD_DIM);
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
        cache[static_cast<size_t>(pos) * dim + i] = x[i];
    }
}

// 单 token decode attention。
// 这是最朴素版本：每个输出元素重复计算 softmax。
// 慢，但代码最清楚。
__global__ void attention_scores_kernel(
    const float* q,
    const float* k_cache,
    float* scores,
    int pos,
    int max_seq
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int group_size = N_HEADS / N_KV_HEADS;
    int kv_h = h / group_size;
    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));

    for (int t = tid; t <= pos; t += blockDim.x) {
        float dot = 0.0f;
        const float* qh = q + h * HEAD_DIM;
        const float* kh = k_cache + static_cast<size_t>(t) * KV_DIM + kv_h * HEAD_DIM;
        for (int r = 0; r < HEAD_DIM; ++r) {
            dot += qh[r] * kh[r];
        }
        scores[static_cast<size_t>(h) * max_seq + t] = dot * scale;
    }
}

__global__ void attention_softmax_kernel(float* scores, int pos, int max_seq) {
    __shared__ float sh[256];
    int h = blockIdx.x;
    int tid = threadIdx.x;

    float max_score = -INFINITY;
    for (int t = tid; t <= pos; t += blockDim.x) {
        max_score = fmaxf(max_score, scores[static_cast<size_t>(h) * max_seq + t]);
    }
    sh[tid] = max_score;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] = fmaxf(sh[tid], sh[tid + stride]);
        __syncthreads();
    }

    max_score = sh[0];
    float denom = 0.0f;
    for (int t = tid; t <= pos; t += blockDim.x) {
        float e = expf(scores[static_cast<size_t>(h) * max_seq + t] - max_score);
        scores[static_cast<size_t>(h) * max_seq + t] = e;
        denom += e;
    }
    sh[tid] = denom;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    denom = sh[0];
    for (int t = tid; t <= pos; t += blockDim.x) {
        scores[static_cast<size_t>(h) * max_seq + t] /= denom;
    }
}

__global__ void attention_apply_kernel(
    const float* probs,
    const float* v_cache,
    float* ctx,
    int pos,
    int max_seq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HIDDEN) return;

    int d = idx % HEAD_DIM;
    int h = idx / HEAD_DIM;
    int group_size = N_HEADS / N_KV_HEADS;
    int kv_h = h / group_size;

    float out = 0.0f;
    for (int t = 0; t <= pos; ++t) {
        float p = probs[static_cast<size_t>(h) * max_seq + t];
        out += p * v_cache[static_cast<size_t>(t) * KV_DIM + kv_h * HEAD_DIM + d];
    }
    ctx[idx] = out;
}

struct LayerWeights {
    float* ln1 = nullptr;
    float* ln2 = nullptr;

    float* wq = nullptr;
    float* wk = nullptr;
    float* wv = nullptr;
    float* wo = nullptr;

    float* bq = nullptr;
    float* bk = nullptr;
    float* bv = nullptr;

    float* wgate = nullptr;
    float* wup = nullptr;
    float* wdown = nullptr;

    float* k_cache = nullptr;
    float* v_cache = nullptr;
};

struct Model {
    float* embed = nullptr;
    float* final_norm = nullptr;
    float* lm_head = nullptr;
    float rms_norm_eps = DEFAULT_RMS_NORM_EPS;
    float rope_theta = DEFAULT_ROPE_THETA;
    LayerWeights layers[N_LAYERS];
};

struct Work {
    float* x = nullptr;
    float* norm = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* ctx = nullptr;
    float* attn_out = nullptr;
    float* gate = nullptr;
    float* up = nullptr;
    float* mid = nullptr;
    float* mlp_out = nullptr;
    float* logits = nullptr;
    float* attn_scores = nullptr;
};

static std::string layer_name(int l, const std::string& suffix) {
    return "model.layers." + std::to_string(l) + "." + suffix;
}

static Model load_model(const std::string& model_dir, int max_seq) {
    ModelConfig cfg = load_config(model_dir);
    auto metas = scan_safetensors(model_dir);

    Model m{};
    m.rms_norm_eps = cfg.rms_norm_eps;
    m.rope_theta = cfg.rope_theta;

    m.embed = load_tensor_gpu(metas, "model.embed_tokens.weight");
    m.final_norm = load_tensor_gpu(metas, "model.norm.weight");
    m.lm_head = load_tensor_gpu(metas, "lm_head.weight");

    for (int l = 0; l < N_LAYERS; ++l) {
        std::cout << "\nloading layer " << l << "\n";

        auto& w = m.layers[l];

        w.ln1 = load_tensor_gpu(metas, layer_name(l, "input_layernorm.weight"));
        w.ln2 = load_tensor_gpu(metas, layer_name(l, "post_attention_layernorm.weight"));

        w.wq = load_tensor_gpu(metas, layer_name(l, "self_attn.q_proj.weight"));
        w.wk = load_tensor_gpu(metas, layer_name(l, "self_attn.k_proj.weight"));
        w.wv = load_tensor_gpu(metas, layer_name(l, "self_attn.v_proj.weight"));
        w.wo = load_tensor_gpu(metas, layer_name(l, "self_attn.o_proj.weight"));

        w.bq = load_tensor_gpu(metas, layer_name(l, "self_attn.q_proj.bias"));
        w.bk = load_tensor_gpu(metas, layer_name(l, "self_attn.k_proj.bias"));
        w.bv = load_tensor_gpu(metas, layer_name(l, "self_attn.v_proj.bias"));

        w.wgate = load_tensor_gpu(metas, layer_name(l, "mlp.gate_proj.weight"));
        w.wup = load_tensor_gpu(metas, layer_name(l, "mlp.up_proj.weight"));
        w.wdown = load_tensor_gpu(metas, layer_name(l, "mlp.down_proj.weight"));

        CK(cudaMalloc(&w.k_cache, static_cast<size_t>(max_seq) * KV_DIM * sizeof(float)));
        CK(cudaMalloc(&w.v_cache, static_cast<size_t>(max_seq) * KV_DIM * sizeof(float)));

        CK(cudaMemset(w.k_cache, 0, static_cast<size_t>(max_seq) * KV_DIM * sizeof(float)));
        CK(cudaMemset(w.v_cache, 0, static_cast<size_t>(max_seq) * KV_DIM * sizeof(float)));
    }

    return m;
}

static Work make_work(int max_seq) {
    Work w{};

    CK(cudaMalloc(&w.x, HIDDEN * sizeof(float)));
    CK(cudaMalloc(&w.norm, HIDDEN * sizeof(float)));
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
    CK(cudaMalloc(&w.attn_scores, static_cast<size_t>(N_HEADS) * max_seq * sizeof(float)));

    return w;
}

static void forward_token(const Model& m, Work& w, int token, int pos, int max_seq) {
    if (token < 0 || token >= VOCAB_SIZE) {
        std::cerr << "bad token id: " << token << "\n";
        std::exit(1);
    }

    int block = 256;

    embedding_kernel<<<(HIDDEN + block - 1) / block, block>>>(token, m.embed, w.x);

    for (int l = 0; l < N_LAYERS; ++l) {
        const auto& layer = m.layers[l];

        rmsnorm_kernel<<<1, block, block * sizeof(float)>>>(w.x, layer.ln1, w.norm, HIDDEN, m.rms_norm_eps);

        linear_kernel<<<HIDDEN, block>>>(w.norm, layer.wq, layer.bq, w.q, HIDDEN, HIDDEN);
        linear_kernel<<<KV_DIM, block>>>(w.norm, layer.wk, layer.bk, w.k, HIDDEN, KV_DIM);
        linear_kernel<<<KV_DIM, block>>>(w.norm, layer.wv, layer.bv, w.v, HIDDEN, KV_DIM);

        rope_kernel<<<(N_HEADS * (HEAD_DIM / 2) + block - 1) / block, block>>>(w.q, N_HEADS, pos, m.rope_theta);
        rope_kernel<<<(N_KV_HEADS * (HEAD_DIM / 2) + block - 1) / block, block>>>(w.k, N_KV_HEADS, pos, m.rope_theta);

        store_kv_kernel<<<(KV_DIM + block - 1) / block, block>>>(layer.k_cache, w.k, pos, KV_DIM);
        store_kv_kernel<<<(KV_DIM + block - 1) / block, block>>>(layer.v_cache, w.v, pos, KV_DIM);

        attention_scores_kernel<<<N_HEADS, block>>>(w.q, layer.k_cache, w.attn_scores, pos, max_seq);
        attention_softmax_kernel<<<N_HEADS, block>>>(w.attn_scores, pos, max_seq);
        attention_apply_kernel<<<(HIDDEN + block - 1) / block, block>>>(w.attn_scores, layer.v_cache, w.ctx, pos, max_seq);

        linear_kernel<<<HIDDEN, block>>>(w.ctx, layer.wo, nullptr, w.attn_out, HIDDEN, HIDDEN);
        add_kernel<<<(HIDDEN + block - 1) / block, block>>>(w.x, w.attn_out, HIDDEN);

        rmsnorm_kernel<<<1, block, block * sizeof(float)>>>(w.x, layer.ln2, w.norm, HIDDEN, m.rms_norm_eps);

        linear_kernel<<<INTERMEDIATE, block>>>(w.norm, layer.wgate, nullptr, w.gate, HIDDEN, INTERMEDIATE);
        linear_kernel<<<INTERMEDIATE, block>>>(w.norm, layer.wup, nullptr, w.up, HIDDEN, INTERMEDIATE);

        silu_mul_kernel<<<(INTERMEDIATE + block - 1) / block, block>>>(w.gate, w.up, w.mid, INTERMEDIATE);

        linear_kernel<<<HIDDEN, block>>>(w.mid, layer.wdown, nullptr, w.mlp_out, INTERMEDIATE, HIDDEN);
        add_kernel<<<(HIDDEN + block - 1) / block, block>>>(w.x, w.mlp_out, HIDDEN);
    }

    rmsnorm_kernel<<<1, block, block * sizeof(float)>>>(w.x, m.final_norm, w.norm, HIDDEN, m.rms_norm_eps);

    linear_kernel<<<VOCAB_SIZE, block>>>(w.norm, m.lm_head, nullptr, w.logits, HIDDEN, VOCAB_SIZE);

    CK(cudaDeviceSynchronize());
}

static std::vector<int> parse_tokens(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ',')) {
        std::string t;
        for (char c : item) {
            if (!std::isspace(static_cast<unsigned char>(c))) t.push_back(c);
        }
        if (!t.empty()) out.push_back(std::stoi(t));
    }

    return out;
}

static int argmax_cpu(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

struct Args {
    std::string model_dir = "/home/lg/推理/推理引擎/deepseek-r1-7b";
    std::string tokens = "1,2,3";
    int steps = 8;
    int max_seq = 128;
};

static Args parse_args(int argc, char** argv) {
    Args a;

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];

        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if (k == "--model") {
            a.model_dir = need_value(k);
        } else if (k == "--tokens") {
            a.tokens = need_value(k);
        } else if (k == "--steps") {
            a.steps = std::stoi(need_value(k));
        } else if (k == "--max-seq") {
            a.max_seq = std::stoi(need_value(k));
        } else if (k == "--help" || k == "-h") {
            std::cout
                << "usage:\n"
                << "  ./deepseek7b_token_cuda_infer "
                << "--model /home/lg/推理/推理引擎/deepseek-r1-7b "
                << "--tokens 1,2,3 --steps 8 --max-seq 128\n";
            std::exit(0);
        } else {
            std::cerr << "unknown arg: " << k << "\n";
            std::exit(1);
        }
    }

    return a;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    reset_time_log();

    if (args.max_seq <= 0) {
        std::cerr << "bad max_seq\n";
        return 1;
    }

    std::vector<int> tokens = parse_tokens(args.tokens);

    if (tokens.empty()) {
        std::cerr << "empty tokens\n";
        return 1;
    }

    if (static_cast<int>(tokens.size()) + args.steps > args.max_seq) {
        std::cerr << "tokens + steps exceeds max_seq\n";
        return 1;
    }

    std::cout << "model dir = " << args.model_dir << "\n";
    std::cout << "initial tokens = ";
    for (int t : tokens) std::cout << t << " ";
    std::cout << "\n";
    std::cout << "steps = " << args.steps << "\n";
    std::cout << "max_seq = " << args.max_seq << "\n";

    Model model = load_model(args.model_dir, args.max_seq);
    Work work = make_work(args.max_seq);

    std::vector<float> logits(VOCAB_SIZE);

    std::cout << "\nPrefill...\n";

    double prefill_forward_ms = 0.0;
    auto prefill_start = Clock::now();
    for (int pos = 0; pos < static_cast<int>(tokens.size()); ++pos) {
        std::cout << "prefill pos " << pos << ", token " << tokens[pos] << "\n";
        auto forward_start = Clock::now();
        forward_token(model, work, tokens[pos], pos, args.max_seq);
        double forward_ms = elapsed_ms(forward_start, Clock::now());
        prefill_forward_ms += forward_ms;
        {
            std::ostringstream os;
            os << "[time] prefill token " << pos << " forward_ms=" << forward_ms;
            time_log(os.str());
        }
    }
    double prefill_ms = elapsed_ms(prefill_start, Clock::now());
    int prefill_tokens = static_cast<int>(tokens.size());
    {
        std::ostringstream os;
        os << "[time] prefill total_ms=" << prefill_ms
           << ", forward_ms=" << prefill_forward_ms
           << ", tokens=" << prefill_tokens
           << ", tokens_per_s=" << (prefill_ms > 0.0 ? 1000.0 * prefill_tokens / prefill_ms : 0.0);
        time_log(os.str());
    }

    std::cout << "\nDecode...\n";

    double decode_ms_total = 0.0;
    double sample_ms_total = 0.0;
    double decode_forward_ms_total = 0.0;
    int decode_tokens = 0;
    for (int i = 0; i < args.steps; ++i) {
        auto decode_start = Clock::now();
        auto sample_start = Clock::now();
        CK(cudaMemcpy(logits.data(), work.logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        int next = argmax_cpu(logits);
        double sample_ms = elapsed_ms(sample_start, Clock::now());
        sample_ms_total += sample_ms;
        int pos = static_cast<int>(tokens.size());

        tokens.push_back(next);
        decode_tokens++;

        std::cout << "step " << i << ", next token = " << next << "\n";

        if (i + 1 < args.steps) {
            auto forward_start = Clock::now();
            forward_token(model, work, next, pos, args.max_seq);
            double forward_ms = elapsed_ms(forward_start, Clock::now());
            decode_forward_ms_total += forward_ms;
            {
                std::ostringstream os;
                os << "[time] decode token " << i
                   << " sample_ms=" << sample_ms
                   << ", forward_ms=" << forward_ms;
                time_log(os.str());
            }
        } else {
            {
                std::ostringstream os;
                os << "[time] decode token " << i
                   << " sample_ms=" << sample_ms
                   << ", forward_ms=0";
                time_log(os.str());
            }
        }
        double decode_ms = elapsed_ms(decode_start, Clock::now());
        decode_ms_total += decode_ms;
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

    std::cout << "\nGenerated token ids:\n";
    for (int t : tokens) std::cout << t << " ";
    std::cout << "\n";

    return 0;
}
