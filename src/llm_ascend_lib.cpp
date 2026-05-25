#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static thread_local std::string g_err;
using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static void reset_time_log() {
    std::ofstream f("log.txt", std::ios::trunc);
    if (f) f << "[Ascend][time] log reset\n";
}

static void time_log(const std::string& line) {
    std::cout << line << "\n";
    std::ofstream f("log.txt", std::ios::app);
    if (f) f << line << "\n";
}

static void check_acl(aclError ret, const char* what) {
    if (ret != ACL_SUCCESS) {
        std::ostringstream oss;
        oss << "AscendCL " << what << " failed, ret=" << static_cast<int>(ret);
        throw std::runtime_error(oss.str());
    }
}

struct ModelConfig {
    int n_layers = 28;
    int hidden = 3584;
    int n_heads = 28;
    int n_kv_heads = 4;
    int intermediate = 18944;
    int vocab_size = 152064;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
};

struct TensorMeta {
    std::string file;
    std::string dtype;
    std::vector<size_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
    uint64_t data_base = 0;
};

static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static std::string path_join(const std::string& a, const std::string& b) {
    return (!a.empty() && a.back() == '/') ? a + b : a + "/" + b;
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

static ModelConfig load_config(const std::string& dir) {
    ModelConfig c;
    const std::string path = path_join(dir, "config.json");
    const std::string json = read_text_file(path);
    if (json.empty()) {
        std::cout << "[Ascend] config.json not found, using compiled DeepSeek-7B defaults\n";
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

    std::cout << "[Ascend] config loaded: layers=" << c.n_layers
              << ", hidden=" << c.hidden
              << ", heads=" << c.n_heads
              << ", kv_heads=" << c.n_kv_heads
              << ", intermediate=" << c.intermediate
              << ", vocab=" << c.vocab_size << "\n";
    return c;
}

static uint64_t read_u64_le(std::ifstream& f) {
    unsigned char b[8];
    f.read(reinterpret_cast<char*>(b), 8);
    if (!f) throw std::runtime_error("read safetensors header length failed");
    uint64_t x = 0;
    for (int i = 0; i < 8; i++) x |= static_cast<uint64_t>(b[i]) << (8 * i);
    return x;
}

static std::vector<size_t> parse_shape(const std::string& s) {
    std::vector<size_t> v;
    std::stringstream ss(s);
    std::string it;
    while (std::getline(ss, it, ',')) {
        std::string t;
        for (char c : it) {
            if (!std::isspace(static_cast<unsigned char>(c))) t.push_back(c);
        }
        if (!t.empty()) v.push_back(static_cast<size_t>(std::stoull(t)));
    }
    return v;
}

static std::vector<std::string> list_safetensors(const std::string& dir) {
    DIR* dp = opendir(dir.c_str());
    if (!dp) throw std::runtime_error("cannot open model dir: " + dir);

    std::vector<std::string> fs;
    while (auto* e = readdir(dp)) {
        std::string n = e->d_name;
        if (ends_with(n, ".safetensors")) fs.push_back(path_join(dir, n));
    }
    closedir(dp);
    std::sort(fs.begin(), fs.end());
    if (fs.empty()) throw std::runtime_error("no safetensors found in " + dir);
    return fs;
}

static std::unordered_map<std::string, TensorMeta> scan_safetensors(const std::string& dir) {
    std::unordered_map<std::string, TensorMeta> m;
    std::regex re("\"([^\"]+)\"\\s*:\\s*\\{[^\\}]*?\"dtype\"\\s*:\\s*\"([^\"]+)\"[^\\}]*?\"shape\"\\s*:\\s*\\[([^\\]]*)\\][^\\}]*?\"data_offsets\"\\s*:\\s*\\[\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\]");

    for (const auto& p : list_safetensors(dir)) {
        std::ifstream f(p, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + p);
        const uint64_t hlen = read_u64_le(f);
        std::string h(hlen, '\0');
        f.read(&h[0], static_cast<std::streamsize>(hlen));
        if (!f) throw std::runtime_error("read safetensors header failed: " + p);

        int count = 0;
        for (auto it = std::sregex_iterator(h.begin(), h.end(), re); it != std::sregex_iterator(); ++it) {
            std::smatch x = *it;
            TensorMeta t;
            t.file = p;
            t.dtype = x[2].str();
            t.shape = parse_shape(x[3].str());
            t.begin = std::stoull(x[4].str());
            t.end = std::stoull(x[5].str());
            t.data_base = 8 + hlen;
            m[x[1].str()] = t;
            count++;
        }
        std::cout << "[Ascend] scanned " << p << ", tensors=" << count << "\n";
    }
    return m;
}

static std::vector<unsigned char> read_tensor_bytes(const TensorMeta& t) {
    std::ifstream f(t.file, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open tensor file " + t.file);
    const uint64_t off = t.data_base + t.begin;
    const uint64_t n = t.end - t.begin;
    f.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    std::vector<unsigned char> b(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(n));
    if (static_cast<uint64_t>(f.gcount()) != n) {
        throw std::runtime_error("read tensor bytes failed: " + t.file);
    }
    return b;
}

static std::string shape_string(const std::vector<size_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) oss << ",";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

static float bf16_to_float(uint16_t b) {
    uint32_t x = static_cast<uint32_t>(b) << 16;
    float y;
    std::memcpy(&y, &x, sizeof(float));
    return y;
}

static uint16_t float_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    const uint32_t lsb = (x >> 16) & 1u;
    const uint32_t rounding_bias = 0x7FFFu + lsb;
    return static_cast<uint16_t>((x + rounding_bias) >> 16);
}

static float f16_to_float(uint16_t h) {
    const uint16_t he = h & 0x7C00u;
    uint16_t hs = h & 0x03FFu;
    const uint32_t fs = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t fe = 0;
    uint32_t ff = 0;
    if (he == 0) {
        if (hs == 0) {
            uint32_t x = fs;
            float y;
            std::memcpy(&y, &x, sizeof(float));
            return y;
        }
        int shift = 0;
        while ((hs & 0x0400u) == 0) {
            hs <<= 1;
            shift++;
        }
        hs &= 0x03FFu;
        fe = static_cast<uint32_t>(127 - 15 - shift) << 23;
        ff = static_cast<uint32_t>(hs) << 13;
    } else if (he == 0x7C00u) {
        fe = 0xFFu << 23;
        ff = static_cast<uint32_t>(hs) << 13;
    } else {
        fe = static_cast<uint32_t>((he >> 10) + (127 - 15)) << 23;
        ff = static_cast<uint32_t>(hs) << 13;
    }
    uint32_t x = fs | fe | ff;
    float y;
    std::memcpy(&y, &x, sizeof(float));
    return y;
}

static uint16_t float_to_f16(float value) {
    uint32_t x;
    std::memcpy(&x, &value, sizeof(float));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int exp = static_cast<int>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return static_cast<uint16_t>(sign | half_mant);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    if (mant & 0x00001000u) half++;
    return static_cast<uint16_t>(half);
}

static int env_int_or(const char* name, int fallback) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    return std::atoi(s);
}

static std::string env_str_or(const char* name, const std::string& fallback) {
    const char* s = std::getenv(name);
    return (s && *s) ? std::string(s) : fallback;
}

struct DeviceTensor {
    TensorMeta meta;
    void* data = nullptr;
    size_t bytes = 0;
};

struct AscendEngine {
    int device_id = 0;
    int max_seq = 0;
    int prompt_len = 0;
    std::string model_dir;
    ModelConfig config;
    std::unordered_map<std::string, TensorMeta> metas;

    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    void* d_tokens = nullptr;
    void* d_hidden = nullptr;
    void* d_q = nullptr;
    size_t token_bytes = 0;
    size_t hidden_bytes = 0;
    size_t hidden_row_bytes = 0;
    size_t q_bytes = 0;
    size_t q_row_bytes = 0;
    bool acl_ready = false;
    std::unordered_map<std::string, DeviceTensor> d_weights;

    AscendEngine(const std::string& dir, int max_seq_)
        : device_id(env_int_or("ASCEND_DEVICE_ID", 0)),
          max_seq(max_seq_),
          model_dir(dir),
          config(load_config(dir)),
          metas(scan_safetensors(dir)) {
        if (max_seq <= 0) throw std::runtime_error("max_seq must be positive");

        auto t0 = Clock::now();
        std::cerr << "[Ascend] aclInit\n";
        check_acl(aclInit(nullptr), "aclInit");
        acl_ready = true;
        std::cerr << "[Ascend] aclrtSetDevice device=" << device_id << "\n";
        check_acl(aclrtSetDevice(device_id), "aclrtSetDevice");
        std::cerr << "[Ascend] aclrtCreateContext\n";
        check_acl(aclrtCreateContext(&context, device_id), "aclrtCreateContext");
        std::cerr << "[Ascend] aclrtCreateStream\n";
        check_acl(aclrtCreateStream(&stream), "aclrtCreateStream");

        token_bytes = static_cast<size_t>(max_seq) * sizeof(int);
        std::cerr << "[Ascend] aclrtMalloc tokens bytes=" << token_bytes << "\n";
        check_acl(aclrtMalloc(&d_tokens, token_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(tokens)");
        check_acl(aclrtMemset(d_tokens, token_bytes, 0, token_bytes), "aclrtMemset(tokens)");
        check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(init)");
        load_requested_weights();

        auto t1 = Clock::now();
        time_log("[Ascend][time] create engine, model=" + model_dir +
                 ", device=" + std::to_string(device_id) +
                 ", max_seq=" + std::to_string(max_seq) +
                 ", tensors=" + std::to_string(metas.size()) +
                 ", device_weights=" + std::to_string(d_weights.size()) +
                 ", init_ms=" + std::to_string(elapsed_ms(t0, t1)));
    }

    ~AscendEngine() {
        for (auto& kv : d_weights) {
            if (kv.second.data) {
                aclrtFree(kv.second.data);
                kv.second.data = nullptr;
            }
        }
        d_weights.clear();
        if (d_hidden) {
            aclrtFree(d_hidden);
            d_hidden = nullptr;
        }
        if (d_q) {
            aclrtFree(d_q);
            d_q = nullptr;
        }
        if (d_tokens) {
            aclrtFree(d_tokens);
            d_tokens = nullptr;
        }
        if (stream) {
            aclrtDestroyStream(stream);
            stream = nullptr;
        }
        if (context) {
            aclrtDestroyContext(context);
            context = nullptr;
        }
        if (acl_ready) {
            aclrtResetDevice(device_id);
            aclFinalize();
            acl_ready = false;
        }
    }

    bool load_weight_to_device(const std::string& name, bool required) {
        if (d_weights.find(name) != d_weights.end()) return true;
        auto it = metas.find(name);
        if (it == metas.end()) {
            if (required) throw std::runtime_error("missing tensor for Ascend HBM load: " + name);
            std::cout << "[Ascend] optional tensor not found, skip: " << name << "\n";
            return false;
        }

        const TensorMeta& meta = it->second;
        auto raw = read_tensor_bytes(meta);
        if (raw.empty()) throw std::runtime_error("empty tensor bytes: " + name);

        DeviceTensor dt;
        dt.meta = meta;
        dt.bytes = raw.size();

        auto t0 = Clock::now();
        check_acl(aclrtMalloc(&dt.data, dt.bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  ("aclrtMalloc(weight " + name + ")").c_str());
        check_acl(aclrtMemcpy(dt.data, dt.bytes, raw.data(), dt.bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                  ("aclrtMemcpy(H2D weight " + name + ")").c_str());

        const size_t check_n = std::min<size_t>(dt.bytes, 64);
        std::vector<unsigned char> check(check_n);
        check_acl(aclrtMemcpy(check.data(), check_n, dt.data, check_n, ACL_MEMCPY_DEVICE_TO_HOST),
                  ("aclrtMemcpy(D2H check weight " + name + ")").c_str());
        if (std::memcmp(check.data(), raw.data(), check_n) != 0) {
            aclrtFree(dt.data);
            throw std::runtime_error("Ascend weight H2D/D2H roundtrip failed: " + name);
        }

        auto t1 = Clock::now();
        time_log("[Ascend][time] weight loaded to HBM, name=" + name +
                 ", dtype=" + meta.dtype +
                 ", shape=" + shape_string(meta.shape) +
                 ", bytes=" + std::to_string(dt.bytes) +
                 ", h2d_ms=" + std::to_string(elapsed_ms(t0, t1)));

        d_weights.emplace(name, dt);
        return true;
    }

    void load_requested_weights() {
        const std::string mode = env_str_or("ASCEND_LOAD_WEIGHTS", "none");
        if (mode == "none" || mode == "0" || mode == "false") {
            std::cout << "[Ascend] ASCEND_LOAD_WEIGHTS=none, skip weight HBM load\n";
            return;
        }

        auto t0 = Clock::now();
        if (mode == "minimal") {
            load_weight_to_device("model.embed_tokens.weight", true);
            load_weight_to_device("model.norm.weight", true);
            load_weight_to_device("lm_head.weight", false);
        } else if (mode == "layer0") {
            load_weight_to_device("model.embed_tokens.weight", true);
            load_weight_to_device("model.layers.0.input_layernorm.weight", true);
            load_weight_to_device("model.layers.0.self_attn.q_proj.weight", true);
            load_weight_to_device("model.layers.0.self_attn.k_proj.weight", true);
            load_weight_to_device("model.layers.0.self_attn.v_proj.weight", true);
            load_weight_to_device("model.layers.0.self_attn.o_proj.weight", true);
            load_weight_to_device("model.layers.0.post_attention_layernorm.weight", true);
            load_weight_to_device("model.layers.0.mlp.gate_proj.weight", true);
            load_weight_to_device("model.layers.0.mlp.up_proj.weight", true);
            load_weight_to_device("model.layers.0.mlp.down_proj.weight", true);
            load_weight_to_device("model.norm.weight", true);
            load_weight_to_device("lm_head.weight", false);
        } else if (mode == "all") {
            std::vector<std::string> names;
            names.reserve(metas.size());
            for (const auto& kv : metas) names.push_back(kv.first);
            std::sort(names.begin(), names.end());
            for (const auto& name : names) load_weight_to_device(name, true);
        } else {
            throw std::runtime_error(
                "unsupported ASCEND_LOAD_WEIGHTS=" + mode +
                ", use one of: none, minimal, layer0, all");
        }

        auto t1 = Clock::now();
        time_log("[Ascend][time] requested weights loaded, mode=" + mode +
                 ", count=" + std::to_string(d_weights.size()) +
                 ", total_ms=" + std::to_string(elapsed_ms(t0, t1)));
    }

    void prefill(const int* ids, int len) {
        if (!ids) throw std::runtime_error("prefill ids is null");
        if (len <= 0) throw std::runtime_error("prefill length must be positive");
        if (len > max_seq) throw std::runtime_error("prefill length exceeds max_seq");

        auto t0 = Clock::now();
        const size_t bytes = static_cast<size_t>(len) * sizeof(int);
        check_acl(aclrtMemcpy(d_tokens, token_bytes, ids, bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(H2D tokens)");

        std::vector<int> roundtrip(len, -1);
        check_acl(aclrtMemcpy(roundtrip.data(), bytes, d_tokens, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(D2H tokens)");
        if (roundtrip.empty() || roundtrip.front() != ids[0] || roundtrip.back() != ids[len - 1]) {
            throw std::runtime_error("Ascend token H2D/D2H roundtrip check failed");
        }

        prompt_len = len;
        if (d_weights.find("model.embed_tokens.weight") != d_weights.end() &&
            env_str_or("ASCEND_RUN_EMBED", "1") != "0") {
            embedding_lookup(ids, len);
        }
        if (d_hidden && env_str_or("ASCEND_RUN_RMSNORM", "0") != "0") {
            rms_norm_reference(len);
        }
        if (d_hidden && env_str_or("ASCEND_RUN_QPROJ", "0") != "0") {
            q_proj_reference(len);
        }
        auto t1 = Clock::now();
        time_log("[Ascend][time] prefill copied token_ids to HBM, tokens=" +
                 std::to_string(len) + ", copy_roundtrip_ms=" + std::to_string(elapsed_ms(t0, t1)));
    }

    static size_t tensor_numel(const std::vector<size_t>& shape) {
        size_t n = 1;
        for (size_t x : shape) n *= x;
        return n;
    }

    static size_t dtype_size_bytes(const TensorMeta& meta) {
        const size_t n = tensor_numel(meta.shape);
        const size_t bytes = static_cast<size_t>(meta.end - meta.begin);
        if (n == 0 || bytes % n != 0) {
            throw std::runtime_error("cannot infer dtype size for tensor shape=" + shape_string(meta.shape));
        }
        return bytes / n;
    }

    static float load_scalar(const unsigned char* p, const std::string& dtype) {
        uint16_t u16 = 0;
        if (dtype == "BF16") {
            std::memcpy(&u16, p, sizeof(uint16_t));
            return bf16_to_float(u16);
        }
        if (dtype == "F16") {
            std::memcpy(&u16, p, sizeof(uint16_t));
            return f16_to_float(u16);
        }
        if (dtype == "F32" || dtype == "FLOAT32") {
            float f = 0.0f;
            std::memcpy(&f, p, sizeof(float));
            return f;
        }
        throw std::runtime_error("unsupported tensor dtype for scalar load: " + dtype);
    }

    static void store_scalar(unsigned char* p, const std::string& dtype, float value) {
        if (dtype == "BF16") {
            const uint16_t u16 = float_to_bf16(value);
            std::memcpy(p, &u16, sizeof(uint16_t));
            return;
        }
        if (dtype == "F16") {
            const uint16_t u16 = float_to_f16(value);
            std::memcpy(p, &u16, sizeof(uint16_t));
            return;
        }
        if (dtype == "F32" || dtype == "FLOAT32") {
            std::memcpy(p, &value, sizeof(float));
            return;
        }
        throw std::runtime_error("unsupported tensor dtype for scalar store: " + dtype);
    }

    const DeviceTensor* find_rms_norm_weight() const {
        auto it = d_weights.find("model.layers.0.input_layernorm.weight");
        if (it != d_weights.end()) return &it->second;
        it = d_weights.find("model.norm.weight");
        if (it != d_weights.end()) return &it->second;
        return nullptr;
    }

    void ensure_hidden_buffer(const TensorMeta& embed_meta) {
        if (d_hidden) return;
        if (embed_meta.shape.size() != 2) {
            throw std::runtime_error("embedding weight must be 2D, got shape=" + shape_string(embed_meta.shape));
        }
        const size_t dtype_bytes = dtype_size_bytes(embed_meta);
        hidden_row_bytes = embed_meta.shape[1] * dtype_bytes;
        hidden_bytes = static_cast<size_t>(max_seq) * hidden_row_bytes;
        check_acl(aclrtMalloc(&d_hidden, hidden_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(hidden states)");
        check_acl(aclrtMemset(d_hidden, hidden_bytes, 0, hidden_bytes),
                  "aclrtMemset(hidden states)");
        time_log("[Ascend][time] hidden buffer allocated, row_bytes=" +
                 std::to_string(hidden_row_bytes) +
                 ", total_bytes=" + std::to_string(hidden_bytes));
    }

    void embedding_lookup(const int* ids, int len) {
        auto it = d_weights.find("model.embed_tokens.weight");
        if (it == d_weights.end()) return;
        const DeviceTensor& embed = it->second;
        const TensorMeta& meta = embed.meta;
        ensure_hidden_buffer(meta);

        const size_t vocab = meta.shape[0];
        const size_t row_bytes = hidden_row_bytes;
        auto t0 = Clock::now();
        for (int i = 0; i < len; ++i) {
            const int token = ids[i];
            if (token < 0 || static_cast<size_t>(token) >= vocab) {
                throw std::runtime_error("token id out of embedding vocab range: " + std::to_string(token));
            }
            char* src = static_cast<char*>(embed.data) + static_cast<size_t>(token) * row_bytes;
            char* dst = static_cast<char*>(d_hidden) + static_cast<size_t>(i) * row_bytes;
            check_acl(aclrtMemcpy(dst, row_bytes, src, row_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE),
                      "aclrtMemcpy(D2D embedding row)");
        }

        const size_t check_n = std::min<size_t>(row_bytes, 64);
        std::vector<unsigned char> src_check(check_n);
        std::vector<unsigned char> dst_check(check_n);
        char* src0 = static_cast<char*>(embed.data) + static_cast<size_t>(ids[0]) * row_bytes;
        check_acl(aclrtMemcpy(src_check.data(), check_n, src0, check_n, ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(D2H embedding check src)");
        check_acl(aclrtMemcpy(dst_check.data(), check_n, d_hidden, check_n, ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(D2H embedding check dst)");
        if (std::memcmp(src_check.data(), dst_check.data(), check_n) != 0) {
            throw std::runtime_error("embedding D2D lookup verification failed");
        }

        auto t1 = Clock::now();
        time_log("[Ascend][time] embedding lookup D2D finished, tokens=" +
                 std::to_string(len) +
                 ", row_bytes=" + std::to_string(row_bytes) +
                 ", total_bytes=" + std::to_string(static_cast<size_t>(len) * row_bytes) +
                 ", elapsed_ms=" + std::to_string(elapsed_ms(t0, t1)));
    }

    void rms_norm_reference(int len) {
        const DeviceTensor* norm = find_rms_norm_weight();
        if (!norm) {
            throw std::runtime_error(
                "ASCEND_RUN_RMSNORM=1 requires model.norm.weight or "
                "model.layers.0.input_layernorm.weight to be loaded");
        }
        if (norm->meta.shape.size() != 1 || norm->meta.shape[0] != static_cast<size_t>(config.hidden)) {
            throw std::runtime_error("bad RMSNorm weight shape=" + shape_string(norm->meta.shape));
        }
        if (!d_hidden || hidden_row_bytes == 0) {
            throw std::runtime_error("RMSNorm requires hidden buffer from embedding lookup");
        }

        const size_t hidden = static_cast<size_t>(config.hidden);
        const TensorMeta& hidden_meta = d_weights.at("model.embed_tokens.weight").meta;
        const size_t hidden_dtype_bytes = dtype_size_bytes(hidden_meta);
        if (hidden_row_bytes != hidden * hidden_dtype_bytes) {
            throw std::runtime_error("hidden buffer row bytes mismatch before RMSNorm");
        }

        auto t0 = Clock::now();
        const size_t active_hidden_bytes = static_cast<size_t>(len) * hidden_row_bytes;
        std::vector<unsigned char> h_hidden(active_hidden_bytes);
        std::vector<unsigned char> h_norm(norm->bytes);

        check_acl(aclrtMemcpy(h_hidden.data(), active_hidden_bytes, d_hidden, active_hidden_bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(D2H hidden for RMSNorm)");
        check_acl(aclrtMemcpy(h_norm.data(), norm->bytes, norm->data, norm->bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(D2H norm weight for RMSNorm)");

        const size_t norm_dtype_bytes = dtype_size_bytes(norm->meta);
        float first_before = 0.0f;
        float first_after = 0.0f;
        for (int tok = 0; tok < len; ++tok) {
            unsigned char* row = h_hidden.data() + static_cast<size_t>(tok) * hidden_row_bytes;
            double sum_sq = 0.0;
            for (size_t j = 0; j < hidden; ++j) {
                const float x = load_scalar(row + j * hidden_dtype_bytes, hidden_meta.dtype);
                sum_sq += static_cast<double>(x) * static_cast<double>(x);
            }
            const float scale = 1.0f / std::sqrt(static_cast<float>(sum_sq / hidden) + config.rms_norm_eps);
            for (size_t j = 0; j < hidden; ++j) {
                const float x = load_scalar(row + j * hidden_dtype_bytes, hidden_meta.dtype);
                const float w = load_scalar(h_norm.data() + j * norm_dtype_bytes, norm->meta.dtype);
                const float y = x * scale * w;
                if (tok == 0 && j == 0) {
                    first_before = x;
                    first_after = y;
                }
                store_scalar(row + j * hidden_dtype_bytes, hidden_meta.dtype, y);
            }
        }

        check_acl(aclrtMemcpy(d_hidden, hidden_bytes, h_hidden.data(), active_hidden_bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(H2D hidden after RMSNorm)");

        auto t1 = Clock::now();
        time_log("[Ascend][time] rmsnorm reference finished, tokens=" +
                 std::to_string(len) +
                 ", hidden=" + std::to_string(hidden) +
                 ", norm_dtype=" + norm->meta.dtype +
                 ", hidden_dtype=" + hidden_meta.dtype +
                 ", first_before=" + std::to_string(first_before) +
                 ", first_after=" + std::to_string(first_after) +
                 ", elapsed_ms=" + std::to_string(elapsed_ms(t0, t1)));
    }

    void ensure_q_buffer(const TensorMeta& q_meta) {
        if (d_q) return;
        if (q_meta.shape.size() != 2) {
            throw std::runtime_error("q_proj weight must be 2D, got shape=" + shape_string(q_meta.shape));
        }
        if (q_meta.shape[1] != static_cast<size_t>(config.hidden)) {
            throw std::runtime_error("q_proj in_features mismatch, shape=" + shape_string(q_meta.shape));
        }
        const size_t dtype_bytes = dtype_size_bytes(q_meta);
        q_row_bytes = q_meta.shape[0] * dtype_bytes;
        q_bytes = static_cast<size_t>(max_seq) * q_row_bytes;
        check_acl(aclrtMalloc(&d_q, q_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(q buffer)");
        check_acl(aclrtMemset(d_q, q_bytes, 0, q_bytes), "aclrtMemset(q buffer)");
        time_log("[Ascend][time] q buffer allocated, row_bytes=" +
                 std::to_string(q_row_bytes) +
                 ", total_bytes=" + std::to_string(q_bytes));
    }

    void q_proj_reference(int len) {
        auto it = d_weights.find("model.layers.0.self_attn.q_proj.weight");
        if (it == d_weights.end()) {
            throw std::runtime_error(
                "ASCEND_RUN_QPROJ=1 requires ASCEND_LOAD_WEIGHTS=layer0 or all");
        }
        if (!d_hidden || hidden_row_bytes == 0) {
            throw std::runtime_error("q_proj requires hidden buffer");
        }

        const DeviceTensor& q_weight = it->second;
        const TensorMeta& q_meta = q_weight.meta;
        ensure_q_buffer(q_meta);

        const size_t out_dim = q_meta.shape[0];
        const size_t in_dim = q_meta.shape[1];
        const TensorMeta& hidden_meta = d_weights.at("model.embed_tokens.weight").meta;
        const size_t hidden_dtype_bytes = dtype_size_bytes(hidden_meta);
        const size_t q_dtype_bytes = dtype_size_bytes(q_meta);
        if (in_dim != static_cast<size_t>(config.hidden)) {
            throw std::runtime_error("q_proj in_dim mismatch");
        }

        auto t0 = Clock::now();
        const size_t active_hidden_bytes = static_cast<size_t>(len) * hidden_row_bytes;
        std::vector<unsigned char> h_hidden(active_hidden_bytes);
        std::vector<unsigned char> h_q_weight(q_weight.bytes);
        std::vector<unsigned char> h_q(static_cast<size_t>(len) * q_row_bytes);

        check_acl(aclrtMemcpy(h_hidden.data(), active_hidden_bytes, d_hidden, active_hidden_bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(D2H hidden for q_proj)");
        check_acl(aclrtMemcpy(h_q_weight.data(), q_weight.bytes, q_weight.data, q_weight.bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(D2H q_proj weight)");

        const int max_tokens = env_int_or("ASCEND_QPROJ_REF_TOKENS", 1);
        const int compute_tokens = std::max(0, std::min(len, max_tokens));
        float first_q = 0.0f;
        for (int tok = 0; tok < compute_tokens; ++tok) {
            const unsigned char* xrow = h_hidden.data() + static_cast<size_t>(tok) * hidden_row_bytes;
            unsigned char* qrow = h_q.data() + static_cast<size_t>(tok) * q_row_bytes;
            for (size_t out = 0; out < out_dim; ++out) {
                const unsigned char* wrow = h_q_weight.data() + out * in_dim * q_dtype_bytes;
                double acc = 0.0;
                for (size_t in = 0; in < in_dim; ++in) {
                    const float x = load_scalar(xrow + in * hidden_dtype_bytes, hidden_meta.dtype);
                    const float w = load_scalar(wrow + in * q_dtype_bytes, q_meta.dtype);
                    acc += static_cast<double>(x) * static_cast<double>(w);
                }
                const float y = static_cast<float>(acc);
                if (tok == 0 && out == 0) first_q = y;
                store_scalar(qrow + out * q_dtype_bytes, q_meta.dtype, y);
            }
        }

        if (compute_tokens > 0) {
            const size_t active_q_bytes = static_cast<size_t>(compute_tokens) * q_row_bytes;
            check_acl(aclrtMemcpy(d_q, q_bytes, h_q.data(), active_q_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                      "aclrtMemcpy(H2D q_proj output)");
        }

        auto t1 = Clock::now();
        time_log("[Ascend][time] q_proj reference finished, tokens_requested=" +
                 std::to_string(len) +
                 ", tokens_computed=" + std::to_string(compute_tokens) +
                 ", in_dim=" + std::to_string(in_dim) +
                 ", out_dim=" + std::to_string(out_dim) +
                 ", weight_dtype=" + q_meta.dtype +
                 ", first_q=" + std::to_string(first_q) +
                 ", elapsed_ms=" + std::to_string(elapsed_ms(t0, t1)));
    }

    int decode_one(int*) {
        throw std::runtime_error(
            "Ascend direct decode kernels are not implemented yet. "
            "Runtime, HBM allocation, safetensors scan, and prefill token copy are ready; "
            "next step is ACL/AscendC kernels for RMSNorm/RoPE/Attention/MLP/LMHead.");
    }
};

static int fail(const std::exception& e) {
    g_err = e.what();
    std::cerr << "[Ascend][error] " << g_err << "\n";
    return -1;
}

extern "C" {

void* llm_create(const char* model_dir, int max_seq) {
    try {
        reset_time_log();
        if (!model_dir) throw std::runtime_error("model_dir is null");
        return new AscendEngine(model_dir, max_seq);
    } catch (const std::exception& e) {
        fail(e);
        return nullptr;
    }
}

void llm_destroy(void* handle) {
    auto* e = reinterpret_cast<AscendEngine*>(handle);
    delete e;
}

int llm_prefill(void* handle, const int* input_ids, int n_tokens) {
    try {
        if (!handle) throw std::runtime_error("engine handle is null");
        auto* e = reinterpret_cast<AscendEngine*>(handle);
        e->prefill(input_ids, n_tokens);
        return 0;
    } catch (const std::exception& e) {
        return fail(e);
    }
}

int llm_decode_one(void* handle, int* out_token) {
    try {
        if (!handle) throw std::runtime_error("engine handle is null");
        auto* e = reinterpret_cast<AscendEngine*>(handle);
        return e->decode_one(out_token);
    } catch (const std::exception& e) {
        return fail(e);
    }
}

int llm_set_repetition_penalty(void*, float) {
    return 0;
}

const char* llm_last_error() {
    return g_err.c_str();
}

const char* llm_backend_name() {
    return "ascend-direct-acl";
}

}
