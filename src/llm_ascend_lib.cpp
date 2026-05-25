#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cctype>
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

static int env_int_or(const char* name, int fallback) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    return std::atoi(s);
}

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
    size_t token_bytes = 0;
    bool acl_ready = false;

    AscendEngine(const std::string& dir, int max_seq_)
        : device_id(env_int_or("ASCEND_DEVICE_ID", 0)),
          max_seq(max_seq_),
          model_dir(dir),
          config(load_config(dir)),
          metas(scan_safetensors(dir)) {
        if (max_seq <= 0) throw std::runtime_error("max_seq must be positive");

        auto t0 = Clock::now();
        check_acl(aclInit(nullptr), "aclInit");
        acl_ready = true;
        check_acl(aclrtSetDevice(device_id), "aclrtSetDevice");
        check_acl(aclrtCreateContext(&context, device_id), "aclrtCreateContext");
        check_acl(aclrtCreateStream(&stream), "aclrtCreateStream");

        token_bytes = static_cast<size_t>(max_seq) * sizeof(int);
        check_acl(aclrtMalloc(&d_tokens, token_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(tokens)");
        check_acl(aclrtMemset(d_tokens, token_bytes, 0, token_bytes), "aclrtMemset(tokens)");
        check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(init)");

        auto t1 = Clock::now();
        time_log("[Ascend][time] create engine, model=" + model_dir +
                 ", device=" + std::to_string(device_id) +
                 ", max_seq=" + std::to_string(max_seq) +
                 ", tensors=" + std::to_string(metas.size()) +
                 ", init_ms=" + std::to_string(elapsed_ms(t0, t1)));
    }

    ~AscendEngine() {
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
        auto t1 = Clock::now();
        time_log("[Ascend][time] prefill copied token_ids to HBM, tokens=" +
                 std::to_string(len) + ", copy_roundtrip_ms=" + std::to_string(elapsed_ms(t0, t1)));
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
