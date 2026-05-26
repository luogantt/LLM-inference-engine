#include <acl/acl.h>

#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#if defined(__linux__)
#include <dlfcn.h>
#endif
#include <dirent.h>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define ASCEND_REF_HAVE_NEON 1
#else
#define ASCEND_REF_HAVE_NEON 0
#endif

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

static bool env_flag_enabled(const char* name, bool fallback) {
    const std::string value = env_str_or(name, fallback ? "1" : "0");
    return value != "0" && value != "false" && value != "False";
}

static bool ref_fast_dot_enabled() {
    static const bool enabled = env_flag_enabled("ASCEND_REF_FAST_DOT", true);
    return enabled;
}

static bool ref_dot4_enabled() {
    static const bool enabled = env_flag_enabled("ASCEND_REF_DOT4", false);
    return enabled;
}

static bool ref_neon_dot_enabled() {
    static const bool enabled = env_flag_enabled("ASCEND_REF_NEON_DOT", false);
    return enabled;
}

static bool ref_u16_weight_enabled() {
    static const bool enabled = env_flag_enabled("ASCEND_REF_U16_WEIGHTS", false);
    return enabled;
}

#if ASCEND_REF_HAVE_NEON
static float dot_product_neon(const float* __restrict__ x, const float* __restrict__ w, size_t n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(x + i + 0), vld1q_f32(w + i + 0));
        acc1 = vmlaq_f32(acc1, vld1q_f32(x + i + 4), vld1q_f32(w + i + 4));
        acc2 = vmlaq_f32(acc2, vld1q_f32(x + i + 8), vld1q_f32(w + i + 8));
        acc3 = vmlaq_f32(acc3, vld1q_f32(x + i + 12), vld1q_f32(w + i + 12));
    }
    float32x4_t accv = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float acc = vaddvq_f32(accv);
    for (; i < n; ++i) acc += x[i] * w[i];
    return acc;
}

static void dot_pair_neon(
    const float* __restrict__ x,
    const float* __restrict__ a,
    const float* __restrict__ b,
    size_t n,
    float& out_a,
    float& out_b) {
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    float32x4_t b0 = vdupq_n_f32(0.0f);
    float32x4_t b1 = vdupq_n_f32(0.0f);
    float32x4_t b2 = vdupq_n_f32(0.0f);
    float32x4_t b3 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        const float32x4_t x0 = vld1q_f32(x + i + 0);
        const float32x4_t x1 = vld1q_f32(x + i + 4);
        const float32x4_t x2 = vld1q_f32(x + i + 8);
        const float32x4_t x3 = vld1q_f32(x + i + 12);
        a0 = vmlaq_f32(a0, x0, vld1q_f32(a + i + 0));
        a1 = vmlaq_f32(a1, x1, vld1q_f32(a + i + 4));
        a2 = vmlaq_f32(a2, x2, vld1q_f32(a + i + 8));
        a3 = vmlaq_f32(a3, x3, vld1q_f32(a + i + 12));
        b0 = vmlaq_f32(b0, x0, vld1q_f32(b + i + 0));
        b1 = vmlaq_f32(b1, x1, vld1q_f32(b + i + 4));
        b2 = vmlaq_f32(b2, x2, vld1q_f32(b + i + 8));
        b3 = vmlaq_f32(b3, x3, vld1q_f32(b + i + 12));
    }
    float32x4_t av = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    float32x4_t bv = vaddq_f32(vaddq_f32(b0, b1), vaddq_f32(b2, b3));
    float acc_a = vaddvq_f32(av);
    float acc_b = vaddvq_f32(bv);
    for (; i < n; ++i) {
        const float xv = x[i];
        acc_a += xv * a[i];
        acc_b += xv * b[i];
    }
    out_a = acc_a;
    out_b = acc_b;
}

static void dot4_neon(
    const float* __restrict__ x,
    const float* __restrict__ w0,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ w3,
    size_t n,
    float& out0,
    float& out1,
    float& out2,
    float& out3) {
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        const float32x4_t x0 = vld1q_f32(x + i + 0);
        const float32x4_t x1 = vld1q_f32(x + i + 4);
        const float32x4_t x2 = vld1q_f32(x + i + 8);
        const float32x4_t x3 = vld1q_f32(x + i + 12);

        a0 = vmlaq_f32(a0, x0, vld1q_f32(w0 + i + 0));
        a0 = vmlaq_f32(a0, x1, vld1q_f32(w0 + i + 4));
        a0 = vmlaq_f32(a0, x2, vld1q_f32(w0 + i + 8));
        a0 = vmlaq_f32(a0, x3, vld1q_f32(w0 + i + 12));

        a1 = vmlaq_f32(a1, x0, vld1q_f32(w1 + i + 0));
        a1 = vmlaq_f32(a1, x1, vld1q_f32(w1 + i + 4));
        a1 = vmlaq_f32(a1, x2, vld1q_f32(w1 + i + 8));
        a1 = vmlaq_f32(a1, x3, vld1q_f32(w1 + i + 12));

        a2 = vmlaq_f32(a2, x0, vld1q_f32(w2 + i + 0));
        a2 = vmlaq_f32(a2, x1, vld1q_f32(w2 + i + 4));
        a2 = vmlaq_f32(a2, x2, vld1q_f32(w2 + i + 8));
        a2 = vmlaq_f32(a2, x3, vld1q_f32(w2 + i + 12));

        a3 = vmlaq_f32(a3, x0, vld1q_f32(w3 + i + 0));
        a3 = vmlaq_f32(a3, x1, vld1q_f32(w3 + i + 4));
        a3 = vmlaq_f32(a3, x2, vld1q_f32(w3 + i + 8));
        a3 = vmlaq_f32(a3, x3, vld1q_f32(w3 + i + 12));
    }

    float acc0 = vaddvq_f32(a0);
    float acc1 = vaddvq_f32(a1);
    float acc2 = vaddvq_f32(a2);
    float acc3 = vaddvq_f32(a3);
    for (; i < n; ++i) {
        const float xv = x[i];
        acc0 += xv * w0[i];
        acc1 += xv * w1[i];
        acc2 += xv * w2[i];
        acc3 += xv * w3[i];
    }
    out0 = acc0;
    out1 = acc1;
    out2 = acc2;
    out3 = acc3;
}
#endif

static float dot_product_reference(const float* __restrict__ x, const float* __restrict__ w, size_t n) {
    if (!ref_fast_dot_enabled()) {
        float acc = 0.0f;
        for (size_t i = 0; i < n; ++i) acc = std::fma(x[i], w[i], acc);
        return acc;
    }

#if ASCEND_REF_HAVE_NEON
    if (ref_neon_dot_enabled()) {
        return dot_product_neon(x, w, n);
    }
#endif

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    size_t i = 0;
    const size_t n4 = n & ~static_cast<size_t>(3);
    for (; i < n4; i += 4) {
        acc0 += x[i + 0] * w[i + 0];
        acc1 += x[i + 1] * w[i + 1];
        acc2 += x[i + 2] * w[i + 2];
        acc3 += x[i + 3] * w[i + 3];
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (; i < n; ++i) acc += x[i] * w[i];
    return acc;
}

static float dot_product_bf16_weight_reference(
    const float* __restrict__ x,
    const uint16_t* __restrict__ w,
    size_t n) {
    if (!ref_fast_dot_enabled()) {
        float acc = 0.0f;
        for (size_t i = 0; i < n; ++i) acc = std::fma(x[i], bf16_to_float(w[i]), acc);
        return acc;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    size_t i = 0;
    const size_t n4 = n & ~static_cast<size_t>(3);
    for (; i < n4; i += 4) {
        acc0 += x[i + 0] * bf16_to_float(w[i + 0]);
        acc1 += x[i + 1] * bf16_to_float(w[i + 1]);
        acc2 += x[i + 2] * bf16_to_float(w[i + 2]);
        acc3 += x[i + 3] * bf16_to_float(w[i + 3]);
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (; i < n; ++i) acc += x[i] * bf16_to_float(w[i]);
    return acc;
}

static float dot_product_f16_weight_reference(
    const float* __restrict__ x,
    const uint16_t* __restrict__ w,
    size_t n) {
    if (!ref_fast_dot_enabled()) {
        float acc = 0.0f;
        for (size_t i = 0; i < n; ++i) acc = std::fma(x[i], f16_to_float(w[i]), acc);
        return acc;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    size_t i = 0;
    const size_t n4 = n & ~static_cast<size_t>(3);
    for (; i < n4; i += 4) {
        acc0 += x[i + 0] * f16_to_float(w[i + 0]);
        acc1 += x[i + 1] * f16_to_float(w[i + 1]);
        acc2 += x[i + 2] * f16_to_float(w[i + 2]);
        acc3 += x[i + 3] * f16_to_float(w[i + 3]);
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (; i < n; ++i) acc += x[i] * f16_to_float(w[i]);
    return acc;
}

static float dot_product_u16_weight_reference(
    const float* __restrict__ x,
    const uint16_t* __restrict__ w,
    size_t n,
    bool bf16) {
    return bf16 ? dot_product_bf16_weight_reference(x, w, n)
                : dot_product_f16_weight_reference(x, w, n);
}

static void dot_pair_reference(
    const float* __restrict__ x,
    const float* __restrict__ a,
    const float* __restrict__ b,
    size_t n,
    float& out_a,
    float& out_b) {
    if (!ref_fast_dot_enabled()) {
        float acc_a = 0.0f;
        float acc_b = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const float xv = x[i];
            acc_a = std::fma(xv, a[i], acc_a);
            acc_b = std::fma(xv, b[i], acc_b);
        }
        out_a = acc_a;
        out_b = acc_b;
        return;
    }

#if ASCEND_REF_HAVE_NEON
    if (ref_neon_dot_enabled()) {
        dot_pair_neon(x, a, b, n, out_a, out_b);
        return;
    }
#endif

    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    float b0 = 0.0f;
    float b1 = 0.0f;
    float b2 = 0.0f;
    float b3 = 0.0f;
    size_t i = 0;
    const size_t n4 = n & ~static_cast<size_t>(3);
    for (; i < n4; i += 4) {
        const float x0 = x[i + 0];
        const float x1 = x[i + 1];
        const float x2 = x[i + 2];
        const float x3 = x[i + 3];
        a0 += x0 * a[i + 0];
        a1 += x1 * a[i + 1];
        a2 += x2 * a[i + 2];
        a3 += x3 * a[i + 3];
        b0 += x0 * b[i + 0];
        b1 += x1 * b[i + 1];
        b2 += x2 * b[i + 2];
        b3 += x3 * b[i + 3];
    }
    float acc_a = (a0 + a1) + (a2 + a3);
    float acc_b = (b0 + b1) + (b2 + b3);
    for (; i < n; ++i) {
        const float xv = x[i];
        acc_a += xv * a[i];
        acc_b += xv * b[i];
    }
    out_a = acc_a;
    out_b = acc_b;
}

static void dot_pair_bf16_weight_reference(
    const float* __restrict__ x,
    const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ b,
    size_t n,
    float& out_a,
    float& out_b) {
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    float b0 = 0.0f;
    float b1 = 0.0f;
    float b2 = 0.0f;
    float b3 = 0.0f;
    size_t i = 0;
    const size_t n4 = n & ~static_cast<size_t>(3);
    for (; i < n4; i += 4) {
        const float x0 = x[i + 0];
        const float x1 = x[i + 1];
        const float x2 = x[i + 2];
        const float x3 = x[i + 3];
        a0 += x0 * bf16_to_float(a[i + 0]);
        a1 += x1 * bf16_to_float(a[i + 1]);
        a2 += x2 * bf16_to_float(a[i + 2]);
        a3 += x3 * bf16_to_float(a[i + 3]);
        b0 += x0 * bf16_to_float(b[i + 0]);
        b1 += x1 * bf16_to_float(b[i + 1]);
        b2 += x2 * bf16_to_float(b[i + 2]);
        b3 += x3 * bf16_to_float(b[i + 3]);
    }
    float acc_a = (a0 + a1) + (a2 + a3);
    float acc_b = (b0 + b1) + (b2 + b3);
    for (; i < n; ++i) {
        const float xv = x[i];
        acc_a += xv * bf16_to_float(a[i]);
        acc_b += xv * bf16_to_float(b[i]);
    }
    out_a = acc_a;
    out_b = acc_b;
}

static void dot_pair_f16_weight_reference(
    const float* __restrict__ x,
    const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ b,
    size_t n,
    float& out_a,
    float& out_b) {
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    float b0 = 0.0f;
    float b1 = 0.0f;
    float b2 = 0.0f;
    float b3 = 0.0f;
    size_t i = 0;
    const size_t n4 = n & ~static_cast<size_t>(3);
    for (; i < n4; i += 4) {
        const float x0 = x[i + 0];
        const float x1 = x[i + 1];
        const float x2 = x[i + 2];
        const float x3 = x[i + 3];
        a0 += x0 * f16_to_float(a[i + 0]);
        a1 += x1 * f16_to_float(a[i + 1]);
        a2 += x2 * f16_to_float(a[i + 2]);
        a3 += x3 * f16_to_float(a[i + 3]);
        b0 += x0 * f16_to_float(b[i + 0]);
        b1 += x1 * f16_to_float(b[i + 1]);
        b2 += x2 * f16_to_float(b[i + 2]);
        b3 += x3 * f16_to_float(b[i + 3]);
    }
    float acc_a = (a0 + a1) + (a2 + a3);
    float acc_b = (b0 + b1) + (b2 + b3);
    for (; i < n; ++i) {
        const float xv = x[i];
        acc_a += xv * f16_to_float(a[i]);
        acc_b += xv * f16_to_float(b[i]);
    }
    out_a = acc_a;
    out_b = acc_b;
}

static void dot_pair_u16_weight_reference(
    const float* __restrict__ x,
    const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ b,
    size_t n,
    bool bf16,
    float& out_a,
    float& out_b) {
    if (bf16) {
        dot_pair_bf16_weight_reference(x, a, b, n, out_a, out_b);
    } else {
        dot_pair_f16_weight_reference(x, a, b, n, out_a, out_b);
    }
}

static void dot4_reference(
    const float* __restrict__ x,
    const float* __restrict__ w0,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ w3,
    size_t n,
    float& out0,
    float& out1,
    float& out2,
    float& out3) {
    if (!ref_fast_dot_enabled()) {
        float a0 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;
        float a3 = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const float xv = x[i];
            a0 = std::fma(xv, w0[i], a0);
            a1 = std::fma(xv, w1[i], a1);
            a2 = std::fma(xv, w2[i], a2);
            a3 = std::fma(xv, w3[i], a3);
        }
        out0 = a0;
        out1 = a1;
        out2 = a2;
        out3 = a3;
        return;
    }

#if ASCEND_REF_HAVE_NEON
    if (ref_neon_dot_enabled()) {
        dot4_neon(x, w0, w1, w2, w3, n, out0, out1, out2, out3);
        return;
    }
#endif

    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    size_t i = 0;
    const size_t n4 = n & ~static_cast<size_t>(3);
    for (; i < n4; i += 4) {
        const float x0 = x[i + 0];
        const float x1 = x[i + 1];
        const float x2 = x[i + 2];
        const float x3 = x[i + 3];
        a0 += x0 * w0[i + 0] + x1 * w0[i + 1] + x2 * w0[i + 2] + x3 * w0[i + 3];
        a1 += x0 * w1[i + 0] + x1 * w1[i + 1] + x2 * w1[i + 2] + x3 * w1[i + 3];
        a2 += x0 * w2[i + 0] + x1 * w2[i + 1] + x2 * w2[i + 2] + x3 * w2[i + 3];
        a3 += x0 * w3[i + 0] + x1 * w3[i + 1] + x2 * w3[i + 2] + x3 * w3[i + 3];
    }
    for (; i < n; ++i) {
        const float xv = x[i];
        a0 += xv * w0[i];
        a1 += xv * w1[i];
        a2 += xv * w2[i];
        a3 += xv * w3[i];
    }
    out0 = a0;
    out1 = a1;
    out2 = a2;
    out3 = a3;
}

struct DeviceTensor {
    TensorMeta meta;
    void* data = nullptr;
    size_t bytes = 0;
};

struct aclTensor;
struct aclOpExecutor;

struct AclRuntimeTensorApi {
    using CreateTensorFn = aclTensor* (*)(
        const int64_t* view_dims,
        uint64_t view_dim_num,
        aclDataType data_type,
        const int64_t* stride,
        int64_t offset,
        aclFormat format,
        const int64_t* storage_dims,
        uint64_t storage_dim_num,
        void* tensor_data);
    using DestroyTensorFn = void (*)(const aclTensor*);

    bool tried = false;
    bool ready = false;
    CreateTensorFn create_tensor = nullptr;
    DestroyTensorFn destroy_tensor = nullptr;
    std::string error;

    template <typename Fn>
    bool load_symbol(Fn& fn, void* handle, const char* name) {
#if defined(__linux__)
        fn = reinterpret_cast<Fn>(dlsym(handle, name));
        if (!fn) {
            error = std::string("missing ACL runtime tensor symbol ") + name;
            return false;
        }
        return true;
#else
        (void)fn;
        (void)handle;
        (void)name;
        error = "ACL runtime tensor dynamic loading is only supported on Linux";
        return false;
#endif
    }

    bool load(void* opapi_handle, std::string& reason) {
        if (tried) {
            if (!ready) reason = error;
            return ready;
        }
        tried = true;
#if defined(__linux__)
        void* handle = RTLD_DEFAULT;
        ready =
            load_symbol(create_tensor, handle, "aclCreateTensor") &&
            load_symbol(destroy_tensor, handle, "aclDestroyTensor");
        if (!ready && opapi_handle) {
            error.clear();
            ready =
                load_symbol(create_tensor, opapi_handle, "aclCreateTensor") &&
                load_symbol(destroy_tensor, opapi_handle, "aclDestroyTensor");
        }
        if (!ready) reason = error;
        return ready;
#else
        error = "ACL runtime tensor dynamic loading is only supported on Linux";
        reason = error;
        return false;
#endif
    }
};

static AclRuntimeTensorApi& global_acl_tensor_api() {
    static AclRuntimeTensorApi api;
    return api;
}

struct AclTensorGuard {
    aclTensor* tensor = nullptr;

    AclTensorGuard() = default;
    explicit AclTensorGuard(aclTensor* t) : tensor(t) {}
    AclTensorGuard(const AclTensorGuard&) = delete;
    AclTensorGuard& operator=(const AclTensorGuard&) = delete;

    AclTensorGuard(AclTensorGuard&& other) noexcept : tensor(other.tensor) {
        other.tensor = nullptr;
    }

    AclTensorGuard& operator=(AclTensorGuard&& other) noexcept {
        if (this != &other) {
            reset();
            tensor = other.tensor;
            other.tensor = nullptr;
        }
        return *this;
    }

    ~AclTensorGuard() {
        reset();
    }

    void reset() {
        if (tensor) {
            AclRuntimeTensorApi& api = global_acl_tensor_api();
            if (api.destroy_tensor) api.destroy_tensor(tensor);
            tensor = nullptr;
        }
    }

    aclTensor* get() const {
        return tensor;
    }
};

struct AclnnApi {
    using MmGetWorkspaceSizeFn = int (*)(const aclTensor*, const aclTensor*, aclTensor*, int8_t, uint64_t*, aclOpExecutor**);
    using MmFn = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
    using SiluGetWorkspaceSizeFn = int (*)(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
    using SiluFn = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
    using MulGetWorkspaceSizeFn = int (*)(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
    using MulFn = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);

    bool tried = false;
    bool ready = false;
    void* handle = nullptr;
    MmGetWorkspaceSizeFn mm_ws = nullptr;
    MmFn mm = nullptr;
    SiluGetWorkspaceSizeFn silu_ws = nullptr;
    SiluFn silu = nullptr;
    MulGetWorkspaceSizeFn mul_ws = nullptr;
    MulFn mul = nullptr;
    std::string error;

    template <typename Fn>
    bool load_symbol(Fn& fn, const char* name) {
#if defined(__linux__)
        fn = reinterpret_cast<Fn>(dlsym(handle, name));
        if (!fn) {
            error = std::string("missing ACLNN symbol ") + name;
            return false;
        }
        return true;
#else
        (void)fn;
        (void)name;
        error = "ACLNN dynamic loading is only supported on Linux";
        return false;
#endif
    }

    bool load(std::string& reason) {
        if (tried) {
            if (!ready) reason = error;
            return ready;
        }
        tried = true;
#if defined(__linux__)
        handle = dlopen("libopapi.so", RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            const char* err = dlerror();
            error = std::string("dlopen libopapi.so failed: ") + (err ? err : "unknown error");
            reason = error;
            return false;
        }
        ready =
            load_symbol(mm_ws, "aclnnMmGetWorkspaceSize") &&
            load_symbol(mm, "aclnnMm") &&
            load_symbol(silu_ws, "aclnnSiluGetWorkspaceSize") &&
            load_symbol(silu, "aclnnSilu") &&
            load_symbol(mul_ws, "aclnnMulGetWorkspaceSize") &&
            load_symbol(mul, "aclnnMul");
        if (!ready) reason = error;
        return ready;
#else
        error = "ACLNN dynamic loading is only supported on Linux";
        reason = error;
        return false;
#endif
    }
};

static AclnnApi& global_aclnn_api() {
    static AclnnApi api;
    return api;
}

struct RefLayerProfile {
    double norm1_ms = 0.0;
    double q_ms = 0.0;
    double kv_ms = 0.0;
    double rope_ms = 0.0;
    double attn_ms = 0.0;
    double o_ms = 0.0;
    double norm2_ms = 0.0;
    double gate_up_ms = 0.0;
    double down_ms = 0.0;
    double total_ms = 0.0;

    void add(const RefLayerProfile& other) {
        norm1_ms += other.norm1_ms;
        q_ms += other.q_ms;
        kv_ms += other.kv_ms;
        rope_ms += other.rope_ms;
        attn_ms += other.attn_ms;
        o_ms += other.o_ms;
        norm2_ms += other.norm2_ms;
        gate_up_ms += other.gate_up_ms;
        down_ms += other.down_ms;
        total_ms += other.total_ms;
    }
};

struct DeviceMlpTiming {
    double gate_up_ms = 0.0;
    double down_ms = 0.0;
};

struct DeviceLinearTiming {
    double total_ms = 0.0;
};

struct DeviceWorkspaceGuard {
    std::vector<void*> ptrs;

    ~DeviceWorkspaceGuard() {
        for (void* p : ptrs) {
            if (p) aclrtFree(p);
        }
    }

    void* allocate(uint64_t bytes, const char* label) {
        if (bytes == 0) return nullptr;
        void* p = nullptr;
        check_acl(aclrtMalloc(&p, static_cast<size_t>(bytes), ACL_MEM_MALLOC_HUGE_FIRST), label);
        ptrs.push_back(p);
        return p;
    }
};

struct RefThreadPool {
    explicit RefThreadPool(int n_threads_) : n_threads(std::max(1, n_threads_)) {
        workers.reserve(static_cast<size_t>(n_threads));
        for (int tid = 0; tid < n_threads; ++tid) {
            workers.emplace_back([this, tid]() { worker_loop(tid); });
        }
    }

    ~RefThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mu);
            stopping = true;
            generation++;
        }
        cv_start.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) worker.join();
        }
    }

    void run(int active_threads, const std::function<void(int)>& fn) {
        active_threads = std::max(1, std::min(active_threads, n_threads));
        if (active_threads == 1) {
            fn(0);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mu);
            active = active_threads;
            pending = active_threads;
            job = fn;
            generation++;
        }
        cv_start.notify_all();
        std::unique_lock<std::mutex> lock(mu);
        cv_done.wait(lock, [this]() { return pending == 0; });
        job = nullptr;
    }

    int n_threads = 1;
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv_start;
    std::condition_variable cv_done;
    std::function<void(int)> job;
    size_t generation = 0;
    int active = 0;
    int pending = 0;
    bool stopping = false;

    void worker_loop(int tid) {
        size_t seen_generation = 0;
        while (true) {
            std::function<void(int)> local_job;
            int local_active = 0;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv_start.wait(lock, [this, &seen_generation]() {
                    return stopping || generation != seen_generation;
                });
                if (stopping) return;
                seen_generation = generation;
                local_job = job;
                local_active = active;
            }
            const bool participates = local_job && tid < local_active;
            if (participates) local_job(tid);
            if (participates) {
                std::lock_guard<std::mutex> lock(mu);
                pending--;
                if (pending == 0) cv_done.notify_one();
            }
        }
    }
};

struct AscendEngine {
    int device_id = 0;
    int max_seq = 0;
    int prompt_len = 0;
    float repetition_penalty = 1.1f;
    std::string model_dir;
    ModelConfig config;
    std::unordered_map<std::string, TensorMeta> metas;
    std::vector<unsigned char> seen_tokens;

    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    void* d_tokens = nullptr;
    void* d_hidden = nullptr;
    void* d_q = nullptr;
    void* d_k = nullptr;
    void* d_v = nullptr;
    void* d_mlp_x = nullptr;
    void* d_mlp_gate = nullptr;
    void* d_mlp_up = nullptr;
    void* d_mlp_mid = nullptr;
    void* d_mlp_out = nullptr;
    void* d_vec_mm_x = nullptr;
    void* d_vec_mm_out = nullptr;
    size_t token_bytes = 0;
    size_t hidden_bytes = 0;
    size_t hidden_row_bytes = 0;
    size_t q_bytes = 0;
    size_t q_row_bytes = 0;
    size_t k_bytes = 0;
    size_t k_row_bytes = 0;
    size_t v_bytes = 0;
    size_t v_row_bytes = 0;
    size_t mlp_hidden_bytes = 0;
    size_t mlp_intermediate_bytes = 0;
    std::string mlp_buffer_dtype;
    size_t vec_mm_x_bytes = 0;
    size_t vec_mm_out_bytes = 0;
    std::string vec_mm_buffer_dtype;
    bool acl_ready = false;
    bool aclnn_mlp_disabled = false;
    bool aclnn_mlp_notice_printed = false;
    bool aclnn_attn_proj_disabled = false;
    bool aclnn_attn_proj_notice_printed = false;
    bool aclnn_lm_head_disabled = false;
    bool aclnn_lm_head_notice_printed = false;
    mutable RefThreadPool* ref_thread_pool = nullptr;
    mutable int ref_thread_pool_size = 0;
    std::unordered_map<std::string, DeviceTensor> d_weights;
    std::unordered_map<std::string, std::vector<unsigned char>> h_weight_raw_cache;
    std::unordered_map<std::string, std::vector<float>> h_weight_cache;
    std::unordered_map<std::string, std::vector<uint16_t>> h_weight_u16_cache;
    std::vector<float> layer0_k_cache;
    std::vector<float> layer0_v_cache;
    int layer0_kv_cached_len = 0;
    int layer0_kv_dim = 0;
    std::vector<std::vector<float>> full_k_cache;
    std::vector<std::vector<float>> full_v_cache;
    int full_ref_cached_len = 0;
    int full_ref_kv_dim = 0;
    std::vector<float> full_last_hidden;

    bool ref_cache_log_enabled() const {
        const std::string explicit_flag = env_str_or("ASCEND_REF_CACHE_LOG", "");
        if (!explicit_flag.empty()) {
            return explicit_flag != "0" && explicit_flag != "false" && explicit_flag != "False";
        }
        return env_str_or("ASCEND_DIRECT_DECODE", "lm_head_ref") != "all_layers_ref";
    }

    bool weight_load_log_enabled() const {
        const std::string explicit_flag = env_str_or("ASCEND_WEIGHT_LOAD_LOG", "");
        if (!explicit_flag.empty()) {
            return explicit_flag != "0" && explicit_flag != "false" && explicit_flag != "False";
        }
        return env_str_or("ASCEND_LOAD_WEIGHTS", "none") != "all";
    }

    bool host_raw_cache_enabled() const {
        const std::string explicit_flag = env_str_or("ASCEND_HOST_RAW_CACHE", "");
        if (!explicit_flag.empty()) {
            return explicit_flag != "0" && explicit_flag != "false" && explicit_flag != "False";
        }
        return env_str_or("ASCEND_DIRECT_DECODE", "lm_head_ref") == "all_layers_ref";
    }

    bool host_raw_drop_after_convert_enabled() const {
        const std::string explicit_flag = env_str_or("ASCEND_HOST_RAW_DROP_AFTER_CONVERT", "");
        if (!explicit_flag.empty()) {
            return explicit_flag != "0" && explicit_flag != "false" && explicit_flag != "False";
        }
        return true;
    }

    bool ref_layer_profile_enabled() const {
        const std::string flag = env_str_or("ASCEND_REF_PROFILE_LAYERS", "0");
        return flag != "0" && flag != "false" && flag != "False";
    }

    int ref_layer_profile_token_limit() const {
        return env_int_or("ASCEND_REF_PROFILE_TOKEN_LIMIT", 0);
    }

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
        seen_tokens.assign(static_cast<size_t>(config.vocab_size), 0);
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
        if (ref_thread_pool) {
            delete ref_thread_pool;
            ref_thread_pool = nullptr;
            ref_thread_pool_size = 0;
        }
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
        if (d_k) {
            aclrtFree(d_k);
            d_k = nullptr;
        }
        if (d_v) {
            aclrtFree(d_v);
            d_v = nullptr;
        }
        free_mlp_buffers();
        free_vec_mm_buffers();
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
        if (weight_load_log_enabled()) {
            time_log("[Ascend][time] weight loaded to HBM, name=" + name +
                     ", dtype=" + meta.dtype +
                     ", shape=" + shape_string(meta.shape) +
                     ", bytes=" + std::to_string(dt.bytes) +
                     ", h2d_ms=" + std::to_string(elapsed_ms(t0, t1)));
        }

        if (host_raw_cache_enabled()) {
            h_weight_raw_cache.emplace(name, std::move(raw));
        }
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
        layer0_k_cache.clear();
        layer0_v_cache.clear();
        layer0_kv_cached_len = 0;
        layer0_kv_dim = 0;
        full_k_cache.clear();
        full_v_cache.clear();
        full_ref_cached_len = 0;
        full_ref_kv_dim = 0;
        full_last_hidden.clear();
        std::fill(seen_tokens.begin(), seen_tokens.end(), 0);
        for (int i = 0; i < len; ++i) {
            if (ids[i] >= 0 && static_cast<size_t>(ids[i]) < seen_tokens.size()) {
                seen_tokens[static_cast<size_t>(ids[i])] = 1;
            }
        }
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
        if (d_hidden && env_str_or("ASCEND_RUN_KVPROJ", "0") != "0") {
            kv_proj_reference(len);
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

    static std::vector<float> raw_tensor_to_float_vector(
        const TensorMeta& meta,
        const unsigned char* raw,
        size_t raw_bytes,
        const std::string& label) {
        const size_t n = tensor_numel(meta.shape);
        const size_t dtype_bytes = dtype_size_bytes(meta);
        if (raw_bytes != n * dtype_bytes) {
            throw std::runtime_error("raw tensor bytes mismatch for " + label);
        }
        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++i) {
            out[i] = load_scalar(raw + i * dtype_bytes, meta.dtype);
        }
        return out;
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

    static bool aclnn_mlp_enabled() {
        const std::string backend = env_str_or("ASCEND_MLP_BACKEND", "");
        if (!backend.empty()) {
            return backend == "aclnn" || backend == "acl" || backend == "ascend" || backend == "1";
        }
        return env_flag_enabled("ASCEND_ACLNN_MLP", false);
    }

    static bool aclnn_mlp_fallback_enabled() {
        return env_flag_enabled("ASCEND_MLP_FALLBACK", true);
    }

    static bool aclnn_mlp_log_enabled() {
        return env_flag_enabled("ASCEND_MLP_LOG", false);
    }

    static bool aclnn_attn_proj_enabled() {
        const std::string backend = env_str_or("ASCEND_ATTN_PROJ_BACKEND", "");
        if (!backend.empty()) {
            return backend == "aclnn" || backend == "acl" || backend == "ascend" || backend == "1";
        }
        return env_flag_enabled("ASCEND_ACLNN_ATTN_PROJ", false);
    }

    static bool aclnn_attn_proj_fallback_enabled() {
        return env_flag_enabled("ASCEND_ATTN_PROJ_FALLBACK", true);
    }

    static bool aclnn_attn_proj_log_enabled() {
        return env_flag_enabled("ASCEND_ATTN_PROJ_LOG", false);
    }

    static bool aclnn_lm_head_enabled() {
        const std::string backend = env_str_or("ASCEND_LM_HEAD_BACKEND", "");
        if (!backend.empty()) {
            return backend == "aclnn" || backend == "acl" || backend == "ascend" || backend == "1";
        }
        return env_flag_enabled("ASCEND_ACLNN_LM_HEAD", false);
    }

    static bool aclnn_lm_head_fallback_enabled() {
        return env_flag_enabled("ASCEND_LM_HEAD_FALLBACK", true);
    }

    static bool aclnn_lm_head_log_enabled() {
        return env_flag_enabled("ASCEND_LM_HEAD_LOG", false);
    }

    static bool is_attention_projection_label(const std::string& label) {
        return label_contains(label, " q_proj") ||
               label_contains(label, " k_proj") ||
               label_contains(label, " v_proj") ||
               label_contains(label, " o_proj");
    }

    static int8_t aclnn_cube_math_type() {
        return static_cast<int8_t>(env_int_or("ASCEND_ACLNN_CUBE_MATH_TYPE", 0));
    }

    static size_t dtype_size_from_string(const std::string& dtype) {
        if (dtype == "BF16" || dtype == "F16") return sizeof(uint16_t);
        if (dtype == "F32" || dtype == "FLOAT32") return sizeof(float);
        throw std::runtime_error("unsupported ACLNN MLP dtype: " + dtype);
    }

    static aclDataType acl_dtype_from_string(const std::string& dtype) {
        if (dtype == "BF16") return ACL_BF16;
        if (dtype == "F16") return ACL_FLOAT16;
        if (dtype == "F32" || dtype == "FLOAT32") return ACL_FLOAT;
        throw std::runtime_error("unsupported ACLNN dtype: " + dtype);
    }

    void free_mlp_buffers() {
        auto free_one = [](void*& p) {
            if (p) {
                aclrtFree(p);
                p = nullptr;
            }
        };
        free_one(d_mlp_x);
        free_one(d_mlp_gate);
        free_one(d_mlp_up);
        free_one(d_mlp_mid);
        free_one(d_mlp_out);
        mlp_hidden_bytes = 0;
        mlp_intermediate_bytes = 0;
        mlp_buffer_dtype.clear();
    }

    void free_vec_mm_buffers() {
        auto free_one = [](void*& p) {
            if (p) {
                aclrtFree(p);
                p = nullptr;
            }
        };
        free_one(d_vec_mm_x);
        free_one(d_vec_mm_out);
        vec_mm_x_bytes = 0;
        vec_mm_out_bytes = 0;
        vec_mm_buffer_dtype.clear();
    }

    void ensure_vec_mm_buffers(const std::string& dtype, size_t in_dim, size_t out_dim) {
        const size_t dtype_bytes = dtype_size_from_string(dtype);
        const size_t need_x = in_dim * dtype_bytes;
        const size_t need_out = out_dim * dtype_bytes;
        if (d_vec_mm_x && d_vec_mm_out &&
            vec_mm_buffer_dtype == dtype &&
            vec_mm_x_bytes >= need_x &&
            vec_mm_out_bytes >= need_out) {
            return;
        }

        free_vec_mm_buffers();
        auto alloc = [](void*& p, size_t bytes, const char* label) {
            check_acl(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST), label);
            check_acl(aclrtMemset(p, bytes, 0, bytes), label);
        };
        alloc(d_vec_mm_x, need_x, "aclrtMalloc(vector MM input)");
        alloc(d_vec_mm_out, need_out, "aclrtMalloc(vector MM output)");
        vec_mm_x_bytes = need_x;
        vec_mm_out_bytes = need_out;
        vec_mm_buffer_dtype = dtype;
        time_log("[Ascend][time] ACLNN vector MM buffers allocated, dtype=" + dtype +
                 ", input_bytes=" + std::to_string(vec_mm_x_bytes) +
                 ", output_bytes=" + std::to_string(vec_mm_out_bytes));
    }

    void ensure_mlp_buffers(const std::string& dtype) {
        const size_t dtype_bytes = dtype_size_from_string(dtype);
        const size_t hidden = static_cast<size_t>(config.hidden);
        const size_t intermediate = static_cast<size_t>(config.intermediate);
        const size_t need_hidden = hidden * dtype_bytes;
        const size_t need_intermediate = intermediate * dtype_bytes;
        if (d_mlp_x && d_mlp_gate && d_mlp_up && d_mlp_mid && d_mlp_out &&
            mlp_buffer_dtype == dtype &&
            mlp_hidden_bytes == need_hidden &&
            mlp_intermediate_bytes == need_intermediate) {
            return;
        }

        free_mlp_buffers();
        auto alloc = [](void*& p, size_t bytes, const char* label) {
            check_acl(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST), label);
            check_acl(aclrtMemset(p, bytes, 0, bytes), label);
        };
        alloc(d_mlp_x, need_hidden, "aclrtMalloc(MLP input)");
        alloc(d_mlp_gate, need_intermediate, "aclrtMalloc(MLP gate)");
        alloc(d_mlp_up, need_intermediate, "aclrtMalloc(MLP up)");
        alloc(d_mlp_mid, need_intermediate, "aclrtMalloc(MLP mid)");
        alloc(d_mlp_out, need_hidden, "aclrtMalloc(MLP output)");
        mlp_hidden_bytes = need_hidden;
        mlp_intermediate_bytes = need_intermediate;
        mlp_buffer_dtype = dtype;
        time_log("[Ascend][time] ACLNN MLP buffers allocated, dtype=" + dtype +
                 ", hidden_bytes=" + std::to_string(mlp_hidden_bytes) +
                 ", intermediate_bytes=" + std::to_string(mlp_intermediate_bytes));
    }

    static AclTensorGuard create_acl_tensor_2d(
        void* data,
        int64_t rows,
        int64_t cols,
        aclDataType dtype,
        const std::string& label) {
        return create_acl_tensor_2d_strided(
            data, rows, cols, cols, 1, rows, cols, dtype, label);
    }

    static AclTensorGuard create_acl_tensor_2d_strided(
        void* data,
        int64_t rows,
        int64_t cols,
        int64_t stride0,
        int64_t stride1,
        int64_t storage_rows,
        int64_t storage_cols,
        aclDataType dtype,
        const std::string& label) {
        int64_t dims[2] = {rows, cols};
        int64_t strides[2] = {stride0, stride1};
        int64_t storage_dims[2] = {storage_rows, storage_cols};
        AclRuntimeTensorApi& api = global_acl_tensor_api();
        if (!api.create_tensor) {
            throw std::runtime_error("aclCreateTensor symbol is not loaded for " + label);
        }
        aclTensor* tensor = api.create_tensor(
            dims,
            2,
            dtype,
            strides,
            0,
            ACL_FORMAT_ND,
            storage_dims,
            2,
            data);
        if (!tensor) throw std::runtime_error("aclCreateTensor failed for " + label);
        return AclTensorGuard(tensor);
    }

    static void fill_raw_from_float_vector(
        const std::vector<float>& x,
        const std::string& dtype,
        std::vector<unsigned char>& raw) {
        const size_t dtype_bytes = dtype_size_from_string(dtype);
        raw.resize(x.size() * dtype_bytes);
        for (size_t i = 0; i < x.size(); ++i) {
            store_scalar(raw.data() + i * dtype_bytes, dtype, x[i]);
        }
    }

    static std::vector<float> raw_to_float_vector(
        const std::vector<unsigned char>& raw,
        const std::string& dtype,
        size_t elements) {
        const size_t dtype_bytes = dtype_size_from_string(dtype);
        if (raw.size() != elements * dtype_bytes) {
            throw std::runtime_error("raw_to_float_vector byte size mismatch");
        }
        std::vector<float> out(elements);
        for (size_t i = 0; i < elements; ++i) {
            out[i] = load_scalar(raw.data() + i * dtype_bytes, dtype);
        }
        return out;
    }

    static void check_aclnn_status(int ret, const std::string& label) {
        if (ret != 0) {
            throw std::runtime_error(label + " failed, ret=" + std::to_string(ret));
        }
    }

    void launch_aclnn_mm(
        AclnnApi& api,
        aclTensor* input,
        aclTensor* weight,
        aclTensor* output,
        DeviceWorkspaceGuard& workspaces,
        const std::string& label) {
        uint64_t workspace_size = 0;
        aclOpExecutor* executor = nullptr;
        check_aclnn_status(
            api.mm_ws(input, weight, output, aclnn_cube_math_type(), &workspace_size, &executor),
            label + " GetWorkspaceSize");
        void* workspace = workspaces.allocate(workspace_size, ("aclrtMalloc(" + label + " workspace)").c_str());
        check_aclnn_status(api.mm(workspace, workspace_size, executor, stream), label);
    }

    void launch_aclnn_silu(
        AclnnApi& api,
        aclTensor* input,
        aclTensor* output,
        DeviceWorkspaceGuard& workspaces,
        const std::string& label) {
        uint64_t workspace_size = 0;
        aclOpExecutor* executor = nullptr;
        check_aclnn_status(
            api.silu_ws(input, output, &workspace_size, &executor),
            label + " GetWorkspaceSize");
        void* workspace = workspaces.allocate(workspace_size, ("aclrtMalloc(" + label + " workspace)").c_str());
        check_aclnn_status(api.silu(workspace, workspace_size, executor, stream), label);
    }

    void launch_aclnn_mul(
        AclnnApi& api,
        aclTensor* lhs,
        aclTensor* rhs,
        aclTensor* output,
        DeviceWorkspaceGuard& workspaces,
        const std::string& label) {
        uint64_t workspace_size = 0;
        aclOpExecutor* executor = nullptr;
        check_aclnn_status(
            api.mul_ws(lhs, rhs, output, &workspace_size, &executor),
            label + " GetWorkspaceSize");
        void* workspace = workspaces.allocate(workspace_size, ("aclrtMalloc(" + label + " workspace)").c_str());
        check_aclnn_status(api.mul(workspace, workspace_size, executor, stream), label);
    }

    bool vector_mm_aclnn_forward(
        const std::vector<float>& x,
        const DeviceTensor& weight,
        size_t out_dim,
        size_t in_dim,
        std::vector<float>& out,
        DeviceLinearTiming& timing,
        const std::string& label,
        std::string& reason) {
        try {
            AclnnApi& api = global_aclnn_api();
            if (!api.load(reason)) return false;
            AclRuntimeTensorApi& tensor_api = global_acl_tensor_api();
            if (!tensor_api.load(api.handle, reason)) return false;

            if (x.size() != in_dim) {
                reason = label + " input dim mismatch";
                return false;
            }
            if (weight.meta.shape.size() != 2 ||
                weight.meta.shape[0] != out_dim ||
                weight.meta.shape[1] != in_dim) {
                reason = label + " weight shape mismatch: " + shape_string(weight.meta.shape);
                return false;
            }
            if (weight.meta.dtype != "BF16" && weight.meta.dtype != "F16") {
                reason = label + " currently supports BF16/F16 weights only, got " + weight.meta.dtype;
                return false;
            }

            const std::string dtype = weight.meta.dtype;
            const aclDataType acl_dtype = acl_dtype_from_string(dtype);
            ensure_vec_mm_buffers(dtype, in_dim, out_dim);

            auto t0 = Clock::now();
            std::vector<unsigned char> raw_in;
            fill_raw_from_float_vector(x, dtype, raw_in);
            check_acl(aclrtMemcpy(d_vec_mm_x, vec_mm_x_bytes, raw_in.data(), raw_in.size(), ACL_MEMCPY_HOST_TO_DEVICE),
                      ("aclrtMemcpy(H2D " + label + " input)").c_str());

            AclTensorGuard input = create_acl_tensor_2d(
                d_vec_mm_x,
                1,
                static_cast<int64_t>(in_dim),
                acl_dtype,
                label + " input");
            AclTensorGuard weight_t = create_acl_tensor_2d_strided(
                weight.data,
                static_cast<int64_t>(in_dim),
                static_cast<int64_t>(out_dim),
                1,
                static_cast<int64_t>(in_dim),
                static_cast<int64_t>(out_dim),
                static_cast<int64_t>(in_dim),
                acl_dtype,
                label + " weight transposed view");
            AclTensorGuard output = create_acl_tensor_2d(
                d_vec_mm_out,
                1,
                static_cast<int64_t>(out_dim),
                acl_dtype,
                label + " output");

            DeviceWorkspaceGuard workspaces;
            launch_aclnn_mm(api, input.get(), weight_t.get(), output.get(), workspaces, label + " mm");
            check_acl(aclrtSynchronizeStream(stream), ("aclrtSynchronizeStream(" + label + ")").c_str());

            const size_t out_bytes = out_dim * dtype_size_from_string(dtype);
            std::vector<unsigned char> raw_out(out_bytes);
            check_acl(aclrtMemcpy(raw_out.data(), raw_out.size(), d_vec_mm_out, out_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                      ("aclrtMemcpy(D2H " + label + " output)").c_str());
            out = raw_to_float_vector(raw_out, dtype, out_dim);
            auto t1 = Clock::now();
            timing.total_ms = elapsed_ms(t0, t1);
            return true;
        } catch (const std::exception& e) {
            reason = e.what();
            return false;
        }
    }

    bool mlp_aclnn_forward(
        const std::vector<float>& mlp_in,
        const std::string& gate_name,
        const std::string& up_name,
        const std::string& down_name,
        std::vector<float>& mlp_out,
        DeviceMlpTiming& timing,
        std::string& reason) {
        try {
            AclnnApi& api = global_aclnn_api();
            if (!api.load(reason)) return false;
            AclRuntimeTensorApi& tensor_api = global_acl_tensor_api();
            if (!tensor_api.load(api.handle, reason)) return false;

            const DeviceTensor& gate = require_device_weight(gate_name);
            const DeviceTensor& up = require_device_weight(up_name);
            const DeviceTensor& down = require_device_weight(down_name);
            const size_t hidden = static_cast<size_t>(config.hidden);
            const size_t intermediate = static_cast<size_t>(config.intermediate);
            if (mlp_in.size() != hidden) {
                reason = "ACLNN MLP input dim mismatch";
                return false;
            }
            if (gate.meta.dtype != up.meta.dtype || gate.meta.dtype != down.meta.dtype) {
                reason = "ACLNN MLP requires gate/up/down to share dtype";
                return false;
            }
            if (gate.meta.dtype != "BF16" && gate.meta.dtype != "F16") {
                reason = "ACLNN MLP currently supports BF16/F16 weights only, got " + gate.meta.dtype;
                return false;
            }
            if (gate.meta.shape.size() != 2 || up.meta.shape.size() != 2 || down.meta.shape.size() != 2 ||
                gate.meta.shape[0] != intermediate || gate.meta.shape[1] != hidden ||
                up.meta.shape[0] != intermediate || up.meta.shape[1] != hidden ||
                down.meta.shape[0] != hidden || down.meta.shape[1] != intermediate) {
                reason = "ACLNN MLP weight shape mismatch";
                return false;
            }

            const std::string dtype = gate.meta.dtype;
            const aclDataType acl_dtype = acl_dtype_from_string(dtype);
            ensure_mlp_buffers(dtype);

            auto gate0 = Clock::now();
            std::vector<unsigned char> raw_in;
            fill_raw_from_float_vector(mlp_in, dtype, raw_in);
            check_acl(aclrtMemcpy(d_mlp_x, mlp_hidden_bytes, raw_in.data(), raw_in.size(), ACL_MEMCPY_HOST_TO_DEVICE),
                      "aclrtMemcpy(H2D ACLNN MLP input)");

            AclTensorGuard x = create_acl_tensor_2d(d_mlp_x, 1, static_cast<int64_t>(hidden), acl_dtype, "MLP input");
            AclTensorGuard gate_w_t = create_acl_tensor_2d_strided(
                gate.data,
                static_cast<int64_t>(hidden),
                static_cast<int64_t>(intermediate),
                1,
                static_cast<int64_t>(hidden),
                static_cast<int64_t>(intermediate),
                static_cast<int64_t>(hidden),
                acl_dtype,
                "MLP gate weight transposed view");
            AclTensorGuard up_w_t = create_acl_tensor_2d_strided(
                up.data,
                static_cast<int64_t>(hidden),
                static_cast<int64_t>(intermediate),
                1,
                static_cast<int64_t>(hidden),
                static_cast<int64_t>(intermediate),
                static_cast<int64_t>(hidden),
                acl_dtype,
                "MLP up weight transposed view");
            AclTensorGuard down_w_t = create_acl_tensor_2d_strided(
                down.data,
                static_cast<int64_t>(intermediate),
                static_cast<int64_t>(hidden),
                1,
                static_cast<int64_t>(intermediate),
                static_cast<int64_t>(hidden),
                static_cast<int64_t>(intermediate),
                acl_dtype,
                "MLP down weight transposed view");
            AclTensorGuard gate_out = create_acl_tensor_2d(d_mlp_gate, 1, static_cast<int64_t>(intermediate), acl_dtype, "MLP gate output");
            AclTensorGuard up_out = create_acl_tensor_2d(d_mlp_up, 1, static_cast<int64_t>(intermediate), acl_dtype, "MLP up output");
            AclTensorGuard mid = create_acl_tensor_2d(d_mlp_mid, 1, static_cast<int64_t>(intermediate), acl_dtype, "MLP mid");
            AclTensorGuard out = create_acl_tensor_2d(d_mlp_out, 1, static_cast<int64_t>(hidden), acl_dtype, "MLP output");

            DeviceWorkspaceGuard workspaces;
            launch_aclnn_mm(api, x.get(), gate_w_t.get(), gate_out.get(), workspaces, "ACLNN MLP gate mm");
            launch_aclnn_mm(api, x.get(), up_w_t.get(), up_out.get(), workspaces, "ACLNN MLP up mm");
            launch_aclnn_silu(api, gate_out.get(), mid.get(), workspaces, "ACLNN MLP silu");
            launch_aclnn_mul(api, mid.get(), up_out.get(), gate_out.get(), workspaces, "ACLNN MLP silu_mul");
            auto gate1 = Clock::now();
            auto down0 = Clock::now();
            launch_aclnn_mm(api, gate_out.get(), down_w_t.get(), out.get(), workspaces, "ACLNN MLP down mm");
            check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(ACLNN MLP)");

            std::vector<unsigned char> raw_out(mlp_hidden_bytes);
            check_acl(aclrtMemcpy(raw_out.data(), raw_out.size(), d_mlp_out, mlp_hidden_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                      "aclrtMemcpy(D2H ACLNN MLP output)");
            auto down1 = Clock::now();
            mlp_out = raw_to_float_vector(raw_out, dtype, hidden);
            timing.gate_up_ms = elapsed_ms(gate0, gate1);
            timing.down_ms = elapsed_ms(down0, down1);

            if (!aclnn_mlp_notice_printed || aclnn_mlp_log_enabled()) {
                time_log("[Ascend][time] ACLNN MLP path active, dtype=" + dtype +
                         ", hidden=" + std::to_string(hidden) +
                         ", intermediate=" + std::to_string(intermediate) +
                         ", gate_up_enqueue_ms=" + std::to_string(timing.gate_up_ms) +
                         ", down_sync_d2h_ms=" + std::to_string(timing.down_ms));
                aclnn_mlp_notice_printed = true;
            }
            return true;
        } catch (const std::exception& e) {
            reason = e.what();
            return false;
        }
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
        ensure_projection_buffer(d_q, q_bytes, q_row_bytes, q_meta, "q");
    }

    void ensure_projection_buffer(
        void*& buffer,
        size_t& total_bytes,
        size_t& row_bytes,
        const TensorMeta& weight_meta,
        const std::string& label) {
        if (buffer) return;
        if (weight_meta.shape.size() != 2) {
            throw std::runtime_error(label + "_proj weight must be 2D, got shape=" + shape_string(weight_meta.shape));
        }
        if (weight_meta.shape[1] != static_cast<size_t>(config.hidden)) {
            throw std::runtime_error(label + "_proj in_features mismatch, shape=" + shape_string(weight_meta.shape));
        }
        const size_t dtype_bytes = dtype_size_bytes(weight_meta);
        row_bytes = weight_meta.shape[0] * dtype_bytes;
        total_bytes = static_cast<size_t>(max_seq) * row_bytes;
        check_acl(aclrtMalloc(&buffer, total_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  ("aclrtMalloc(" + label + " buffer)").c_str());
        check_acl(aclrtMemset(buffer, total_bytes, 0, total_bytes),
                  ("aclrtMemset(" + label + " buffer)").c_str());
        time_log("[Ascend][time] " + label + " buffer allocated, row_bytes=" +
                 std::to_string(row_bytes) +
                 ", total_bytes=" + std::to_string(total_bytes));
    }

    void linear_projection_reference(
        const std::string& weight_name,
        void*& out_buffer,
        size_t& out_total_bytes,
        size_t& out_row_bytes,
        const std::string& label,
        int len) {
        auto it = d_weights.find(weight_name);
        if (it == d_weights.end()) {
            throw std::runtime_error(
                "projection " + weight_name + " requires ASCEND_LOAD_WEIGHTS=layer0 or all");
        }
        if (!d_hidden || hidden_row_bytes == 0) {
            throw std::runtime_error(label + "_proj requires hidden buffer");
        }

        const DeviceTensor& weight = it->second;
        const TensorMeta& weight_meta = weight.meta;
        ensure_projection_buffer(out_buffer, out_total_bytes, out_row_bytes, weight_meta, label);

        const size_t out_dim = weight_meta.shape[0];
        const size_t in_dim = weight_meta.shape[1];
        const TensorMeta& hidden_meta = d_weights.at("model.embed_tokens.weight").meta;
        const size_t hidden_dtype_bytes = dtype_size_bytes(hidden_meta);
        const size_t weight_dtype_bytes = dtype_size_bytes(weight_meta);
        if (in_dim != static_cast<size_t>(config.hidden)) {
            throw std::runtime_error(label + "_proj in_dim mismatch");
        }

        auto t0 = Clock::now();
        const size_t active_hidden_bytes = static_cast<size_t>(len) * hidden_row_bytes;
        std::vector<unsigned char> h_hidden(active_hidden_bytes);
        std::vector<unsigned char> h_weight(weight.bytes);
        std::vector<unsigned char> h_out(static_cast<size_t>(len) * out_row_bytes);

        check_acl(aclrtMemcpy(h_hidden.data(), active_hidden_bytes, d_hidden, active_hidden_bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  ("aclrtMemcpy(D2H hidden for " + label + "_proj)").c_str());
        check_acl(aclrtMemcpy(h_weight.data(), weight.bytes, weight.data, weight.bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  ("aclrtMemcpy(D2H " + label + "_proj weight)").c_str());

        const int max_tokens = env_int_or("ASCEND_PROJ_REF_TOKENS", env_int_or("ASCEND_QPROJ_REF_TOKENS", 1));
        const int compute_tokens = std::max(0, std::min(len, max_tokens));
        float first_value = 0.0f;
        for (int tok = 0; tok < compute_tokens; ++tok) {
            const unsigned char* xrow = h_hidden.data() + static_cast<size_t>(tok) * hidden_row_bytes;
            unsigned char* outrow = h_out.data() + static_cast<size_t>(tok) * out_row_bytes;
            for (size_t out = 0; out < out_dim; ++out) {
                const unsigned char* wrow = h_weight.data() + out * in_dim * weight_dtype_bytes;
                double acc = 0.0;
                for (size_t in = 0; in < in_dim; ++in) {
                    const float x = load_scalar(xrow + in * hidden_dtype_bytes, hidden_meta.dtype);
                    const float w = load_scalar(wrow + in * weight_dtype_bytes, weight_meta.dtype);
                    acc += static_cast<double>(x) * static_cast<double>(w);
                }
                const float y = static_cast<float>(acc);
                if (tok == 0 && out == 0) first_value = y;
                store_scalar(outrow + out * weight_dtype_bytes, weight_meta.dtype, y);
            }
        }

        if (compute_tokens > 0) {
            const size_t active_out_bytes = static_cast<size_t>(compute_tokens) * out_row_bytes;
            check_acl(aclrtMemcpy(out_buffer, out_total_bytes, h_out.data(), active_out_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                      ("aclrtMemcpy(H2D " + label + "_proj output)").c_str());
        }

        auto t1 = Clock::now();
        time_log("[Ascend][time] " + label + "_proj reference finished, tokens_requested=" +
                 std::to_string(len) +
                 ", tokens_computed=" + std::to_string(compute_tokens) +
                 ", in_dim=" + std::to_string(in_dim) +
                 ", out_dim=" + std::to_string(out_dim) +
                 ", weight_dtype=" + weight_meta.dtype +
                 ", first_value=" + std::to_string(first_value) +
                 ", elapsed_ms=" + std::to_string(elapsed_ms(t0, t1)));
    }

    void q_proj_reference(int len) {
        linear_projection_reference(
            "model.layers.0.self_attn.q_proj.weight",
            d_q,
            q_bytes,
            q_row_bytes,
            "q",
            len);
    }

    void kv_proj_reference(int len) {
        linear_projection_reference(
            "model.layers.0.self_attn.k_proj.weight",
            d_k,
            k_bytes,
            k_row_bytes,
            "k",
            len);
        linear_projection_reference(
            "model.layers.0.self_attn.v_proj.weight",
            d_v,
            v_bytes,
            v_row_bytes,
            "v",
            len);
    }

    void set_repetition_penalty(float penalty) {
        repetition_penalty = penalty > 0.0f ? penalty : 1.0f;
    }

    const DeviceTensor& require_device_weight(const std::string& name) const {
        auto it = d_weights.find(name);
        if (it == d_weights.end()) {
            throw std::runtime_error("missing loaded weight " + name + "; use ASCEND_LOAD_WEIGHTS=layer0 or all");
        }
        return it->second;
    }

    RefThreadPool& ref_pool_for(int n_threads) const {
        n_threads = std::max(1, n_threads);
        if (!ref_thread_pool || ref_thread_pool_size < n_threads) {
            if (ref_thread_pool) delete ref_thread_pool;
            ref_thread_pool = new RefThreadPool(n_threads);
            ref_thread_pool_size = n_threads;
        }
        return *ref_thread_pool;
    }

    static bool label_contains(const std::string& label, const std::string& needle) {
        return label.find(needle) != std::string::npos;
    }

    int reference_threads_for_label(
        const std::string& label,
        size_t out_dim,
        bool gate_up) const {
        const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
        int fallback = env_int_or("ASCEND_REF_LINEAR_THREADS", 0);
        if (fallback <= 0) fallback = static_cast<int>(hw_threads);

        int requested = fallback;
        if (gate_up || label_contains(label, "down_proj")) {
            requested = env_int_or("ASCEND_REF_MLP_THREADS", fallback);
            if (label_contains(label, "down_proj")) {
                requested = env_int_or("ASCEND_REF_DOWN_THREADS", requested);
            }
        } else if (label_contains(label, "q_proj") ||
                   label_contains(label, "k_proj") ||
                   label_contains(label, "v_proj") ||
                   label_contains(label, "o_proj")) {
            requested = env_int_or("ASCEND_REF_ATTN_LINEAR_THREADS", fallback);
        }

        int n_threads = std::max(
            1,
            std::min<int>(requested > 0 ? requested : static_cast<int>(hw_threads),
                          static_cast<int>(std::max<size_t>(1, out_dim))));
        if (out_dim < 1024 && requested <= 0) n_threads = 1;
        return n_threads;
    }

    std::pair<size_t, size_t> matrix_shape(const std::string& name) const {
        const DeviceTensor& w = require_device_weight(name);
        if (w.meta.shape.size() != 2) {
            throw std::runtime_error("weight must be 2D: " + name + " shape=" + shape_string(w.meta.shape));
        }
        return {w.meta.shape[0], w.meta.shape[1]};
    }

    std::vector<float> device_tensor_to_float_vector(const DeviceTensor& tensor, const std::string& label) {
        const size_t n = tensor_numel(tensor.meta.shape);
        const size_t dtype_bytes = dtype_size_bytes(tensor.meta);
        std::vector<unsigned char> raw(tensor.bytes);
        check_acl(aclrtMemcpy(raw.data(), tensor.bytes, tensor.data, tensor.bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  ("aclrtMemcpy(D2H " + label + ")").c_str());

        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++i) {
            out[i] = load_scalar(raw.data() + i * dtype_bytes, tensor.meta.dtype);
        }
        return out;
    }

    std::vector<float> weight_to_float_vector(const std::string& name) {
        auto raw_it = h_weight_raw_cache.find(name);
        if (raw_it != h_weight_raw_cache.end()) {
            std::vector<float> value = raw_tensor_to_float_vector(
                require_device_weight(name).meta,
                raw_it->second.data(),
                raw_it->second.size(),
                name);
            if (host_raw_drop_after_convert_enabled()) {
                h_weight_raw_cache.erase(raw_it);
            }
            return value;
        }
        return device_tensor_to_float_vector(require_device_weight(name), name);
    }

    std::vector<float> load_weight_float(const std::string& name) {
        const bool cache_enabled = env_str_or("ASCEND_REF_CACHE_WEIGHTS", "1") != "0";
        if (cache_enabled) {
            auto cached = h_weight_cache.find(name);
            if (cached != h_weight_cache.end()) return cached->second;
        }

        auto t0 = Clock::now();
        std::vector<float> value = weight_to_float_vector(name);
        auto t1 = Clock::now();
        if (cache_enabled) {
            const size_t bytes = value.size() * sizeof(float);
            h_weight_cache.emplace(name, value);
            if (ref_cache_log_enabled()) {
                time_log("[Ascend][time] cached host reference weight, name=" + name +
                         ", elements=" + std::to_string(value.size()) +
                         ", fp32_bytes=" + std::to_string(bytes) +
                         ", load_convert_ms=" + std::to_string(elapsed_ms(t0, t1)));
            }
            return h_weight_cache.at(name);
        }
        return value;
    }

    const std::vector<float>& cached_weight_float_ref(const std::string& name) {
        auto cached = h_weight_cache.find(name);
        if (cached != h_weight_cache.end()) return cached->second;

        auto t0 = Clock::now();
        std::vector<float> value = weight_to_float_vector(name);
        auto t1 = Clock::now();
        const size_t bytes = value.size() * sizeof(float);
        auto inserted = h_weight_cache.emplace(name, std::move(value));
        if (ref_cache_log_enabled()) {
            time_log("[Ascend][time] cached host reference weight, name=" + name +
                     ", elements=" + std::to_string(inserted.first->second.size()) +
                     ", fp32_bytes=" + std::to_string(bytes) +
                     ", load_convert_ms=" + std::to_string(elapsed_ms(t0, t1)));
        }
        return inserted.first->second;
    }

    const std::vector<uint16_t>& cached_weight_u16_ref(const std::string& name) {
        auto cached = h_weight_u16_cache.find(name);
        if (cached != h_weight_u16_cache.end()) return cached->second;

        const DeviceTensor& tensor = require_device_weight(name);
        if (tensor.meta.dtype != "BF16" && tensor.meta.dtype != "F16") {
            throw std::runtime_error("raw u16 weight cache requires BF16/F16 tensor: " + name +
                                     " dtype=" + tensor.meta.dtype);
        }
        const size_t n = tensor_numel(tensor.meta.shape);
        if (tensor.bytes != n * sizeof(uint16_t)) {
            throw std::runtime_error("raw u16 weight bytes mismatch for " + name);
        }

        auto t0 = Clock::now();
        std::vector<uint16_t> value(n);
        auto raw_it = h_weight_raw_cache.find(name);
        if (raw_it != h_weight_raw_cache.end()) {
            std::memcpy(value.data(), raw_it->second.data(), tensor.bytes);
            if (host_raw_drop_after_convert_enabled()) {
                h_weight_raw_cache.erase(raw_it);
            }
        } else {
            check_acl(aclrtMemcpy(value.data(), tensor.bytes, tensor.data, tensor.bytes,
                                  ACL_MEMCPY_DEVICE_TO_HOST),
                      ("aclrtMemcpy(D2H raw u16 " + name + ")").c_str());
        }
        auto t1 = Clock::now();
        auto inserted = h_weight_u16_cache.emplace(name, std::move(value));
        if (ref_cache_log_enabled()) {
            time_log("[Ascend][time] cached host raw u16 weight, name=" + name +
                     ", elements=" + std::to_string(inserted.first->second.size()) +
                     ", raw_bytes=" + std::to_string(tensor.bytes) +
                     ", load_ms=" + std::to_string(elapsed_ms(t0, t1)));
        }
        return inserted.first->second;
    }

    const std::vector<float>* optional_cached_weight_float_ref(const std::string& name) {
        if (d_weights.find(name) == d_weights.end()) return nullptr;
        return &cached_weight_float_ref(name);
    }

    static std::string layer_weight_name(int layer, const std::string& suffix) {
        return "model.layers." + std::to_string(layer) + "." + suffix;
    }

    std::vector<float> load_hidden_rows_float_range(int start, int count) {
        if (!d_hidden || hidden_row_bytes == 0) {
            throw std::runtime_error("hidden rows require embedding prefill first");
        }
        if (start < 0 || count < 0 || start > max_seq || count > max_seq - start) {
            throw std::runtime_error("hidden row range out of bounds");
        }
        if (count == 0) {
            return {};
        }
        const TensorMeta& hidden_meta = require_device_weight("model.embed_tokens.weight").meta;
        const size_t hidden_dtype_bytes = dtype_size_bytes(hidden_meta);
        const size_t hidden = static_cast<size_t>(config.hidden);
        const size_t active_bytes = static_cast<size_t>(count) * hidden_row_bytes;
        std::vector<unsigned char> raw(active_bytes);
        char* src = static_cast<char*>(d_hidden) + static_cast<size_t>(start) * hidden_row_bytes;
        check_acl(aclrtMemcpy(raw.data(), active_bytes, src, active_bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(D2H hidden rows)");

        std::vector<float> rows(static_cast<size_t>(count) * hidden);
        for (int tok = 0; tok < count; ++tok) {
            const unsigned char* row = raw.data() + static_cast<size_t>(tok) * hidden_row_bytes;
            for (size_t j = 0; j < hidden; ++j) {
                rows[static_cast<size_t>(tok) * hidden + j] =
                    load_scalar(row + j * hidden_dtype_bytes, hidden_meta.dtype);
            }
        }
        return rows;
    }

    std::vector<float> load_hidden_rows_float(int len) {
        return load_hidden_rows_float_range(0, len);
    }

    void store_hidden_row_float(int row_idx, const std::vector<float>& x) {
        if (row_idx < 0 || row_idx >= max_seq) throw std::runtime_error("hidden row index out of range");
        if (x.size() != static_cast<size_t>(config.hidden)) {
            throw std::runtime_error("store hidden row size mismatch");
        }
        const TensorMeta& hidden_meta = require_device_weight("model.embed_tokens.weight").meta;
        const size_t hidden_dtype_bytes = dtype_size_bytes(hidden_meta);
        std::vector<unsigned char> row(hidden_row_bytes);
        for (size_t j = 0; j < x.size(); ++j) {
            store_scalar(row.data() + j * hidden_dtype_bytes, hidden_meta.dtype, x[j]);
        }
        char* dst = static_cast<char*>(d_hidden) + static_cast<size_t>(row_idx) * hidden_row_bytes;
        check_acl(aclrtMemcpy(dst, hidden_row_bytes, row.data(), hidden_row_bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(H2D hidden row)");
    }

    void rms_norm_inplace(std::vector<float>& x, const std::vector<float>& weight) const {
        if (x.size() != weight.size()) throw std::runtime_error("RMSNorm vector size mismatch");
        double sum_sq = 0.0;
        for (float v : x) sum_sq += static_cast<double>(v) * static_cast<double>(v);
        const float scale = 1.0f / std::sqrt(static_cast<float>(sum_sq / x.size()) + config.rms_norm_eps);
        for (size_t i = 0; i < x.size(); ++i) x[i] = x[i] * scale * weight[i];
    }

    std::vector<float> linear_with_weight(
        const std::vector<float>& x,
        const std::vector<float>& weight,
        size_t out_dim,
        size_t in_dim,
        const std::string& label,
        const std::vector<float>* bias = nullptr) const {
        if (x.size() != in_dim) {
            throw std::runtime_error(label + " input dim mismatch");
        }
        if (weight.size() != out_dim * in_dim) {
            throw std::runtime_error(label + " weight size mismatch");
        }
        if (bias && bias->size() != out_dim) {
            throw std::runtime_error(label + " bias size mismatch");
        }
        std::vector<float> y(out_dim);

        const int n_threads = reference_threads_for_label(label, out_dim, false);

        auto compute_range = [&](int tid) {
            const size_t begin = (out_dim * static_cast<size_t>(tid)) / static_cast<size_t>(n_threads);
            const size_t end = (out_dim * static_cast<size_t>(tid + 1)) / static_cast<size_t>(n_threads);
            size_t out = begin;
            if (ref_dot4_enabled() && in_dim >= 16) {
                for (; out + 3 < end; out += 4) {
                    const float* w0 = weight.data() + (out + 0) * in_dim;
                    const float* w1 = weight.data() + (out + 1) * in_dim;
                    const float* w2 = weight.data() + (out + 2) * in_dim;
                    const float* w3 = weight.data() + (out + 3) * in_dim;
#if defined(__GNUC__) || defined(__clang__)
                    if (out + 4 < end) __builtin_prefetch(weight.data() + (out + 4) * in_dim, 0, 1);
#endif
                    float y0 = 0.0f;
                    float y1 = 0.0f;
                    float y2 = 0.0f;
                    float y3 = 0.0f;
                    dot4_reference(x.data(), w0, w1, w2, w3, in_dim, y0, y1, y2, y3);
                    if (bias) {
                        y0 += (*bias)[out + 0];
                        y1 += (*bias)[out + 1];
                        y2 += (*bias)[out + 2];
                        y3 += (*bias)[out + 3];
                    }
                    y[out + 0] = y0;
                    y[out + 1] = y1;
                    y[out + 2] = y2;
                    y[out + 3] = y3;
                }
            }
            for (; out < end; ++out) {
                const float* wrow = weight.data() + out * in_dim;
#if defined(__GNUC__) || defined(__clang__)
                if (out + 1 < end) __builtin_prefetch(weight.data() + (out + 1) * in_dim, 0, 1);
#endif
                float acc = dot_product_reference(x.data(), wrow, in_dim);
                if (bias) acc += (*bias)[out];
                y[out] = acc;
            }
        };

        if (n_threads == 1) {
            compute_range(0);
        } else {
            ref_pool_for(n_threads).run(n_threads, compute_range);
        }
        return y;
    }

    static bool u16_weight_dtype_supported(const std::string& dtype) {
        return dtype == "BF16" || dtype == "F16";
    }

    std::vector<float> linear_with_u16_weight(
        const std::vector<float>& x,
        const std::vector<uint16_t>& weight,
        size_t out_dim,
        size_t in_dim,
        const std::string& weight_dtype,
        const std::string& label,
        const std::vector<float>* bias = nullptr) const {
        if (x.size() != in_dim) {
            throw std::runtime_error(label + " input dim mismatch");
        }
        if (weight.size() != out_dim * in_dim) {
            throw std::runtime_error(label + " raw u16 weight size mismatch");
        }
        if (bias && bias->size() != out_dim) {
            throw std::runtime_error(label + " bias size mismatch");
        }
        if (!u16_weight_dtype_supported(weight_dtype)) {
            throw std::runtime_error(label + " raw u16 weight dtype mismatch: " + weight_dtype);
        }
        std::vector<float> y(out_dim);

        const int n_threads = reference_threads_for_label(label, out_dim, false);
        const bool bf16 = weight_dtype == "BF16";

        auto compute_range = [&](int tid) {
            const size_t begin = (out_dim * static_cast<size_t>(tid)) / static_cast<size_t>(n_threads);
            const size_t end = (out_dim * static_cast<size_t>(tid + 1)) / static_cast<size_t>(n_threads);
            for (size_t out = begin; out < end; ++out) {
                const uint16_t* wrow = weight.data() + out * in_dim;
#if defined(__GNUC__) || defined(__clang__)
                if (out + 1 < end) __builtin_prefetch(weight.data() + (out + 1) * in_dim, 0, 1);
#endif
                float acc = dot_product_u16_weight_reference(x.data(), wrow, in_dim, bf16);
                if (bias) acc += (*bias)[out];
                y[out] = acc;
            }
        };

        if (n_threads == 1) {
            compute_range(0);
        } else {
            ref_pool_for(n_threads).run(n_threads, compute_range);
        }
        return y;
    }

    std::vector<float> linear_with_named_weight(
        const std::vector<float>& x,
        const std::string& weight_name,
        size_t out_dim,
        size_t in_dim,
        const std::string& label,
        const std::vector<float>* bias = nullptr) {
        const DeviceTensor& tensor = require_device_weight(weight_name);
        if (tensor.meta.shape.size() != 2 ||
            tensor.meta.shape[0] != out_dim ||
            tensor.meta.shape[1] != in_dim) {
            throw std::runtime_error(label + " weight shape mismatch: " + weight_name +
                                     " shape=" + shape_string(tensor.meta.shape));
        }
        if (aclnn_attn_proj_enabled() &&
            !aclnn_attn_proj_disabled &&
            is_attention_projection_label(label)) {
            std::vector<float> y;
            DeviceLinearTiming timing;
            std::string reason;
            if (vector_mm_aclnn_forward(x, tensor, out_dim, in_dim, y, timing, "ACLNN " + label, reason)) {
                if (bias) {
                    if (bias->size() != out_dim) throw std::runtime_error(label + " bias size mismatch");
                    for (size_t i = 0; i < out_dim; ++i) y[i] += (*bias)[i];
                }
                if (!aclnn_attn_proj_notice_printed || aclnn_attn_proj_log_enabled()) {
                    time_log("[Ascend][time] ACLNN attention projection active, label=" + label +
                             ", dtype=" + tensor.meta.dtype +
                             ", out_dim=" + std::to_string(out_dim) +
                             ", elapsed_ms=" + std::to_string(timing.total_ms));
                    aclnn_attn_proj_notice_printed = true;
                }
                return y;
            }

            aclnn_attn_proj_disabled = true;
            time_log("[Ascend][warn] ACLNN attention projection disabled, fallback=cpu, label=" +
                     label + ", reason=" + reason);
            if (!aclnn_attn_proj_fallback_enabled()) {
                throw std::runtime_error("ACLNN attention projection failed and ASCEND_ATTN_PROJ_FALLBACK=0: " + reason);
            }
        }
        if (ref_u16_weight_enabled() && u16_weight_dtype_supported(tensor.meta.dtype)) {
            const std::vector<uint16_t>& weight = cached_weight_u16_ref(weight_name);
            return linear_with_u16_weight(x, weight, out_dim, in_dim, tensor.meta.dtype, label, bias);
        }
        const std::vector<float>& weight = cached_weight_float_ref(weight_name);
        return linear_with_weight(x, weight, out_dim, in_dim, label, bias);
    }

    std::vector<float> gate_up_silu_reference(
        const std::vector<float>& x,
        const std::vector<float>& gate_weight,
        const std::vector<float>& up_weight,
        size_t out_dim,
        size_t in_dim,
        const std::string& label) const {
        if (x.size() != in_dim) {
            throw std::runtime_error(label + " input dim mismatch");
        }
        if (gate_weight.size() != out_dim * in_dim || up_weight.size() != out_dim * in_dim) {
            throw std::runtime_error(label + " weight size mismatch");
        }
        std::vector<float> mid(out_dim);

        const int n_threads = reference_threads_for_label(label, out_dim, true);

        auto compute_range = [&](int tid) {
            const size_t begin = (out_dim * static_cast<size_t>(tid)) / static_cast<size_t>(n_threads);
            const size_t end = (out_dim * static_cast<size_t>(tid + 1)) / static_cast<size_t>(n_threads);
            for (size_t out = begin; out < end; ++out) {
                const float* grow = gate_weight.data() + out * in_dim;
                const float* urow = up_weight.data() + out * in_dim;
#if defined(__GNUC__) || defined(__clang__)
                if (out + 1 < end) {
                    __builtin_prefetch(gate_weight.data() + (out + 1) * in_dim, 0, 1);
                    __builtin_prefetch(up_weight.data() + (out + 1) * in_dim, 0, 1);
                }
#endif
                float gacc = 0.0f;
                float uacc = 0.0f;
                dot_pair_reference(x.data(), grow, urow, in_dim, gacc, uacc);
                mid[out] = (gacc / (1.0f + std::exp(-gacc))) * uacc;
            }
        };

        if (n_threads == 1) {
            compute_range(0);
        } else {
            ref_pool_for(n_threads).run(n_threads, compute_range);
        }
        return mid;
    }

    std::vector<float> gate_up_silu_u16_reference(
        const std::vector<float>& x,
        const std::vector<uint16_t>& gate_weight,
        const std::vector<uint16_t>& up_weight,
        size_t out_dim,
        size_t in_dim,
        const std::string& weight_dtype,
        const std::string& label) const {
        if (x.size() != in_dim) {
            throw std::runtime_error(label + " input dim mismatch");
        }
        if (gate_weight.size() != out_dim * in_dim || up_weight.size() != out_dim * in_dim) {
            throw std::runtime_error(label + " raw u16 weight size mismatch");
        }
        if (!u16_weight_dtype_supported(weight_dtype)) {
            throw std::runtime_error(label + " raw u16 weight dtype mismatch: " + weight_dtype);
        }
        std::vector<float> mid(out_dim);

        const int n_threads = reference_threads_for_label(label, out_dim, true);
        const bool bf16 = weight_dtype == "BF16";

        auto compute_range = [&](int tid) {
            const size_t begin = (out_dim * static_cast<size_t>(tid)) / static_cast<size_t>(n_threads);
            const size_t end = (out_dim * static_cast<size_t>(tid + 1)) / static_cast<size_t>(n_threads);
            for (size_t out = begin; out < end; ++out) {
                const uint16_t* grow = gate_weight.data() + out * in_dim;
                const uint16_t* urow = up_weight.data() + out * in_dim;
#if defined(__GNUC__) || defined(__clang__)
                if (out + 1 < end) {
                    __builtin_prefetch(gate_weight.data() + (out + 1) * in_dim, 0, 1);
                    __builtin_prefetch(up_weight.data() + (out + 1) * in_dim, 0, 1);
                }
#endif
                float gacc = 0.0f;
                float uacc = 0.0f;
                dot_pair_u16_weight_reference(x.data(), grow, urow, in_dim, bf16, gacc, uacc);
                mid[out] = (gacc / (1.0f + std::exp(-gacc))) * uacc;
            }
        };

        if (n_threads == 1) {
            compute_range(0);
        } else {
            ref_pool_for(n_threads).run(n_threads, compute_range);
        }
        return mid;
    }

    std::vector<float> gate_up_silu_named(
        const std::vector<float>& x,
        const std::string& gate_name,
        const std::string& up_name,
        size_t out_dim,
        size_t in_dim,
        const std::string& label) {
        const DeviceTensor& gate = require_device_weight(gate_name);
        const DeviceTensor& up = require_device_weight(up_name);
        if (gate.meta.shape.size() != 2 || up.meta.shape.size() != 2 ||
            gate.meta.shape[0] != out_dim || gate.meta.shape[1] != in_dim ||
            up.meta.shape[0] != out_dim || up.meta.shape[1] != in_dim) {
            throw std::runtime_error(label + " weight shape mismatch");
        }
        if (ref_u16_weight_enabled() &&
            gate.meta.dtype == up.meta.dtype &&
            u16_weight_dtype_supported(gate.meta.dtype)) {
            const std::vector<uint16_t>& gate_weight = cached_weight_u16_ref(gate_name);
            const std::vector<uint16_t>& up_weight = cached_weight_u16_ref(up_name);
            return gate_up_silu_u16_reference(
                x, gate_weight, up_weight, out_dim, in_dim, gate.meta.dtype, label);
        }
        const std::vector<float>& gate_weight = cached_weight_float_ref(gate_name);
        const std::vector<float>& up_weight = cached_weight_float_ref(up_name);
        return gate_up_silu_reference(x, gate_weight, up_weight, out_dim, in_dim, label);
    }

    void apply_rope(std::vector<float>& x, int heads, int pos) const {
        const int head_dim = config.hidden / config.n_heads;
        if (head_dim <= 0 || head_dim % 2 != 0) throw std::runtime_error("bad RoPE head_dim");
        if (x.size() != static_cast<size_t>(heads * head_dim)) {
            throw std::runtime_error("RoPE vector size mismatch");
        }
        const int half = head_dim / 2;
        for (int h = 0; h < heads; ++h) {
            const int base = h * head_dim;
            for (int p = 0; p < half; ++p) {
                const float inv = std::pow(config.rope_theta, -static_cast<float>(2 * p) / head_dim);
                const float angle = static_cast<float>(pos) * inv;
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                const int d0 = base + p;
                const int d1 = base + p + half;
                const float v0 = x[d0];
                const float v1 = x[d1];
                x[d0] = v0 * c - v1 * s;
                x[d1] = v0 * s + v1 * c;
            }
        }
    }

    std::vector<float> final_norm_vector(std::vector<float> x) {
        const std::vector<float>& norm = cached_weight_float_ref("model.norm.weight");
        rms_norm_inplace(x, norm);
        return x;
    }

    void ensure_full_ref_cache(int kv_dim) {
        if (kv_dim <= 0) throw std::runtime_error("bad kv_dim for all_layers_ref");
        if (full_ref_kv_dim != kv_dim ||
            full_k_cache.size() != static_cast<size_t>(config.n_layers) ||
            full_v_cache.size() != static_cast<size_t>(config.n_layers)) {
            full_k_cache.assign(static_cast<size_t>(config.n_layers), {});
            full_v_cache.assign(static_cast<size_t>(config.n_layers), {});
            full_ref_cached_len = 0;
            full_ref_kv_dim = kv_dim;
            full_last_hidden.clear();
        }
        const size_t cache_elems = static_cast<size_t>(max_seq) * static_cast<size_t>(kv_dim);
        for (int layer = 0; layer < config.n_layers; ++layer) {
            if (full_k_cache[static_cast<size_t>(layer)].size() < cache_elems) {
                full_k_cache[static_cast<size_t>(layer)].assign(cache_elems, 0.0f);
                full_v_cache[static_cast<size_t>(layer)].assign(cache_elems, 0.0f);
            }
        }
    }

    std::vector<float> layer_forward_reference(
        std::vector<float> x,
        int layer,
        int pos,
        RefLayerProfile* profile = nullptr) {
        auto layer0 = Clock::now();
        const size_t hidden = static_cast<size_t>(config.hidden);
        const int head_dim = config.hidden / config.n_heads;
        const int kv_dim = config.n_kv_heads * head_dim;
        const int group = config.n_heads / config.n_kv_heads;
        if (x.size() != hidden) throw std::runtime_error("layer input size mismatch");
        if (head_dim <= 0 || group <= 0 || config.n_heads % config.n_kv_heads != 0) {
            throw std::runtime_error("bad attention head config for all_layers_ref");
        }
        ensure_full_ref_cache(kv_dim);

        const std::string prefix = "layer" + std::to_string(layer);
        const std::string ln1_name = layer_weight_name(layer, "input_layernorm.weight");
        const std::string wq_name = layer_weight_name(layer, "self_attn.q_proj.weight");
        const std::string wk_name = layer_weight_name(layer, "self_attn.k_proj.weight");
        const std::string wv_name = layer_weight_name(layer, "self_attn.v_proj.weight");
        const std::string wo_name = layer_weight_name(layer, "self_attn.o_proj.weight");
        const std::string bq_name = layer_weight_name(layer, "self_attn.q_proj.bias");
        const std::string bk_name = layer_weight_name(layer, "self_attn.k_proj.bias");
        const std::string bv_name = layer_weight_name(layer, "self_attn.v_proj.bias");
        const std::vector<float>& ln1 = cached_weight_float_ref(ln1_name);
        const std::vector<float>* bq = optional_cached_weight_float_ref(bq_name);
        const std::vector<float>* bk = optional_cached_weight_float_ref(bk_name);
        const std::vector<float>* bv = optional_cached_weight_float_ref(bv_name);

        std::vector<float> residual = x;
        std::vector<float> qkv_in = x;
        auto norm1_0 = Clock::now();
        rms_norm_inplace(qkv_in, ln1);
        auto norm1_1 = Clock::now();
        auto q0 = Clock::now();
        std::vector<float> q = linear_with_named_weight(qkv_in, wq_name, hidden, hidden, prefix + " q_proj", bq);
        auto q1 = Clock::now();
        auto kv0 = Clock::now();
        std::vector<float> k = linear_with_named_weight(qkv_in, wk_name, static_cast<size_t>(kv_dim), hidden, prefix + " k_proj", bk);
        std::vector<float> v = linear_with_named_weight(qkv_in, wv_name, static_cast<size_t>(kv_dim), hidden, prefix + " v_proj", bv);
        auto kv1 = Clock::now();
        auto rope0 = Clock::now();
        apply_rope(q, config.n_heads, pos);
        apply_rope(k, config.n_kv_heads, pos);
        auto rope1 = Clock::now();

        std::vector<float>& k_cache = full_k_cache[static_cast<size_t>(layer)];
        std::vector<float>& v_cache = full_v_cache[static_cast<size_t>(layer)];
        std::copy(k.begin(), k.end(), k_cache.begin() + static_cast<size_t>(pos) * kv_dim);
        std::copy(v.begin(), v.end(), v_cache.begin() + static_cast<size_t>(pos) * kv_dim);

        std::vector<float> ctx(hidden, 0.0f);
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::vector<float> scores(static_cast<size_t>(pos + 1));
        auto attn0 = Clock::now();
        for (int h = 0; h < config.n_heads; ++h) {
            const int kh = h / group;
            const float* qh = q.data() + static_cast<size_t>(h) * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            for (int tok = 0; tok <= pos; ++tok) {
                const float* kk = k_cache.data() + static_cast<size_t>(tok) * kv_dim + static_cast<size_t>(kh) * head_dim;
                const float dot = dot_product_reference(qh, kk, static_cast<size_t>(head_dim));
                scores[static_cast<size_t>(tok)] = dot * attn_scale;
                max_score = std::max(max_score, scores[static_cast<size_t>(tok)]);
            }

            double denom = 0.0;
            for (int tok = 0; tok <= pos; ++tok) {
                float& s = scores[static_cast<size_t>(tok)];
                s = std::exp(s - max_score);
                denom += s;
            }
            const float inv_denom = denom > 0.0 ? static_cast<float>(1.0 / denom) : 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int tok = 0; tok <= pos; ++tok) {
                    const float prob = scores[static_cast<size_t>(tok)] * inv_denom;
                    const float* vv = v_cache.data() + static_cast<size_t>(tok) * kv_dim + static_cast<size_t>(kh) * head_dim;
                    acc = std::fma(prob, vv[d], acc);
                }
                ctx[static_cast<size_t>(h) * head_dim + d] = acc;
            }
        }
        auto attn1 = Clock::now();

        auto o0 = Clock::now();
        std::vector<float> attn_out = linear_with_named_weight(ctx, wo_name, hidden, hidden, prefix + " o_proj");
        auto o1 = Clock::now();
        std::vector<float> after_attn(hidden);
        for (size_t i = 0; i < hidden; ++i) after_attn[i] = residual[i] + attn_out[i];

        const std::string ln2_name = layer_weight_name(layer, "post_attention_layernorm.weight");
        const std::string wgate_name = layer_weight_name(layer, "mlp.gate_proj.weight");
        const std::string wup_name = layer_weight_name(layer, "mlp.up_proj.weight");
        const std::string wdown_name = layer_weight_name(layer, "mlp.down_proj.weight");
        const std::vector<float>& ln2 = cached_weight_float_ref(ln2_name);
        std::vector<float> mlp_in = after_attn;
        auto norm2_0 = Clock::now();
        rms_norm_inplace(mlp_in, ln2);
        auto norm2_1 = Clock::now();
        DeviceMlpTiming mlp_timing;
        bool used_device_mlp = false;
        std::vector<float> mlp_out;
        if (aclnn_mlp_enabled() && !aclnn_mlp_disabled) {
            std::string reason;
            used_device_mlp = mlp_aclnn_forward(
                mlp_in,
                wgate_name,
                wup_name,
                wdown_name,
                mlp_out,
                mlp_timing,
                reason);
            if (!used_device_mlp) {
                aclnn_mlp_disabled = true;
                time_log("[Ascend][warn] ACLNN MLP disabled, fallback=cpu, reason=" + reason);
                if (!aclnn_mlp_fallback_enabled()) {
                    throw std::runtime_error("ACLNN MLP failed and ASCEND_MLP_FALLBACK=0: " + reason);
                }
            }
        }
        auto gate0 = Clock::now();
        auto gate1 = gate0;
        auto down0 = gate0;
        auto down1 = gate0;
        if (!used_device_mlp) {
            std::vector<float> mid = gate_up_silu_named(
                mlp_in,
                wgate_name,
                wup_name,
                static_cast<size_t>(config.intermediate),
                hidden,
                prefix + " gate_up_silu");
            gate1 = Clock::now();
            down0 = Clock::now();
            mlp_out = linear_with_named_weight(mid, wdown_name, hidden, static_cast<size_t>(config.intermediate), prefix + " down_proj");
            down1 = Clock::now();
            mlp_timing.gate_up_ms = elapsed_ms(gate0, gate1);
            mlp_timing.down_ms = elapsed_ms(down0, down1);
        }
        for (size_t i = 0; i < hidden; ++i) after_attn[i] += mlp_out[i];
        auto layer1 = Clock::now();
        if (profile) {
            profile->norm1_ms += elapsed_ms(norm1_0, norm1_1);
            profile->q_ms += elapsed_ms(q0, q1);
            profile->kv_ms += elapsed_ms(kv0, kv1);
            profile->rope_ms += elapsed_ms(rope0, rope1);
            profile->attn_ms += elapsed_ms(attn0, attn1);
            profile->o_ms += elapsed_ms(o0, o1);
            profile->norm2_ms += elapsed_ms(norm2_0, norm2_1);
            profile->gate_up_ms += mlp_timing.gate_up_ms;
            profile->down_ms += mlp_timing.down_ms;
            profile->total_ms += elapsed_ms(layer0, layer1);
        }
        return after_attn;
    }

    std::vector<float> all_layers_last_token_reference() {
        if (prompt_len <= 0) throw std::runtime_error("all_layers_ref requires prefill first");
        const int head_dim = config.hidden / config.n_heads;
        const int kv_dim = config.n_kv_heads * head_dim;
        ensure_full_ref_cache(kv_dim);
        if (full_ref_cached_len > prompt_len) {
            full_ref_cached_len = 0;
            full_last_hidden.clear();
        }

        auto t0 = Clock::now();
        const int start = full_ref_cached_len;
        auto load0 = Clock::now();
        std::vector<float> hidden_rows = load_hidden_rows_float_range(start, prompt_len - start);
        auto load1 = Clock::now();
        auto layers0 = Clock::now();
        const bool profile_layers = ref_layer_profile_enabled();
        const int profile_token_limit = ref_layer_profile_token_limit();
        int profiled_tokens = 0;
        RefLayerProfile profile_total;
        for (int tok = start; tok < prompt_len; ++tok) {
            const bool profile_this_token =
                profile_layers && (profile_token_limit <= 0 || tok < profile_token_limit);
            const size_t local_tok = static_cast<size_t>(tok - start);
            std::vector<float> x(
                hidden_rows.begin() + local_tok * static_cast<size_t>(config.hidden),
                hidden_rows.begin() + (local_tok + 1) * static_cast<size_t>(config.hidden));
            RefLayerProfile token_profile;
            for (int layer = 0; layer < config.n_layers; ++layer) {
                x = layer_forward_reference(
                    std::move(x),
                    layer,
                    tok,
                    profile_this_token ? &token_profile : nullptr);
            }
            if (profile_this_token) {
                profiled_tokens++;
                profile_total.add(token_profile);
                time_log("[Ascend][profile] all_layers token=" + std::to_string(tok) +
                         ", norm1_ms=" + std::to_string(token_profile.norm1_ms) +
                         ", q_ms=" + std::to_string(token_profile.q_ms) +
                         ", kv_ms=" + std::to_string(token_profile.kv_ms) +
                         ", rope_ms=" + std::to_string(token_profile.rope_ms) +
                         ", attn_ms=" + std::to_string(token_profile.attn_ms) +
                         ", o_ms=" + std::to_string(token_profile.o_ms) +
                         ", norm2_ms=" + std::to_string(token_profile.norm2_ms) +
                         ", gate_up_ms=" + std::to_string(token_profile.gate_up_ms) +
                         ", down_ms=" + std::to_string(token_profile.down_ms) +
                         ", total_ms=" + std::to_string(token_profile.total_ms));
            }
            full_last_hidden = std::move(x);
            full_ref_cached_len = tok + 1;
        }
        auto layers1 = Clock::now();
        if (profile_layers && profiled_tokens > 0) {
            time_log("[Ascend][profile] all_layers aggregate, tokens_profiled=" +
                     std::to_string(profiled_tokens) +
                     ", norm1_ms=" + std::to_string(profile_total.norm1_ms) +
                     ", q_ms=" + std::to_string(profile_total.q_ms) +
                     ", kv_ms=" + std::to_string(profile_total.kv_ms) +
                     ", rope_ms=" + std::to_string(profile_total.rope_ms) +
                     ", attn_ms=" + std::to_string(profile_total.attn_ms) +
                     ", o_ms=" + std::to_string(profile_total.o_ms) +
                     ", norm2_ms=" + std::to_string(profile_total.norm2_ms) +
                     ", gate_up_ms=" + std::to_string(profile_total.gate_up_ms) +
                     ", down_ms=" + std::to_string(profile_total.down_ms) +
                     ", total_ms=" + std::to_string(profile_total.total_ms));
        }
        if (full_last_hidden.empty()) {
            throw std::runtime_error("all_layers_ref has no cached hidden state");
        }
        auto norm0 = Clock::now();
        std::vector<float> final_out = final_norm_vector(full_last_hidden);
        auto norm1 = Clock::now();
        auto t1 = Clock::now();
        time_log("[Ascend][time] all_layers reference finished, tokens=" +
                 std::to_string(prompt_len) +
                 ", processed_from=" + std::to_string(start) +
                 ", processed_to=" + std::to_string(full_ref_cached_len) +
                 ", layers=" + std::to_string(config.n_layers) +
                 ", load_hidden_ms=" + std::to_string(elapsed_ms(load0, load1)) +
                 ", layers_ms=" + std::to_string(elapsed_ms(layers0, layers1)) +
                 ", final_norm_ms=" + std::to_string(elapsed_ms(norm0, norm1)) +
                 ", elapsed_ms=" + std::to_string(elapsed_ms(t0, t1)));
        return final_out;
    }

    std::vector<float> layer0_last_token_reference() {
        if (prompt_len <= 0) throw std::runtime_error("layer0_ref requires prefill first");
        const size_t hidden = static_cast<size_t>(config.hidden);
        const int head_dim = config.hidden / config.n_heads;
        const int kv_dim = config.n_kv_heads * head_dim;
        const int group = config.n_heads / config.n_kv_heads;
        if (head_dim <= 0 || group <= 0 || config.n_heads % config.n_kv_heads != 0) {
            throw std::runtime_error("bad attention head config for layer0_ref");
        }

        auto t0 = Clock::now();
        const int len = prompt_len;
        std::vector<float> hidden_rows = load_hidden_rows_float(len);
        const size_t last_off = static_cast<size_t>(len - 1) * hidden;
        std::vector<float> residual(hidden_rows.begin() + last_off, hidden_rows.begin() + last_off + hidden);

        const std::vector<float>& ln1 = cached_weight_float_ref("model.layers.0.input_layernorm.weight");
        const std::vector<float>& wq = cached_weight_float_ref("model.layers.0.self_attn.q_proj.weight");
        const std::vector<float>& wk = cached_weight_float_ref("model.layers.0.self_attn.k_proj.weight");
        const std::vector<float>& wv = cached_weight_float_ref("model.layers.0.self_attn.v_proj.weight");
        const std::vector<float>& wo = cached_weight_float_ref("model.layers.0.self_attn.o_proj.weight");

        const auto q_shape = matrix_shape("model.layers.0.self_attn.q_proj.weight");
        const auto k_shape = matrix_shape("model.layers.0.self_attn.k_proj.weight");
        const auto v_shape = matrix_shape("model.layers.0.self_attn.v_proj.weight");
        const auto o_shape = matrix_shape("model.layers.0.self_attn.o_proj.weight");
        if (q_shape.first != hidden || q_shape.second != hidden ||
            k_shape.first != static_cast<size_t>(kv_dim) || k_shape.second != hidden ||
            v_shape.first != static_cast<size_t>(kv_dim) || v_shape.second != hidden ||
            o_shape.first != hidden || o_shape.second != hidden) {
            throw std::runtime_error("layer0 attention projection shape mismatch");
        }

        auto t_load = Clock::now();
        std::vector<float> q_in = residual;
        rms_norm_inplace(q_in, ln1);
        std::vector<float> q = linear_with_weight(q_in, wq, hidden, hidden, "layer0 q_proj");
        apply_rope(q, config.n_heads, len - 1);
        auto t_q = Clock::now();

        const bool kv_cache_enabled = env_str_or("ASCEND_REF_KV_CACHE", "1") != "0";
        if (!kv_cache_enabled || layer0_kv_dim != kv_dim || layer0_kv_cached_len > len) {
            layer0_k_cache.clear();
            layer0_v_cache.clear();
            layer0_kv_cached_len = 0;
            layer0_kv_dim = kv_dim;
        }
        if (layer0_k_cache.size() < static_cast<size_t>(len) * kv_dim) {
            layer0_k_cache.resize(static_cast<size_t>(len) * kv_dim);
            layer0_v_cache.resize(static_cast<size_t>(len) * kv_dim);
        }

        const int cached_before = layer0_kv_cached_len;
        for (int tok = layer0_kv_cached_len; tok < len; ++tok) {
            std::vector<float> x(hidden_rows.begin() + static_cast<size_t>(tok) * hidden,
                                 hidden_rows.begin() + static_cast<size_t>(tok + 1) * hidden);
            rms_norm_inplace(x, ln1);
            std::vector<float> k = linear_with_weight(x, wk, static_cast<size_t>(kv_dim), hidden, "layer0 k_proj");
            std::vector<float> v = linear_with_weight(x, wv, static_cast<size_t>(kv_dim), hidden, "layer0 v_proj");
            apply_rope(k, config.n_kv_heads, tok);
            std::copy(k.begin(), k.end(), layer0_k_cache.begin() + static_cast<size_t>(tok) * kv_dim);
            std::copy(v.begin(), v.end(), layer0_v_cache.begin() + static_cast<size_t>(tok) * kv_dim);
        }
        layer0_kv_cached_len = len;
        auto t_kv = Clock::now();

        std::vector<float> ctx(hidden, 0.0f);
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::vector<float> scores(len);
        for (int h = 0; h < config.n_heads; ++h) {
            const int kh = h / group;
            const float* qh = q.data() + static_cast<size_t>(h) * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            for (int tok = 0; tok < len; ++tok) {
                const float* kk = layer0_k_cache.data() + static_cast<size_t>(tok) * kv_dim + static_cast<size_t>(kh) * head_dim;
                const float dot = dot_product_reference(qh, kk, static_cast<size_t>(head_dim));
                scores[tok] = dot * attn_scale;
                max_score = std::max(max_score, scores[tok]);
            }

            double denom = 0.0;
            for (int tok = 0; tok < len; ++tok) {
                scores[tok] = std::exp(scores[tok] - max_score);
                denom += scores[tok];
            }
            const float inv_denom = denom > 0.0 ? static_cast<float>(1.0 / denom) : 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int tok = 0; tok < len; ++tok) {
                    const float prob = scores[tok] * inv_denom;
                    const float* vv = layer0_v_cache.data() + static_cast<size_t>(tok) * kv_dim + static_cast<size_t>(kh) * head_dim;
                    acc = std::fma(prob, vv[d], acc);
                }
                ctx[static_cast<size_t>(h) * head_dim + d] = acc;
            }
        }
        auto t_attn = Clock::now();

        std::vector<float> attn_out = linear_with_weight(ctx, wo, hidden, hidden, "layer0 o_proj");
        std::vector<float> after_attn(hidden);
        for (size_t i = 0; i < hidden; ++i) after_attn[i] = residual[i] + attn_out[i];
        auto t_o = Clock::now();

        const std::vector<float>& ln2 = cached_weight_float_ref("model.layers.0.post_attention_layernorm.weight");
        std::vector<float> mlp_in = after_attn;
        rms_norm_inplace(mlp_in, ln2);

        const std::vector<float>& wgate = cached_weight_float_ref("model.layers.0.mlp.gate_proj.weight");
        const std::vector<float>& wup = cached_weight_float_ref("model.layers.0.mlp.up_proj.weight");
        const std::vector<float>& wdown = cached_weight_float_ref("model.layers.0.mlp.down_proj.weight");
        const auto gate_shape = matrix_shape("model.layers.0.mlp.gate_proj.weight");
        const auto up_shape = matrix_shape("model.layers.0.mlp.up_proj.weight");
        const auto down_shape = matrix_shape("model.layers.0.mlp.down_proj.weight");
        if (gate_shape.first != static_cast<size_t>(config.intermediate) || gate_shape.second != hidden ||
            up_shape.first != static_cast<size_t>(config.intermediate) || up_shape.second != hidden ||
            down_shape.first != hidden || down_shape.second != static_cast<size_t>(config.intermediate)) {
            throw std::runtime_error("layer0 MLP projection shape mismatch");
        }

        std::vector<float> mid = gate_up_silu_reference(
            mlp_in,
            wgate,
            wup,
            static_cast<size_t>(config.intermediate),
            hidden,
            "layer0 gate_up_silu");

        std::vector<float> mlp_out = linear_with_weight(mid, wdown, hidden, static_cast<size_t>(config.intermediate), "layer0 down_proj");
        std::vector<float> out(hidden);
        for (size_t i = 0; i < hidden; ++i) out[i] = after_attn[i] + mlp_out[i];

        auto t_mlp = Clock::now();
        std::vector<float> final_out = final_norm_vector(out);
        auto t1 = Clock::now();
        time_log("[Ascend][time] layer0 reference finished, tokens=" +
                 std::to_string(len) +
                 ", kv_cached_before=" + std::to_string(cached_before) +
                 ", kv_cached_after=" + std::to_string(layer0_kv_cached_len) +
                 ", hidden=" + std::to_string(hidden) +
                 ", head_dim=" + std::to_string(head_dim) +
                 ", kv_dim=" + std::to_string(kv_dim) +
                 ", load_ms=" + std::to_string(elapsed_ms(t0, t_load)) +
                 ", q_ms=" + std::to_string(elapsed_ms(t_load, t_q)) +
                 ", kv_ms=" + std::to_string(elapsed_ms(t_q, t_kv)) +
                 ", attn_ms=" + std::to_string(elapsed_ms(t_kv, t_attn)) +
                 ", o_ms=" + std::to_string(elapsed_ms(t_attn, t_o)) +
                 ", mlp_ms=" + std::to_string(elapsed_ms(t_o, t_mlp)) +
                 ", final_norm_ms=" + std::to_string(elapsed_ms(t_mlp, t1)) +
                 ", elapsed_ms=" + std::to_string(elapsed_ms(t0, t1)));
        return final_out;
    }

    std::vector<float> load_last_hidden_with_final_norm() {
        if (prompt_len <= 0) throw std::runtime_error("decode requires prefill first");
        if (!d_hidden || hidden_row_bytes == 0) {
            throw std::runtime_error("decode requires hidden buffer; load embedding and keep ASCEND_RUN_EMBED=1");
        }

        auto norm_it = d_weights.find("model.norm.weight");
        if (norm_it == d_weights.end()) {
            throw std::runtime_error("decode requires model.norm.weight; use ASCEND_LOAD_WEIGHTS=minimal/layer0/all");
        }
        const DeviceTensor& norm = norm_it->second;
        if (norm.meta.shape.size() != 1 || norm.meta.shape[0] != static_cast<size_t>(config.hidden)) {
            throw std::runtime_error("bad final RMSNorm weight shape=" + shape_string(norm.meta.shape));
        }

        const TensorMeta& hidden_meta = d_weights.at("model.embed_tokens.weight").meta;
        const size_t hidden_dtype_bytes = dtype_size_bytes(hidden_meta);
        const size_t norm_dtype_bytes = dtype_size_bytes(norm.meta);
        const size_t hidden = static_cast<size_t>(config.hidden);
        if (hidden_row_bytes != hidden * hidden_dtype_bytes) {
            throw std::runtime_error("hidden row bytes mismatch before decode");
        }

        std::vector<unsigned char> h_hidden(hidden_row_bytes);
        std::vector<unsigned char> h_norm(norm.bytes);
        const size_t row_index = static_cast<size_t>(prompt_len - 1);
        char* row_ptr = static_cast<char*>(d_hidden) + row_index * hidden_row_bytes;
        check_acl(aclrtMemcpy(h_hidden.data(), hidden_row_bytes, row_ptr, hidden_row_bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(D2H last hidden for decode)");
        check_acl(aclrtMemcpy(h_norm.data(), norm.bytes, norm.data, norm.bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(D2H final norm for decode)");

        std::vector<float> x(hidden);
        double sum_sq = 0.0;
        for (size_t j = 0; j < hidden; ++j) {
            const float v = load_scalar(h_hidden.data() + j * hidden_dtype_bytes, hidden_meta.dtype);
            x[j] = v;
            sum_sq += static_cast<double>(v) * static_cast<double>(v);
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(sum_sq / hidden) + config.rms_norm_eps);
        for (size_t j = 0; j < hidden; ++j) {
            const float w = load_scalar(h_norm.data() + j * norm_dtype_bytes, norm.meta.dtype);
            x[j] = x[j] * scale * w;
        }
        return x;
    }

    int lm_head_argmax_aclnn(
        const std::vector<float>& x,
        const DeviceTensor& head,
        size_t vocab,
        size_t hidden,
        size_t vocab_limit,
        bool suppress_special,
        float& best_out,
        double& elapsed_out,
        std::string& reason) {
        std::vector<float> logits;
        DeviceLinearTiming timing;
        if (!vector_mm_aclnn_forward(
                x,
                head,
                vocab,
                hidden,
                logits,
                timing,
                "ACLNN lm_head",
                reason)) {
            return -1;
        }
        if (logits.size() != vocab) {
            reason = "ACLNN lm_head logits size mismatch";
            return -1;
        }

        float best = -std::numeric_limits<float>::infinity();
        int best_id = 0;
        for (size_t tok = 0; tok < vocab_limit; ++tok) {
            if (suppress_special && tok >= 151000) continue;
            float logit = logits[tok];
            if (tok < seen_tokens.size() && seen_tokens[tok] && repetition_penalty > 1.0f) {
                logit = logit >= 0.0f ? logit / repetition_penalty : logit * repetition_penalty;
            }
            if (logit > best || (logit == best && static_cast<int>(tok) < best_id)) {
                best = logit;
                best_id = static_cast<int>(tok);
            }
        }

        best_out = best;
        elapsed_out = timing.total_ms;
        return best_id;
    }

    int lm_head_argmax_reference(const std::vector<float>& x) {
        auto head_it = d_weights.find("lm_head.weight");
        if (head_it == d_weights.end()) {
            throw std::runtime_error("decode requires lm_head.weight; use ASCEND_LOAD_WEIGHTS=minimal/layer0/all");
        }

        const DeviceTensor& head = head_it->second;
        const TensorMeta& meta = head.meta;
        if (meta.shape.size() != 2 ||
            meta.shape[0] != static_cast<size_t>(config.vocab_size) ||
            meta.shape[1] != static_cast<size_t>(config.hidden)) {
            throw std::runtime_error("bad lm_head.weight shape=" + shape_string(meta.shape));
        }

        const size_t vocab = meta.shape[0];
        const size_t hidden = meta.shape[1];
        auto t0 = Clock::now();
        const size_t vocab_limit_env = static_cast<size_t>(std::max(0, env_int_or("ASCEND_LM_HEAD_REF_VOCAB", 0)));
        const size_t vocab_limit = vocab_limit_env > 0 ? std::min(vocab, vocab_limit_env) : vocab;
        const bool suppress_special = env_str_or("ASCEND_SUPPRESS_SPECIAL", "0") != "0";

        if (aclnn_lm_head_enabled() && !aclnn_lm_head_disabled) {
            std::string reason;
            float best = -std::numeric_limits<float>::infinity();
            double device_ms = 0.0;
            const int token = lm_head_argmax_aclnn(
                x,
                head,
                vocab,
                hidden,
                vocab_limit,
                suppress_special,
                best,
                device_ms,
                reason);
            if (token >= 0) {
                auto t1 = Clock::now();
                if (!aclnn_lm_head_notice_printed || aclnn_lm_head_log_enabled()) {
                    time_log("[Ascend][time] ACLNN lm_head active, vocab_scanned=" +
                             std::to_string(vocab_limit) +
                             ", hidden=" + std::to_string(hidden) +
                             ", weight_dtype=" + meta.dtype +
                             ", token=" + std::to_string(token) +
                             ", logit=" + std::to_string(best) +
                             ", device_ms=" + std::to_string(device_ms) +
                             ", elapsed_ms=" + std::to_string(elapsed_ms(t0, t1)));
                    aclnn_lm_head_notice_printed = true;
                } else {
                    time_log("[Ascend][time] lm_head argmax aclnn finished, vocab_scanned=" +
                             std::to_string(vocab_limit) +
                             ", hidden=" + std::to_string(hidden) +
                             ", weight_dtype=" + meta.dtype +
                             ", token=" + std::to_string(token) +
                             ", logit=" + std::to_string(best) +
                             ", elapsed_ms=" + std::to_string(elapsed_ms(t0, t1)));
                }
                return token;
            }

            aclnn_lm_head_disabled = true;
            time_log("[Ascend][warn] ACLNN lm_head disabled, fallback=cpu, reason=" + reason);
            if (!aclnn_lm_head_fallback_enabled()) {
                throw std::runtime_error("ACLNN lm_head failed and ASCEND_LM_HEAD_FALLBACK=0: " + reason);
            }
        }

        const bool use_u16_head = ref_u16_weight_enabled() && u16_weight_dtype_supported(meta.dtype);
        const bool head_bf16 = meta.dtype == "BF16";
        const std::vector<uint16_t>* h_head_u16 = nullptr;
        const std::vector<float>* h_head_float = nullptr;
        if (use_u16_head) {
            h_head_u16 = &cached_weight_u16_ref("lm_head.weight");
        } else {
            h_head_float = &cached_weight_float_ref("lm_head.weight");
        }
        const int requested_threads = env_int_or("ASCEND_LM_HEAD_THREADS", 0);
        const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
        const int n_threads = std::max(
            1,
            std::min<int>(
                requested_threads > 0 ? requested_threads : static_cast<int>(hw_threads),
                static_cast<int>(std::max<size_t>(1, vocab_limit))));

        struct LocalBest {
            float value = -std::numeric_limits<float>::infinity();
            int token = 0;
        };
        std::vector<LocalBest> local(static_cast<size_t>(n_threads));

        auto scan_range = [&](int tid) {
            const size_t begin = (vocab_limit * static_cast<size_t>(tid)) / static_cast<size_t>(n_threads);
            const size_t end = (vocab_limit * static_cast<size_t>(tid + 1)) / static_cast<size_t>(n_threads);
            float best = -std::numeric_limits<float>::infinity();
            int best_id = static_cast<int>(begin);
            for (size_t tok = begin; tok < end; ++tok) {
                if (suppress_special && tok >= 151000) continue;
                float logit = 0.0f;
                if (use_u16_head) {
                    const uint16_t* wrow = h_head_u16->data() + tok * hidden;
                    logit = dot_product_u16_weight_reference(x.data(), wrow, hidden, head_bf16);
                } else {
                    const float* wrow = h_head_float->data() + tok * hidden;
                    logit = dot_product_reference(x.data(), wrow, hidden);
                }
                if (tok < seen_tokens.size() && seen_tokens[tok] && repetition_penalty > 1.0f) {
                    logit = logit >= 0.0f ? logit / repetition_penalty : logit * repetition_penalty;
                }
                if (logit > best || (logit == best && static_cast<int>(tok) < best_id)) {
                    best = logit;
                    best_id = static_cast<int>(tok);
                }
            }
            local[static_cast<size_t>(tid)] = {best, best_id};
        };

        if (n_threads == 1) {
            scan_range(0);
        } else {
            ref_pool_for(n_threads).run(n_threads, scan_range);
        }

        int best_id = 0;
        float best = -std::numeric_limits<float>::infinity();
        for (const auto& item : local) {
            if (item.value > best || (item.value == best && item.token < best_id)) {
                best = item.value;
                best_id = item.token;
            }
        }

        auto t1 = Clock::now();
        time_log("[Ascend][time] lm_head argmax reference finished, vocab_scanned=" +
                 std::to_string(vocab_limit) +
                 ", hidden=" + std::to_string(hidden) +
                 ", weight_dtype=" + meta.dtype +
                 ", weight_cache=" + std::string(use_u16_head ? "u16" : "fp32") +
                 ", threads=" + std::to_string(n_threads) +
                 ", token=" + std::to_string(best_id) +
                 ", logit=" + std::to_string(best) +
                 ", elapsed_ms=" + std::to_string(elapsed_ms(t0, t1)));
        return best_id;
    }

    void append_generated_token(int token) {
        if (prompt_len >= max_seq) throw std::runtime_error("decode exceeds max_seq");
        const size_t token_offset = static_cast<size_t>(prompt_len) * sizeof(int);
        check_acl(aclrtMemcpy(static_cast<char*>(d_tokens) + token_offset,
                              token_bytes - token_offset,
                              &token,
                              sizeof(int),
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(H2D generated token)");

        auto embed_it = d_weights.find("model.embed_tokens.weight");
        if (embed_it != d_weights.end() && d_hidden && hidden_row_bytes > 0) {
            const DeviceTensor& embed = embed_it->second;
            if (token < 0 || static_cast<size_t>(token) >= embed.meta.shape[0]) {
                throw std::runtime_error("generated token out of embedding vocab range: " + std::to_string(token));
            }
            char* src = static_cast<char*>(embed.data) + static_cast<size_t>(token) * hidden_row_bytes;
            char* dst = static_cast<char*>(d_hidden) + static_cast<size_t>(prompt_len) * hidden_row_bytes;
            check_acl(aclrtMemcpy(dst, hidden_row_bytes, src, hidden_row_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE),
                      "aclrtMemcpy(D2D generated token embedding)");
        }

        if (token >= 0 && static_cast<size_t>(token) < seen_tokens.size()) {
            seen_tokens[static_cast<size_t>(token)] = 1;
        }
        prompt_len++;
    }

    int decode_one(int* out_token) {
        if (!out_token) throw std::runtime_error("decode output pointer is null");
        const std::string mode = env_str_or("ASCEND_DIRECT_DECODE", "lm_head_ref");
        if (mode != "lm_head_ref" && mode != "layer0_ref" && mode != "all_layers_ref") {
            throw std::runtime_error(
                "unsupported ASCEND_DIRECT_DECODE=" + mode +
                ", use one of: lm_head_ref, layer0_ref, all_layers_ref");
        }

        auto t0 = Clock::now();
        std::vector<float> x;
        if (mode == "all_layers_ref") {
            x = all_layers_last_token_reference();
        } else if (mode == "layer0_ref") {
            x = layer0_last_token_reference();
        } else {
            x = load_last_hidden_with_final_norm();
        }
        const int token = lm_head_argmax_reference(x);
        append_generated_token(token);
        *out_token = token;
        auto t1 = Clock::now();
        time_log("[Ascend][time] decode " + mode + " finished, token=" +
                 std::to_string(token) +
                 ", pos=" + std::to_string(prompt_len) +
                 ", elapsed_ms=" + std::to_string(elapsed_ms(t0, t1)));
        return 0;
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

int llm_set_repetition_penalty(void* handle, float penalty) {
    try {
        if (!handle) throw std::runtime_error("engine handle is null");
        auto* e = reinterpret_cast<AscendEngine*>(handle);
        e->set_repetition_penalty(penalty);
        return 0;
    } catch (const std::exception& e) {
        return fail(e);
    }
}

const char* llm_last_error() {
    return g_err.c_str();
}

const char* llm_backend_name() {
    return "ascend-direct-acl";
}

}
