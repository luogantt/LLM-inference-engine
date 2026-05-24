#ifndef USE_MUSA
#define USE_MUSA 0
#endif

#if USE_MUSA
#include <musa_runtime.h>
#include <musa_fp16.h>
#else
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#endif
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>
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

#if USE_MUSA
#define BACKEND_NAME "MUSA"
using cudaError_t = musaError_t;
#define cudaSuccess musaSuccess
#define cudaGetErrorString musaGetErrorString
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaMemcpy musaMemcpy
#define cudaMemset musaMemset
#define cudaFree musaFree
#define cudaDeviceSynchronize musaDeviceSynchronize
template <typename T>
static inline musaError_t cuda_malloc_compat(T** p,size_t n){
    return musaMalloc(reinterpret_cast<void**>(p),n);
}
#define cudaMalloc cuda_malloc_compat
#else
#define BACKEND_NAME "CUDA"
#endif

#define CK(x) do { cudaError_t _cuda_err=(x); if(_cuda_err!=cudaSuccess) throw std::runtime_error(std::string(BACKEND_NAME ": ")+cudaGetErrorString(_cuda_err)); } while(0)

constexpr int N_LAYERS=28;
constexpr int HIDDEN=3584;
constexpr int N_HEADS=28;
constexpr int N_KV_HEADS=4;
constexpr int HEAD_DIM=128;
constexpr int KV_DIM=N_KV_HEADS*HEAD_DIM;
constexpr int INTERMEDIATE=18944;
constexpr int VOCAB_SIZE=152064;
constexpr float DEFAULT_RMS_NORM_EPS=1e-6f;
constexpr float DEFAULT_ROPE_THETA=1000000.0f;
#ifndef LINEAR_THREADS_CFG
#define LINEAR_THREADS_CFG 128
#endif
#ifndef USE_ATTENTION_SHM
#define USE_ATTENTION_SHM 0
#endif
#ifndef USE_FUSED_ROPE_ATTENTION
#define USE_FUSED_ROPE_ATTENTION 0
#endif
#ifndef USE_FUSED_MLP_QUANT
#define USE_FUSED_MLP_QUANT 0
#endif
#ifndef USE_FAST_SILU
#define USE_FAST_SILU 0
#endif
#ifndef USE_INT8_WEIGHTS
#define USE_INT8_WEIGHTS 0
#endif
#ifndef USE_INT4_WEIGHTS
#define USE_INT4_WEIGHTS 0
#endif
#ifndef USE_INT4_DP4A
#define USE_INT4_DP4A 0
#endif
#ifndef USE_INT4_DP4A_PREPACK
#define USE_INT4_DP4A_PREPACK 0
#endif
#ifndef USE_SOFT_DP4A
#define USE_SOFT_DP4A 0
#endif
#ifndef USE_LINEAR_I4_DP4A2
#define USE_LINEAR_I4_DP4A2 0
#endif
#ifndef USE_QKV_GATE_I4_DP4A2
#define USE_QKV_GATE_I4_DP4A2 0
#endif
#ifndef USE_LINEAR_I4_DP4A4
#define USE_LINEAR_I4_DP4A4 0
#endif
#ifndef USE_QKV_GATE_I4_DP4A4
#define USE_QKV_GATE_I4_DP4A4 0
#endif
#ifndef USE_FP16_KV_CACHE
#define USE_FP16_KV_CACHE 0
#endif
#if USE_INT8_WEIGHTS && USE_INT4_WEIGHTS
#error "USE_INT8_WEIGHTS and USE_INT4_WEIGHTS cannot both be enabled"
#endif
#if USE_INT4_WEIGHTS
using WeightT = uint8_t;
#elif USE_INT8_WEIGHTS
using WeightT = int8_t;
#else
using WeightT = half;
#endif
#if USE_FP16_KV_CACHE
using KvCacheT = half;
#else
using KvCacheT = float;
#endif
struct WeightMatrix {
    WeightT* data=nullptr;
    float* scale=nullptr;
};
constexpr int WMMA_TILE=16;
constexpr int LINEAR_THREADS=LINEAR_THREADS_CFG;
constexpr int ARGMAX_BLOCKS=256;
#ifndef USE_WMMA_LINEAR
#define USE_WMMA_LINEAR 0
#endif

__device__ __forceinline__ float kv_cache_load(KvCacheT v) {
#if USE_FP16_KV_CACHE
    return __half2float(v);
#else
    return v;
#endif
}

__device__ __forceinline__ KvCacheT kv_cache_store(float v) {
#if USE_FP16_KV_CACHE
    return __float2half_rn(v);
#else
    return v;
#endif
}

static thread_local std::string g_err;
using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static void reset_time_log() {
    std::ofstream f("log.txt", std::ios::trunc);
    if (f) f << "[C++][time] log reset\n";
}

static void time_log(const std::string& line) {
    std::cout << line << "\n";
    std::ofstream f("log.txt", std::ios::app);
    if (f) f << line << "\n";
}

struct ModelConfig {
    int n_layers=N_LAYERS;
    int hidden=HIDDEN;
    int n_heads=N_HEADS;
    int n_kv_heads=N_KV_HEADS;
    int intermediate=INTERMEDIATE;
    int vocab_size=VOCAB_SIZE;
    float rms_norm_eps=DEFAULT_RMS_NORM_EPS;
    float rope_theta=DEFAULT_ROPE_THETA;
};

struct TensorMeta {
    std::string file;
    std::string dtype;
    std::vector<size_t> shape;
    uint64_t begin=0,end=0,data_base=0;
};

static bool ends_with(const std::string& s,const std::string& suf){
    return s.size()>=suf.size() && s.compare(s.size()-suf.size(),suf.size(),suf)==0;
}
static std::string path_join(const std::string& a,const std::string& b){
    return (!a.empty() && a.back()=='/') ? a+b : a+"/"+b;
}
static std::string read_text_file(const std::string& path){
    std::ifstream f(path);
    if(!f) return "";
    std::stringstream ss; ss<<f.rdbuf(); return ss.str();
}
static int json_int_or(const std::string& json,const std::string& key,int fallback){
    std::regex re("\""+key+"\"\\s*:\\s*(-?\\d+)");
    std::smatch m;
    return std::regex_search(json,m,re) ? std::stoi(m[1].str()) : fallback;
}
static float json_float_or(const std::string& json,const std::string& key,float fallback){
    std::regex re("\""+key+"\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch m;
    return std::regex_search(json,m,re) ? std::stof(m[1].str()) : fallback;
}
static void require_config_value(const std::string& name,int actual,int expected){
    if(actual!=expected){
        throw std::runtime_error("unsupported config "+name+"="+std::to_string(actual)+
            ", this binary was compiled for "+std::to_string(expected));
    }
}
static ModelConfig load_config(const std::string& dir){
    ModelConfig c;
    std::string path=path_join(dir,"config.json");
    std::string json=read_text_file(path);
    if(json.empty()){
        std::cout<<"[C++] config.json not found, using compiled defaults\n";
        return c;
    }
    c.n_layers=json_int_or(json,"num_hidden_layers",c.n_layers);
    c.hidden=json_int_or(json,"hidden_size",c.hidden);
    c.n_heads=json_int_or(json,"num_attention_heads",c.n_heads);
    c.n_kv_heads=json_int_or(json,"num_key_value_heads",c.n_kv_heads);
    c.intermediate=json_int_or(json,"intermediate_size",c.intermediate);
    c.vocab_size=json_int_or(json,"vocab_size",c.vocab_size);
    c.rms_norm_eps=json_float_or(json,"rms_norm_eps",c.rms_norm_eps);
    c.rope_theta=json_float_or(json,"rope_theta",c.rope_theta);

    require_config_value("num_hidden_layers",c.n_layers,N_LAYERS);
    require_config_value("hidden_size",c.hidden,HIDDEN);
    require_config_value("num_attention_heads",c.n_heads,N_HEADS);
    require_config_value("num_key_value_heads",c.n_kv_heads,N_KV_HEADS);
    require_config_value("intermediate_size",c.intermediate,INTERMEDIATE);
    require_config_value("vocab_size",c.vocab_size,VOCAB_SIZE);
    std::cout<<"[C++] config loaded: rms_norm_eps="<<c.rms_norm_eps
             <<", rope_theta="<<c.rope_theta<<"\n";
    return c;
}
static uint64_t read_u64_le(std::ifstream& f){
    unsigned char b[8]; f.read((char*)b,8);
    if(!f) throw std::runtime_error("read safetensors header length failed");
    uint64_t x=0; for(int i=0;i<8;i++) x|=(uint64_t)b[i]<<(8*i); return x;
}
static std::vector<size_t> parse_shape(const std::string& s){
    std::vector<size_t> v; std::stringstream ss(s); std::string it;
    while(std::getline(ss,it,',')){
        std::string t; for(char c:it) if(!std::isspace((unsigned char)c)) t.push_back(c);
        if(!t.empty()) v.push_back((size_t)std::stoull(t));
    }
    return v;
}
static size_t numel(const std::vector<size_t>& s){
    size_t n=1; for(size_t x:s) n*=x; return n;
}
static std::vector<std::string> list_safetensors(const std::string& dir){
    DIR* dp=opendir(dir.c_str());
    if(!dp) throw std::runtime_error("cannot open dir: "+dir);
    std::vector<std::string> fs;
    while(auto* e=readdir(dp)){
        std::string n=e->d_name;
        if(ends_with(n,".safetensors")) fs.push_back(path_join(dir,n));
    }
    closedir(dp); std::sort(fs.begin(),fs.end());
    if(fs.empty()) throw std::runtime_error("no safetensors found in "+dir);
    return fs;
}
static std::unordered_map<std::string,TensorMeta> scan_safetensors(const std::string& dir){
    std::unordered_map<std::string,TensorMeta> m;
    std::regex re("\"([^\"]+)\"\\s*:\\s*\\{[^\\}]*?\"dtype\"\\s*:\\s*\"([^\"]+)\"[^\\}]*?\"shape\"\\s*:\\s*\\[([^\\]]*)\\][^\\}]*?\"data_offsets\"\\s*:\\s*\\[\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\]");
    for(auto& p:list_safetensors(dir)){
        std::ifstream f(p,std::ios::binary);
        if(!f) throw std::runtime_error("cannot open "+p);
        uint64_t hlen=read_u64_le(f);
        std::string h(hlen,'\0');
        f.read(&h[0],hlen);
        if(!f) throw std::runtime_error("read header failed: "+p);
        int c=0;
        for(auto it=std::sregex_iterator(h.begin(),h.end(),re); it!=std::sregex_iterator(); ++it){
            std::smatch x=*it;
            TensorMeta t;
            t.file=p; t.dtype=x[2].str(); t.shape=parse_shape(x[3].str());
            t.begin=std::stoull(x[4].str()); t.end=std::stoull(x[5].str());
            t.data_base=8+hlen;
            m[x[1].str()]=t; c++;
        }
        std::cout<<"[C++] scanned "<<p<<", tensors="<<c<<"\n";
    }
    return m;
}
static float bf16_to_float(uint16_t b){
    uint32_t x=(uint32_t)b<<16; float y; std::memcpy(&y,&x,4); return y;
}
static float f16_to_float(uint16_t h){
    uint16_t he=h&0x7C00u, hs=h&0x03FFu; uint32_t fs=(uint32_t)(h&0x8000u)<<16;
    uint32_t fe,ff;
    if(he==0){
        if(hs==0){uint32_t x=fs; float y; std::memcpy(&y,&x,4); return y;}
        int sh=0; while((hs&0x0400u)==0){hs<<=1; sh++;} hs&=0x03FFu;
        fe=(uint32_t)(127-15-sh)<<23; ff=(uint32_t)hs<<13;
    }else if(he==0x7C00u){
        fe=0xFFu<<23; ff=(uint32_t)hs<<13;
    }else{
        fe=(uint32_t)((he>>10)+(127-15))<<23; ff=(uint32_t)hs<<13;
    }
    uint32_t x=fs|fe|ff; float y; std::memcpy(&y,&x,4); return y;
}
static std::vector<unsigned char> read_bytes(const TensorMeta& t){
    std::ifstream f(t.file,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open tensor file "+t.file);
    uint64_t off=t.data_base+t.begin, n=t.end-t.begin;
    f.seekg((std::streamoff)off,std::ios::beg);
    std::vector<unsigned char> b(n);
    f.read((char*)b.data(),n);
    if((uint64_t)f.gcount()!=n) throw std::runtime_error("read tensor bytes failed "+t.file);
    return b;
}
static float* load_tensor(const std::unordered_map<std::string,TensorMeta>& metas,const std::string& name){
    auto it=metas.find(name);
    if(it==metas.end()) throw std::runtime_error("missing tensor: "+name);
    auto& t=it->second; size_t n=numel(t.shape);
    auto raw=read_bytes(t);
    std::vector<float> h(n);
    if(t.dtype=="BF16"){
        if(raw.size()!=n*2) throw std::runtime_error("bad BF16 size "+name);
        auto* p=(const uint16_t*)raw.data(); for(size_t i=0;i<n;i++) h[i]=bf16_to_float(p[i]);
    }else if(t.dtype=="F16"){
        if(raw.size()!=n*2) throw std::runtime_error("bad F16 size "+name);
        auto* p=(const uint16_t*)raw.data(); for(size_t i=0;i<n;i++) h[i]=f16_to_float(p[i]);
    }else if(t.dtype=="F32"){
        if(raw.size()!=n*4) throw std::runtime_error("bad F32 size "+name);
        std::memcpy(h.data(),raw.data(),n*4);
    }else throw std::runtime_error("unsupported dtype "+t.dtype+" for "+name);
    float* d=nullptr; CK(cudaMalloc(&d,n*sizeof(float))); CK(cudaMemcpy(d,h.data(),n*sizeof(float),cudaMemcpyHostToDevice));
    std::cout<<"[C++] loaded "<<name<<", dtype="<<t.dtype<<", numel="<<n<<"\n";
    return d;
}
static WeightMatrix load_weight_tensor(const std::unordered_map<std::string,TensorMeta>& metas,const std::string& name){
    auto it=metas.find(name);
    if(it==metas.end()) throw std::runtime_error("missing tensor: "+name);
    auto& t=it->second; size_t n=numel(t.shape);
    auto raw=read_bytes(t);
    WeightMatrix w;
#if USE_INT8_WEIGHTS || USE_INT4_WEIGHTS
    size_t rows=t.shape.empty() ? 1 : t.shape[0];
    if(rows==0 || n%rows!=0) throw std::runtime_error("bad weight shape for "+name);
    size_t cols=n/rows;
#if USE_INT4_WEIGHTS && !USE_INT4_DP4A_PREPACK
    size_t packed_cols=(cols+1)/2;
    size_t packed_n=rows*packed_cols;
    std::vector<WeightT> h(packed_n);
#else
    std::vector<WeightT> h(n);
#endif
    std::vector<float> scales(rows);
    auto quantize_rows = [&](auto value_at){
        for(size_t r=0;r<rows;r++){
            float max_abs=0.0f;
            size_t base=r*cols;
            for(size_t c=0;c<cols;c++){
                float v=value_at(base+c);
                max_abs=fmaxf(max_abs,fabsf(v));
            }
#if USE_INT4_WEIGHTS
            float inv=max_abs>0.0f ? 7.0f/max_abs : 0.0f;
            float scale=max_abs>0.0f ? max_abs/7.0f : 1.0f;
#else
            float inv=max_abs>0.0f ? 127.0f/max_abs : 0.0f;
            float scale=max_abs>0.0f ? max_abs/127.0f : 1.0f;
#endif
            scales[r]=scale;
#if USE_INT4_WEIGHTS && USE_INT4_DP4A_PREPACK
            for(size_t c=0;c<cols;c++){
                int q=max_abs>0.0f ? (int)lrintf(value_at(base+c)*inv) : 0;
                q=std::max(-7,std::min(7,q));
                h[base+c]=(WeightT)((uint8_t)((int8_t)q));
            }
#elif USE_INT4_WEIGHTS
            size_t packed_base=r*packed_cols;
            for(size_t c=0;c<cols;c+=2){
                int q0=max_abs>0.0f ? (int)lrintf(value_at(base+c)*inv) : 0;
                q0=std::max(-7,std::min(7,q0));
                int q1=0;
                if(c+1<cols){
                    q1=max_abs>0.0f ? (int)lrintf(value_at(base+c+1)*inv) : 0;
                    q1=std::max(-7,std::min(7,q1));
                }
                h[packed_base+(c>>1)]=(WeightT)((q0&0x0F)|((q1&0x0F)<<4));
            }
#else
            for(size_t c=0;c<cols;c++){
                int q=max_abs>0.0f ? (int)lrintf(value_at(base+c)*inv) : 0;
                q=std::max(-127,std::min(127,q));
                h[base+c]=(WeightT)q;
            }
#endif
        }
    };
    if(t.dtype=="BF16"){
        if(raw.size()!=n*2) throw std::runtime_error("bad BF16 size "+name);
        auto* p=(const uint16_t*)raw.data();
        quantize_rows([&](size_t i){ return bf16_to_float(p[i]); });
    }else if(t.dtype=="F16"){
        if(raw.size()!=n*2) throw std::runtime_error("bad F16 size "+name);
        auto* p=(const uint16_t*)raw.data();
        quantize_rows([&](size_t i){ return f16_to_float(p[i]); });
    }else if(t.dtype=="F32"){
        if(raw.size()!=n*4) throw std::runtime_error("bad F32 size "+name);
        auto* p=(const float*)raw.data();
        quantize_rows([&](size_t i){ return p[i]; });
    }else throw std::runtime_error("unsupported dtype "+t.dtype+" for "+name);
    CK(cudaMalloc(&w.data,h.size()*sizeof(WeightT)));
    CK(cudaMemcpy(w.data,h.data(),h.size()*sizeof(WeightT),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&w.scale,rows*sizeof(float)));
    CK(cudaMemcpy(w.scale,scales.data(),rows*sizeof(float),cudaMemcpyHostToDevice));
#if USE_INT4_WEIGHTS
#if USE_INT4_DP4A_PREPACK
    std::cout<<"[C++] loaded "<<name<<", dtype="<<t.dtype
             <<", stored=INT4(dp4a_prepack,rowwise), rows="<<rows
             <<", cols="<<cols<<", bytes="<<h.size()<<", numel="<<n<<"\n";
#else
    std::cout<<"[C++] loaded "<<name<<", dtype="<<t.dtype
             <<", stored=INT4(rowwise), rows="<<rows<<", cols="<<cols
             <<", packed_bytes="<<h.size()<<", numel="<<n<<"\n";
#endif
#else
    std::cout<<"[C++] loaded "<<name<<", dtype="<<t.dtype
             <<", stored=INT8(rowwise), rows="<<rows<<", cols="<<cols
             <<", numel="<<n<<"\n";
#endif
#else
    std::vector<WeightT> h(n);
    if(t.dtype=="BF16"){
        if(raw.size()!=n*2) throw std::runtime_error("bad BF16 size "+name);
        auto* p=(const uint16_t*)raw.data();
        for(size_t i=0;i<n;i++) h[i]=__float2half_rn(bf16_to_float(p[i]));
    }else if(t.dtype=="F16"){
        if(raw.size()!=n*2) throw std::runtime_error("bad F16 size "+name);
        std::memcpy(h.data(),raw.data(),raw.size());
    }else if(t.dtype=="F32"){
        if(raw.size()!=n*4) throw std::runtime_error("bad F32 size "+name);
        auto* p=(const float*)raw.data();
        for(size_t i=0;i<n;i++) h[i]=__float2half_rn(p[i]);
    }else throw std::runtime_error("unsupported dtype "+t.dtype+" for "+name);
    CK(cudaMalloc(&w.data,n*sizeof(WeightT)));
    CK(cudaMemcpy(w.data,h.data(),n*sizeof(WeightT),cudaMemcpyHostToDevice));
    std::cout<<"[C++] loaded "<<name<<", dtype="<<t.dtype<<", stored=FP16, numel="<<n<<"\n";
#endif
    return w;
}

__device__ __forceinline__ float weight_to_float(WeightT w,float scale=1.0f){
#if USE_INT8_WEIGHTS
    return (float)w*scale;
#elif USE_INT4_WEIGHTS
#if USE_INT4_DP4A_PREPACK
    return (float)((int8_t)w)*scale;
#else
    int q=(int)(w&0x0F);
    q=(q>=8) ? q-16 : q;
    return (float)q*scale;
#endif
#else
    return __half2float(w);
#endif
}
__device__ __forceinline__ float row_scale_at(const float* scales,int row){
#if USE_INT8_WEIGHTS || USE_INT4_WEIGHTS
    return scales ? __ldg(scales+row) : 1.0f;
#else
    (void)scales; (void)row;
    return 1.0f;
#endif
}
__device__ __forceinline__ const WeightT* row_weight_ptr(const WeightT* W,int row,int IN){
#if USE_INT4_WEIGHTS && !USE_INT4_DP4A_PREPACK
    return W+(size_t)row*((IN+1)>>1);
#else
    return W+(size_t)row*IN;
#endif
}
__device__ __forceinline__ int unpack_i4(uint8_t packed,bool high){
    int q=high ? ((packed>>4)&0x0F) : (packed&0x0F);
    return q>=8 ? q-16 : q;
}
__device__ __forceinline__ int unpack_i4_shift(uint32_t packed,int shift){
    int q=(packed>>shift)&0x0F;
    return q>=8 ? q-16 : q;
}
__device__ __forceinline__ int pack_i8x4(int q0,int q1,int q2,int q3){
    uint32_t packed=((uint32_t)((uint8_t)((int8_t)q0))) |
                    ((uint32_t)((uint8_t)((int8_t)q1))<<8) |
                    ((uint32_t)((uint8_t)((int8_t)q2))<<16) |
                    ((uint32_t)((uint8_t)((int8_t)q3))<<24);
    return (int)packed;
}
__device__ __forceinline__ int pack_i4x4_to_i8x4(uint32_t packed,int shift){
    uint32_t x=(packed>>shift)&0xFFFFu;
    uint32_t y=(x&0x000Fu) |
               ((x&0x00F0u)<<4) |
               ((x&0x0F00u)<<8) |
               ((x&0xF000u)<<12);
    y|=(y&0x08080808u)*0x1Eu;
    return (int)y;
}
__device__ __forceinline__ int dp4a_i8(int a,int b,int acc){
#if USE_SOFT_DP4A
    const int8_t a0=(int8_t)(a&0xFF);
    const int8_t a1=(int8_t)((a>>8)&0xFF);
    const int8_t a2=(int8_t)((a>>16)&0xFF);
    const int8_t a3=(int8_t)((a>>24)&0xFF);
    const int8_t b0=(int8_t)(b&0xFF);
    const int8_t b1=(int8_t)((b>>8)&0xFF);
    const int8_t b2=(int8_t)((b>>16)&0xFF);
    const int8_t b3=(int8_t)((b>>24)&0xFF);
    return acc + (int)a0*(int)b0 + (int)a1*(int)b1 + (int)a2*(int)b2 + (int)a3*(int)b3;
#else
    return __dp4a(a,b,acc);
#endif
}
__device__ __forceinline__ float packed_i4_at(const WeightT* row,int i,float scale){
    uint8_t packed=row[i>>1];
    return (float)unpack_i4(packed,(i&1)!=0)*scale;
}
__device__ __forceinline__ int dot_i4_i8_dp4a(const WeightT* row,const int8_t* x,int IN,int tid,int stride){
#if USE_INT4_DP4A_PREPACK
    const int* row4=reinterpret_cast<const int*>(row);
    const int* x4=reinterpret_cast<const int*>(x);
    int packs4=IN>>2;
    int acc=0;
    for(int j=tid;j<packs4;j+=stride){
        acc=dp4a_i8(__ldg(row4+j),__ldg(x4+j),acc);
    }
    for(int i=(packs4<<2)+tid;i<IN;i+=stride){
        acc+=(int)((int8_t)row[i])*(int)x[i];
    }
    return acc;
#else
    const uint32_t* row32=reinterpret_cast<const uint32_t*>(row);
    const int* x4=reinterpret_cast<const int*>(x);
    int packs8=IN>>3;
    int acc=0;
    for(int j=tid;j<packs8;j+=stride){
        uint32_t packed=__ldg(row32+j);
        int i4=j<<1;
        acc=dp4a_i8(pack_i4x4_to_i8x4(packed,0),__ldg(x4+i4),acc);
        acc=dp4a_i8(pack_i4x4_to_i8x4(packed,16),__ldg(x4+i4+1),acc);
    }
    for(int i=(packs8<<3)+tid;i<IN;i+=stride){
        int q=unpack_i4(row[i>>1],(i&1)!=0);
        acc+=q*(int)x[i];
    }
    return acc;
#endif
}
__device__ __forceinline__ void dot_i4_i8_dp4a2(
    const WeightT* row0,const WeightT* row1,
    const int8_t* x,int IN,int tid,int stride,
    int& acc0,int& acc1
){
    acc0=0;
    acc1=0;
#if USE_INT4_DP4A_PREPACK
    const int* row04=reinterpret_cast<const int*>(row0);
    const int* row14=reinterpret_cast<const int*>(row1);
    const int* x4=reinterpret_cast<const int*>(x);
    int packs4=IN>>2;
    for(int j=tid;j<packs4;j+=stride){
        int xv=__ldg(x4+j);
        acc0=dp4a_i8(__ldg(row04+j),xv,acc0);
        if(row1) acc1=dp4a_i8(__ldg(row14+j),xv,acc1);
    }
    for(int i=(packs4<<2)+tid;i<IN;i+=stride){
        int xv=(int)x[i];
        acc0+=(int)((int8_t)row0[i])*xv;
        if(row1) acc1+=(int)((int8_t)row1[i])*xv;
    }
#else
    const uint32_t* row032=reinterpret_cast<const uint32_t*>(row0);
    const uint32_t* row132=reinterpret_cast<const uint32_t*>(row1);
    const int* x4=reinterpret_cast<const int*>(x);
    int packs8=IN>>3;
    for(int j=tid;j<packs8;j+=stride){
        int i4=j<<1;
        int xv0=__ldg(x4+i4);
        int xv1=__ldg(x4+i4+1);
        uint32_t packed0=__ldg(row032+j);
        acc0=dp4a_i8(pack_i4x4_to_i8x4(packed0,0),xv0,acc0);
        acc0=dp4a_i8(pack_i4x4_to_i8x4(packed0,16),xv1,acc0);
        if(row1){
            uint32_t packed1=__ldg(row132+j);
            acc1=dp4a_i8(pack_i4x4_to_i8x4(packed1,0),xv0,acc1);
            acc1=dp4a_i8(pack_i4x4_to_i8x4(packed1,16),xv1,acc1);
        }
    }
    for(int i=(packs8<<3)+tid;i<IN;i+=stride){
        int xv=(int)x[i];
        acc0+=unpack_i4(row0[i>>1],(i&1)!=0)*xv;
        if(row1) acc1+=unpack_i4(row1[i>>1],(i&1)!=0)*xv;
    }
#endif
}
__device__ __forceinline__ void dot_i4_i8_dp4a4(
    const WeightT* row0,const WeightT* row1,const WeightT* row2,const WeightT* row3,
    const int8_t* x,int IN,int tid,int stride,
    int& acc0,int& acc1,int& acc2,int& acc3
){
    acc0=0;
    acc1=0;
    acc2=0;
    acc3=0;
#if USE_INT4_DP4A_PREPACK
    const int* row04=reinterpret_cast<const int*>(row0);
    const int* row14=reinterpret_cast<const int*>(row1);
    const int* row24=reinterpret_cast<const int*>(row2);
    const int* row34=reinterpret_cast<const int*>(row3);
    const int* x4=reinterpret_cast<const int*>(x);
    int packs4=IN>>2;
    for(int j=tid;j<packs4;j+=stride){
        int xv=__ldg(x4+j);
        acc0=dp4a_i8(__ldg(row04+j),xv,acc0);
        if(row1) acc1=dp4a_i8(__ldg(row14+j),xv,acc1);
        if(row2) acc2=dp4a_i8(__ldg(row24+j),xv,acc2);
        if(row3) acc3=dp4a_i8(__ldg(row34+j),xv,acc3);
    }
    for(int i=(packs4<<2)+tid;i<IN;i+=stride){
        int xv=(int)x[i];
        acc0+=(int)((int8_t)row0[i])*xv;
        if(row1) acc1+=(int)((int8_t)row1[i])*xv;
        if(row2) acc2+=(int)((int8_t)row2[i])*xv;
        if(row3) acc3+=(int)((int8_t)row3[i])*xv;
    }
#else
    const uint32_t* row032=reinterpret_cast<const uint32_t*>(row0);
    const uint32_t* row132=reinterpret_cast<const uint32_t*>(row1);
    const uint32_t* row232=reinterpret_cast<const uint32_t*>(row2);
    const uint32_t* row332=reinterpret_cast<const uint32_t*>(row3);
    const int* x4=reinterpret_cast<const int*>(x);
    int packs8=IN>>3;
    for(int j=tid;j<packs8;j+=stride){
        int i4=j<<1;
        int xv0=__ldg(x4+i4);
        int xv1=__ldg(x4+i4+1);
        uint32_t packed0=__ldg(row032+j);
        acc0=dp4a_i8(pack_i4x4_to_i8x4(packed0,0),xv0,acc0);
        acc0=dp4a_i8(pack_i4x4_to_i8x4(packed0,16),xv1,acc0);
        if(row1){
            uint32_t packed1=__ldg(row132+j);
            acc1=dp4a_i8(pack_i4x4_to_i8x4(packed1,0),xv0,acc1);
            acc1=dp4a_i8(pack_i4x4_to_i8x4(packed1,16),xv1,acc1);
        }
        if(row2){
            uint32_t packed2=__ldg(row232+j);
            acc2=dp4a_i8(pack_i4x4_to_i8x4(packed2,0),xv0,acc2);
            acc2=dp4a_i8(pack_i4x4_to_i8x4(packed2,16),xv1,acc2);
        }
        if(row3){
            uint32_t packed3=__ldg(row332+j);
            acc3=dp4a_i8(pack_i4x4_to_i8x4(packed3,0),xv0,acc3);
            acc3=dp4a_i8(pack_i4x4_to_i8x4(packed3,16),xv1,acc3);
        }
    }
    for(int i=(packs8<<3)+tid;i<IN;i+=stride){
        int xv=(int)x[i];
        acc0+=unpack_i4(row0[i>>1],(i&1)!=0)*xv;
        if(row1) acc1+=unpack_i4(row1[i>>1],(i&1)!=0)*xv;
        if(row2) acc2+=unpack_i4(row2[i>>1],(i&1)!=0)*xv;
        if(row3) acc3+=unpack_i4(row3[i>>1],(i&1)!=0)*xv;
    }
#endif
}
__device__ __forceinline__ float dot_weight_float_x(const WeightT* row,const float* x,int IN,int tid,int stride,float scale){
#if USE_INT8_WEIGHTS
    (void)scale;
    const char4* row4=reinterpret_cast<const char4*>(row);
    int packs=IN>>2;
    float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
    for(int j=tid;j<packs;j+=stride){
        char4 q=row4[j];
        int i=j<<2;
        s0=fmaf((float)q.x,__ldg(x+i),s0);
        s1=fmaf((float)q.y,__ldg(x+i+1),s1);
        s2=fmaf((float)q.z,__ldg(x+i+2),s2);
        s3=fmaf((float)q.w,__ldg(x+i+3),s3);
    }
    for(int i=(packs<<2)+tid;i<IN;i+=stride) s0=fmaf((float)row[i],__ldg(x+i),s0);
    return (s0+s1)+(s2+s3);
#elif USE_INT4_WEIGHTS
    (void)scale;
    const uint32_t* row32=reinterpret_cast<const uint32_t*>(row);
    int packed_bytes=(IN+1)>>1;
    int packs32=packed_bytes>>2;
    float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
    for(int j=tid;j<packs32;j+=stride){
        uint32_t packed=__ldg(row32+j);
        int i=j<<3;
        s0=fmaf((float)unpack_i4_shift(packed,0),__ldg(x+i),s0);
        s1=fmaf((float)unpack_i4_shift(packed,4),__ldg(x+i+1),s1);
        s2=fmaf((float)unpack_i4_shift(packed,8),__ldg(x+i+2),s2);
        s3=fmaf((float)unpack_i4_shift(packed,12),__ldg(x+i+3),s3);
        s0=fmaf((float)unpack_i4_shift(packed,16),__ldg(x+i+4),s0);
        s1=fmaf((float)unpack_i4_shift(packed,20),__ldg(x+i+5),s1);
        s2=fmaf((float)unpack_i4_shift(packed,24),__ldg(x+i+6),s2);
        s3=fmaf((float)unpack_i4_shift(packed,28),__ldg(x+i+7),s3);
    }
    for(int j=(packs32<<2)+tid;j<packed_bytes;j+=stride){
        uint8_t packed=row[j];
        int i=j<<1;
        if(i<IN) s0=fmaf((float)unpack_i4(packed,false),__ldg(x+i),s0);
        if(i+1<IN) s1=fmaf((float)unpack_i4(packed,true),__ldg(x+i+1),s1);
    }
    return (s0+s1)+(s2+s3);
#else
    (void)scale;
    const half2* row2=reinterpret_cast<const half2*>(row);
    int pairs=IN>>1;
    float s=0.0f;
    for(int j=tid;j<pairs;j+=stride){
        half2 hw=row2[j];
        float2 wf=__half22float2(hw);
        int i=j<<1;
        s=fmaf(wf.x,__ldg(x+i),s);
        s=fmaf(wf.y,__ldg(x+i+1),s);
    }
    if((IN&1) && tid==0) s=fmaf(weight_to_float(row[IN-1]),__ldg(x+IN-1),s);
    return s;
#endif
}

__global__ void embedding_kernel(int token,const WeightT* emb,const float* scales,float* x){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<HIDDEN){
#if USE_INT4_WEIGHTS && !USE_INT4_DP4A_PREPACK
        x[i]=packed_i4_at(row_weight_ptr(emb,token,HIDDEN),i,row_scale_at(scales,token));
#else
        x[i]=weight_to_float(emb[(size_t)token*HIDDEN+i],row_scale_at(scales,token));
#endif
    }
}
__global__ void rmsnorm_kernel(const float* x,const float* w,float* y,int D,float eps){
    extern __shared__ float sh[];
    int tid=threadIdx.x; float s=0;
    for(int i=tid;i<D;i+=blockDim.x){float v=x[i]; s+=v*v;}
    sh[tid]=s; __syncthreads();
    for(int st=blockDim.x/2;st>0;st>>=1){if(tid<st) sh[tid]+=sh[tid+st]; __syncthreads();}
    float inv=rsqrtf(sh[0]/D+eps);
    for(int i=tid;i<D;i+=blockDim.x) y[i]=x[i]*inv*w[i];
}
__global__ void add_rmsnorm_kernel(float* x,const float* res,const float* w,float* y,int D,float eps){
    extern __shared__ float sh[];
    int tid=threadIdx.x;
    float s=0.0f;
    for(int i=tid;i<D;i+=blockDim.x){
        float v=x[i]+res[i];
        x[i]=v;
        s+=v*v;
    }
    sh[tid]=s;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]+=sh[tid+stride];
        __syncthreads();
    }
    float inv=rsqrtf(sh[0]/D+eps);
    for(int i=tid;i<D;i+=blockDim.x) y[i]=x[i]*inv*w[i];
}
__global__ void linear_kernel(const float* x,const WeightT* W,const float* scales,const float* b,float* y,int IN,int OUT){
    __shared__ float sh[256];
    int o=blockIdx.x;
    int tid=threadIdx.x;
    if(o>=OUT) return;
    const WeightT* row=row_weight_ptr(W,o,IN);
    float weight_scale=row_scale_at(scales,o);
    float s=dot_weight_float_x(row,x,IN,tid,blockDim.x,weight_scale);
    sh[tid]=s;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]+=sh[tid+stride];
        __syncthreads();
    }
    if(tid==0){
        float out=sh[0]*weight_scale;
        y[o]=b ? out+b[o] : out;
    }
}
__global__ void qkv_linear_kernel(
    const float* x,
    const WeightT* Wq,const WeightT* Wk,const WeightT* Wv,
    const float* Sq,const float* Sk,const float* Sv,
    const float* bq,const float* bk,const float* bv,
    float* q,float* k,float* v,
    int IN
){
    __shared__ float sh[256];
    int o=blockIdx.x;
    int tid=threadIdx.x;
    const WeightT* W=nullptr;
    const float* S=nullptr;
    const float* b=nullptr;
    float* y=nullptr;
    int local_o=o;
    if(o<HIDDEN){
        W=Wq; S=Sq; b=bq; y=q;
    }else if(o<HIDDEN+KV_DIM){
        local_o=o-HIDDEN; W=Wk; S=Sk; b=bk; y=k;
    }else{
        local_o=o-HIDDEN-KV_DIM; W=Wv; S=Sv; b=bv; y=v;
    }
    const WeightT* row=row_weight_ptr(W,local_o,IN);
    float weight_scale=row_scale_at(S,local_o);
    float s=dot_weight_float_x(row,x,IN,tid,blockDim.x,weight_scale);
    sh[tid]=s;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]+=sh[tid+stride];
        __syncthreads();
    }
    if(tid==0){
        float out=sh[0]*weight_scale;
        y[local_o]=b ? out+b[local_o] : out;
    }
}
__global__ void gate_up_linear_kernel(
    const float* x,
    const WeightT* Wgate,const WeightT* Wup,
    const float* Sgate,const float* Sup,
    float* gate,float* up,
    int IN
){
    __shared__ float sh[256];
    int o=blockIdx.x;
    int tid=threadIdx.x;
    bool is_gate=o<INTERMEDIATE;
    int local_o=is_gate ? o : o-INTERMEDIATE;
    const WeightT* W=is_gate ? Wgate : Wup;
    const float* S=is_gate ? Sgate : Sup;
    float* y=is_gate ? gate : up;
    const WeightT* row=row_weight_ptr(W,local_o,IN);
    float weight_scale=row_scale_at(S,local_o);
    float s=dot_weight_float_x(row,x,IN,tid,blockDim.x,weight_scale);
    sh[tid]=s;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]+=sh[tid+stride];
        __syncthreads();
    }
    if(tid==0) y[local_o]=sh[0]*weight_scale;
}
__global__ void quantize_int8_kernel(const float* x,int8_t* q,float* scale,int n){
    __shared__ float sh[256];
    int tid=threadIdx.x;
    float m=0.0f;
    for(int i=tid;i<n;i+=blockDim.x) m=fmaxf(m,fabsf(x[i]));
    sh[tid]=m;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]=fmaxf(sh[tid],sh[tid+stride]);
        __syncthreads();
    }
    float max_abs=sh[0];
    float inv=max_abs>0.0f ? 127.0f/max_abs : 0.0f;
    if(tid==0) scale[0]=max_abs>0.0f ? max_abs/127.0f : 1.0f;
    __syncthreads();
    for(int i=tid;i<n;i+=blockDim.x){
        int v=max_abs>0.0f ? (int)lrintf(x[i]*inv) : 0;
        v=v<-127 ? -127 : (v>127 ? 127 : v);
        q[i]=(int8_t)v;
    }
}
__global__ void linear_i4_dp4a_kernel(const int8_t* xq,const float* x_scale,const WeightT* W,const float* scales,const float* b,float* y,int IN,int OUT){
    __shared__ int sh[256];
    int o=blockIdx.x;
    int tid=threadIdx.x;
    if(o>=OUT) return;
    const WeightT* row=row_weight_ptr(W,o,IN);
    int s=dot_i4_i8_dp4a(row,xq,IN,tid,blockDim.x);
    sh[tid]=s;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]+=sh[tid+stride];
        __syncthreads();
    }
    if(tid==0){
        float out=(float)sh[0]*row_scale_at(scales,o)*x_scale[0];
        y[o]=b ? out+b[o] : out;
    }
}
__global__ void linear_i4_dp4a2_kernel(const int8_t* xq,const float* x_scale,const WeightT* W,const float* scales,const float* b,float* y,int IN,int OUT){
    __shared__ int sh0[256];
    __shared__ int sh1[256];
    int o0=blockIdx.x<<1;
    int o1=o0+1;
    int tid=threadIdx.x;
    if(o0>=OUT) return;
    const WeightT* row0=row_weight_ptr(W,o0,IN);
    const WeightT* row1=(o1<OUT) ? row_weight_ptr(W,o1,IN) : nullptr;
    int s0=0,s1=0;
    dot_i4_i8_dp4a2(row0,row1,xq,IN,tid,blockDim.x,s0,s1);
    sh0[tid]=s0;
    sh1[tid]=s1;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            sh0[tid]+=sh0[tid+stride];
            sh1[tid]+=sh1[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0){
        float xs=x_scale[0];
        float out0=(float)sh0[0]*row_scale_at(scales,o0)*xs;
        y[o0]=b ? out0+b[o0] : out0;
        if(o1<OUT){
            float out1=(float)sh1[0]*row_scale_at(scales,o1)*xs;
            y[o1]=b ? out1+b[o1] : out1;
        }
    }
}
__global__ void linear_i4_dp4a4_kernel(const int8_t* xq,const float* x_scale,const WeightT* W,const float* scales,const float* b,float* y,int IN,int OUT){
    __shared__ int sh0[256];
    __shared__ int sh1[256];
    __shared__ int sh2[256];
    __shared__ int sh3[256];
    int o0=blockIdx.x<<2;
    int o1=o0+1;
    int o2=o0+2;
    int o3=o0+3;
    int tid=threadIdx.x;
    if(o0>=OUT) return;
    const WeightT* row0=row_weight_ptr(W,o0,IN);
    const WeightT* row1=(o1<OUT) ? row_weight_ptr(W,o1,IN) : nullptr;
    const WeightT* row2=(o2<OUT) ? row_weight_ptr(W,o2,IN) : nullptr;
    const WeightT* row3=(o3<OUT) ? row_weight_ptr(W,o3,IN) : nullptr;
    int s0=0,s1=0,s2=0,s3=0;
    dot_i4_i8_dp4a4(row0,row1,row2,row3,xq,IN,tid,blockDim.x,s0,s1,s2,s3);
    sh0[tid]=s0;
    sh1[tid]=s1;
    sh2[tid]=s2;
    sh3[tid]=s3;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            sh0[tid]+=sh0[tid+stride];
            sh1[tid]+=sh1[tid+stride];
            sh2[tid]+=sh2[tid+stride];
            sh3[tid]+=sh3[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0){
        float xs=x_scale[0];
        float out0=(float)sh0[0]*row_scale_at(scales,o0)*xs;
        y[o0]=b ? out0+b[o0] : out0;
        if(o1<OUT){
            float out1=(float)sh1[0]*row_scale_at(scales,o1)*xs;
            y[o1]=b ? out1+b[o1] : out1;
        }
        if(o2<OUT){
            float out2=(float)sh2[0]*row_scale_at(scales,o2)*xs;
            y[o2]=b ? out2+b[o2] : out2;
        }
        if(o3<OUT){
            float out3=(float)sh3[0]*row_scale_at(scales,o3)*xs;
            y[o3]=b ? out3+b[o3] : out3;
        }
    }
}
__global__ void qkv_i4_dp4a_kernel(
    const int8_t* xq,const float* x_scale,
    const WeightT* Wq,const WeightT* Wk,const WeightT* Wv,
    const float* Sq,const float* Sk,const float* Sv,
    const float* bq,const float* bk,const float* bv,
    float* q,float* k,float* v,
    int IN
){
    __shared__ int sh[256];
    int o=blockIdx.x;
    int tid=threadIdx.x;
    const WeightT* W=nullptr;
    const float* S=nullptr;
    const float* b=nullptr;
    float* y=nullptr;
    int local_o=o;
    if(o<HIDDEN){
        W=Wq; S=Sq; b=bq; y=q;
    }else if(o<HIDDEN+KV_DIM){
        local_o=o-HIDDEN; W=Wk; S=Sk; b=bk; y=k;
    }else{
        local_o=o-HIDDEN-KV_DIM; W=Wv; S=Sv; b=bv; y=v;
    }
    const WeightT* row=row_weight_ptr(W,local_o,IN);
    int s=dot_i4_i8_dp4a(row,xq,IN,tid,blockDim.x);
    sh[tid]=s;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]+=sh[tid+stride];
        __syncthreads();
    }
    if(tid==0){
        float out=(float)sh[0]*row_scale_at(S,local_o)*x_scale[0];
        y[local_o]=b ? out+b[local_o] : out;
    }
}
__global__ void qkv_i4_dp4a2_kernel(
    const int8_t* xq,const float* x_scale,
    const WeightT* Wq,const WeightT* Wk,const WeightT* Wv,
    const float* Sq,const float* Sk,const float* Sv,
    const float* bq,const float* bk,const float* bv,
    float* q,float* k,float* v,
    int IN
){
    __shared__ int sh0[256];
    __shared__ int sh1[256];
    constexpr int TOTAL_QKV=HIDDEN+2*KV_DIM;
    int o0=blockIdx.x<<1;
    int o1=o0+1;
    int tid=threadIdx.x;
    if(o0>=TOTAL_QKV) return;
    const WeightT *W0=nullptr,*W1=nullptr;
    const float *S0=nullptr,*S1=nullptr,*b0=nullptr,*b1=nullptr;
    float *y0=nullptr,*y1=nullptr;
    int local0=o0,local1=o1;
    if(o0<HIDDEN){
        W0=Wq; S0=Sq; b0=bq; y0=q;
    }else if(o0<HIDDEN+KV_DIM){
        local0=o0-HIDDEN; W0=Wk; S0=Sk; b0=bk; y0=k;
    }else{
        local0=o0-HIDDEN-KV_DIM; W0=Wv; S0=Sv; b0=bv; y0=v;
    }
    if(o1<TOTAL_QKV){
        if(o1<HIDDEN){
            W1=Wq; S1=Sq; b1=bq; y1=q;
        }else if(o1<HIDDEN+KV_DIM){
            local1=o1-HIDDEN; W1=Wk; S1=Sk; b1=bk; y1=k;
        }else{
            local1=o1-HIDDEN-KV_DIM; W1=Wv; S1=Sv; b1=bv; y1=v;
        }
    }
    const WeightT* row0=row_weight_ptr(W0,local0,IN);
    const WeightT* row1=(o1<TOTAL_QKV) ? row_weight_ptr(W1,local1,IN) : nullptr;
    int s0=0,s1=0;
    dot_i4_i8_dp4a2(row0,row1,xq,IN,tid,blockDim.x,s0,s1);
    sh0[tid]=s0;
    sh1[tid]=s1;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            sh0[tid]+=sh0[tid+stride];
            sh1[tid]+=sh1[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0){
        float xs=x_scale[0];
        float out0=(float)sh0[0]*row_scale_at(S0,local0)*xs;
        y0[local0]=b0 ? out0+b0[local0] : out0;
        if(o1<TOTAL_QKV){
            float out1=(float)sh1[0]*row_scale_at(S1,local1)*xs;
            y1[local1]=b1 ? out1+b1[local1] : out1;
        }
    }
}
__device__ __forceinline__ void select_qkv_row(
    int o,
    const WeightT* Wq,const WeightT* Wk,const WeightT* Wv,
    const float* Sq,const float* Sk,const float* Sv,
    const float* bq,const float* bk,const float* bv,
    float* q,float* k,float* v,
    const WeightT*& W,const float*& S,const float*& b,float*& y,int& local_o
){
    local_o=o;
    if(o<HIDDEN){
        W=Wq; S=Sq; b=bq; y=q;
    }else if(o<HIDDEN+KV_DIM){
        local_o=o-HIDDEN; W=Wk; S=Sk; b=bk; y=k;
    }else{
        local_o=o-HIDDEN-KV_DIM; W=Wv; S=Sv; b=bv; y=v;
    }
}
__global__ void qkv_i4_dp4a4_kernel(
    const int8_t* xq,const float* x_scale,
    const WeightT* Wq,const WeightT* Wk,const WeightT* Wv,
    const float* Sq,const float* Sk,const float* Sv,
    const float* bq,const float* bk,const float* bv,
    float* q,float* k,float* v,
    int IN
){
    __shared__ int sh0[256];
    __shared__ int sh1[256];
    __shared__ int sh2[256];
    __shared__ int sh3[256];
    constexpr int TOTAL_QKV=HIDDEN+2*KV_DIM;
    int o0=blockIdx.x<<2;
    int o1=o0+1;
    int o2=o0+2;
    int o3=o0+3;
    int tid=threadIdx.x;
    if(o0>=TOTAL_QKV) return;
    const WeightT *W0=nullptr,*W1=nullptr,*W2=nullptr,*W3=nullptr;
    const float *S0=nullptr,*S1=nullptr,*S2=nullptr,*S3=nullptr;
    const float *b0=nullptr,*b1=nullptr,*b2=nullptr,*b3=nullptr;
    float *y0=nullptr,*y1=nullptr,*y2=nullptr,*y3=nullptr;
    int local0=0,local1=0,local2=0,local3=0;
    select_qkv_row(o0,Wq,Wk,Wv,Sq,Sk,Sv,bq,bk,bv,q,k,v,W0,S0,b0,y0,local0);
    if(o1<TOTAL_QKV) select_qkv_row(o1,Wq,Wk,Wv,Sq,Sk,Sv,bq,bk,bv,q,k,v,W1,S1,b1,y1,local1);
    if(o2<TOTAL_QKV) select_qkv_row(o2,Wq,Wk,Wv,Sq,Sk,Sv,bq,bk,bv,q,k,v,W2,S2,b2,y2,local2);
    if(o3<TOTAL_QKV) select_qkv_row(o3,Wq,Wk,Wv,Sq,Sk,Sv,bq,bk,bv,q,k,v,W3,S3,b3,y3,local3);
    const WeightT* row0=row_weight_ptr(W0,local0,IN);
    const WeightT* row1=(o1<TOTAL_QKV) ? row_weight_ptr(W1,local1,IN) : nullptr;
    const WeightT* row2=(o2<TOTAL_QKV) ? row_weight_ptr(W2,local2,IN) : nullptr;
    const WeightT* row3=(o3<TOTAL_QKV) ? row_weight_ptr(W3,local3,IN) : nullptr;
    int s0=0,s1=0,s2=0,s3=0;
    dot_i4_i8_dp4a4(row0,row1,row2,row3,xq,IN,tid,blockDim.x,s0,s1,s2,s3);
    sh0[tid]=s0;
    sh1[tid]=s1;
    sh2[tid]=s2;
    sh3[tid]=s3;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            sh0[tid]+=sh0[tid+stride];
            sh1[tid]+=sh1[tid+stride];
            sh2[tid]+=sh2[tid+stride];
            sh3[tid]+=sh3[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0){
        float xs=x_scale[0];
        float out0=(float)sh0[0]*row_scale_at(S0,local0)*xs;
        y0[local0]=b0 ? out0+b0[local0] : out0;
        if(o1<TOTAL_QKV){
            float out1=(float)sh1[0]*row_scale_at(S1,local1)*xs;
            y1[local1]=b1 ? out1+b1[local1] : out1;
        }
        if(o2<TOTAL_QKV){
            float out2=(float)sh2[0]*row_scale_at(S2,local2)*xs;
            y2[local2]=b2 ? out2+b2[local2] : out2;
        }
        if(o3<TOTAL_QKV){
            float out3=(float)sh3[0]*row_scale_at(S3,local3)*xs;
            y3[local3]=b3 ? out3+b3[local3] : out3;
        }
    }
}
__global__ void gate_up_i4_dp4a_kernel(
    const int8_t* xq,const float* x_scale,
    const WeightT* Wgate,const WeightT* Wup,
    const float* Sgate,const float* Sup,
    float* gate,float* up,
    int IN
){
    __shared__ int sh[256];
    int o=blockIdx.x;
    int tid=threadIdx.x;
    bool is_gate=o<INTERMEDIATE;
    int local_o=is_gate ? o : o-INTERMEDIATE;
    const WeightT* W=is_gate ? Wgate : Wup;
    const float* S=is_gate ? Sgate : Sup;
    float* y=is_gate ? gate : up;
    const WeightT* row=row_weight_ptr(W,local_o,IN);
    int s=dot_i4_i8_dp4a(row,xq,IN,tid,blockDim.x);
    sh[tid]=s;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]+=sh[tid+stride];
        __syncthreads();
    }
    if(tid==0) y[local_o]=(float)sh[0]*row_scale_at(S,local_o)*x_scale[0];
}
__global__ void gate_up_i4_dp4a2_kernel(
    const int8_t* xq,const float* x_scale,
    const WeightT* Wgate,const WeightT* Wup,
    const float* Sgate,const float* Sup,
    float* gate,float* up,
    int IN
){
    __shared__ int sh0[256];
    __shared__ int sh1[256];
    constexpr int TOTAL_GATE_UP=2*INTERMEDIATE;
    int o0=blockIdx.x<<1;
    int o1=o0+1;
    int tid=threadIdx.x;
    if(o0>=TOTAL_GATE_UP) return;
    bool is_gate0=o0<INTERMEDIATE;
    bool is_gate1=o1<INTERMEDIATE;
    int local0=is_gate0 ? o0 : o0-INTERMEDIATE;
    int local1=is_gate1 ? o1 : o1-INTERMEDIATE;
    const WeightT* W0=is_gate0 ? Wgate : Wup;
    const WeightT* W1=(o1<TOTAL_GATE_UP) ? (is_gate1 ? Wgate : Wup) : nullptr;
    const float* S0=is_gate0 ? Sgate : Sup;
    const float* S1=is_gate1 ? Sgate : Sup;
    float* y0=is_gate0 ? gate : up;
    float* y1=is_gate1 ? gate : up;
    const WeightT* row0=row_weight_ptr(W0,local0,IN);
    const WeightT* row1=(o1<TOTAL_GATE_UP) ? row_weight_ptr(W1,local1,IN) : nullptr;
    int s0=0,s1=0;
    dot_i4_i8_dp4a2(row0,row1,xq,IN,tid,blockDim.x,s0,s1);
    sh0[tid]=s0;
    sh1[tid]=s1;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            sh0[tid]+=sh0[tid+stride];
            sh1[tid]+=sh1[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0){
        float xs=x_scale[0];
        y0[local0]=(float)sh0[0]*row_scale_at(S0,local0)*xs;
        if(o1<TOTAL_GATE_UP){
            y1[local1]=(float)sh1[0]*row_scale_at(S1,local1)*xs;
        }
    }
}
__global__ void gate_up_i4_dp4a4_kernel(
    const int8_t* xq,const float* x_scale,
    const WeightT* Wgate,const WeightT* Wup,
    const float* Sgate,const float* Sup,
    float* gate,float* up,
    int IN
){
    __shared__ int sh0[256];
    __shared__ int sh1[256];
    __shared__ int sh2[256];
    __shared__ int sh3[256];
    constexpr int TOTAL_GATE_UP=2*INTERMEDIATE;
    int o0=blockIdx.x<<2;
    int o1=o0+1;
    int o2=o0+2;
    int o3=o0+3;
    int tid=threadIdx.x;
    if(o0>=TOTAL_GATE_UP) return;
    bool is_gate0=o0<INTERMEDIATE;
    bool is_gate1=o1<INTERMEDIATE;
    bool is_gate2=o2<INTERMEDIATE;
    bool is_gate3=o3<INTERMEDIATE;
    int local0=is_gate0 ? o0 : o0-INTERMEDIATE;
    int local1=is_gate1 ? o1 : o1-INTERMEDIATE;
    int local2=is_gate2 ? o2 : o2-INTERMEDIATE;
    int local3=is_gate3 ? o3 : o3-INTERMEDIATE;
    const WeightT* W0=is_gate0 ? Wgate : Wup;
    const WeightT* W1=(o1<TOTAL_GATE_UP) ? (is_gate1 ? Wgate : Wup) : nullptr;
    const WeightT* W2=(o2<TOTAL_GATE_UP) ? (is_gate2 ? Wgate : Wup) : nullptr;
    const WeightT* W3=(o3<TOTAL_GATE_UP) ? (is_gate3 ? Wgate : Wup) : nullptr;
    const float* S0=is_gate0 ? Sgate : Sup;
    const float* S1=is_gate1 ? Sgate : Sup;
    const float* S2=is_gate2 ? Sgate : Sup;
    const float* S3=is_gate3 ? Sgate : Sup;
    float* y0=is_gate0 ? gate : up;
    float* y1=is_gate1 ? gate : up;
    float* y2=is_gate2 ? gate : up;
    float* y3=is_gate3 ? gate : up;
    const WeightT* row0=row_weight_ptr(W0,local0,IN);
    const WeightT* row1=(o1<TOTAL_GATE_UP) ? row_weight_ptr(W1,local1,IN) : nullptr;
    const WeightT* row2=(o2<TOTAL_GATE_UP) ? row_weight_ptr(W2,local2,IN) : nullptr;
    const WeightT* row3=(o3<TOTAL_GATE_UP) ? row_weight_ptr(W3,local3,IN) : nullptr;
    int s0=0,s1=0,s2=0,s3=0;
    dot_i4_i8_dp4a4(row0,row1,row2,row3,xq,IN,tid,blockDim.x,s0,s1,s2,s3);
    sh0[tid]=s0;
    sh1[tid]=s1;
    sh2[tid]=s2;
    sh3[tid]=s3;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            sh0[tid]+=sh0[tid+stride];
            sh1[tid]+=sh1[tid+stride];
            sh2[tid]+=sh2[tid+stride];
            sh3[tid]+=sh3[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0){
        float xs=x_scale[0];
        y0[local0]=(float)sh0[0]*row_scale_at(S0,local0)*xs;
        if(o1<TOTAL_GATE_UP) y1[local1]=(float)sh1[0]*row_scale_at(S1,local1)*xs;
        if(o2<TOTAL_GATE_UP) y2[local2]=(float)sh2[0]*row_scale_at(S2,local2)*xs;
        if(o3<TOTAL_GATE_UP) y3[local3]=(float)sh3[0]*row_scale_at(S3,local3)*xs;
    }
}
__global__ void float_to_half_kernel(const float* x,half* xh,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) xh[i]=__float2half_rn(x[i]);
}
__global__ void wmma_linear_kernel(const half* xh,const WeightT* W,const float* b,float* y,int IN,int OUT){
#if !USE_MUSA && !USE_INT8_WEIGHTS && !USE_INT4_WEIGHTS && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    using namespace nvcuda;
    __shared__ half a_tile[WMMA_TILE*WMMA_TILE];
    __shared__ half b_tile[WMMA_TILE*WMMA_TILE];
    __shared__ float c_tile[WMMA_TILE*WMMA_TILE];
    const int out0=blockIdx.x*WMMA_TILE;
    const int tid=threadIdx.x;
    wmma::fragment<wmma::matrix_a,WMMA_TILE,WMMA_TILE,WMMA_TILE,half,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,WMMA_TILE,WMMA_TILE,WMMA_TILE,half,wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator,WMMA_TILE,WMMA_TILE,WMMA_TILE,float> acc_frag;
    wmma::fill_fragment(acc_frag,0.0f);
    for(int k0=0;k0<IN;k0+=WMMA_TILE){
        for(int idx=tid;idx<WMMA_TILE*WMMA_TILE;idx+=blockDim.x){
            int r=idx/WMMA_TILE;
            int c=idx%WMMA_TILE;
            int o=out0+r;
            int k=k0+c;
            a_tile[idx]=(o<OUT && k<IN) ? W[(size_t)o*IN+k] : __float2half_rn(0.0f);
        }
        for(int idx=tid;idx<WMMA_TILE*WMMA_TILE;idx+=blockDim.x){
            int k=idx/WMMA_TILE;
            int kk=k0+k;
            b_tile[idx]=(kk<IN) ? xh[kk] : __float2half_rn(0.0f);
        }
        __syncthreads();
        wmma::load_matrix_sync(a_frag,a_tile,WMMA_TILE);
        wmma::load_matrix_sync(b_frag,b_tile,WMMA_TILE);
        wmma::mma_sync(acc_frag,a_frag,b_frag,acc_frag);
        __syncthreads();
    }
    wmma::store_matrix_sync(c_tile,acc_frag,WMMA_TILE,wmma::mem_row_major);
    __syncthreads();
    for(int r=tid;r<WMMA_TILE;r+=blockDim.x){
        int o=out0+r;
        if(o<OUT) y[o]=c_tile[r*WMMA_TILE]+(b ? b[o] : 0.0f);
    }
#else
    (void)xh;(void)W;(void)b;(void)y;(void)IN;(void)OUT;
#endif
}
__global__ void add_kernel(float* x,const float* y,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=y[i];
}
__device__ __forceinline__ float silu_value(float x){
#if USE_FAST_SILU
    return x/(1.f+__expf(-x));
#else
    return x/(1.f+expf(-x));
#endif
}
__global__ void silu_mul_kernel(const float* g,const float* u,float* o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){float x=g[i]; o[i]=silu_value(x)*u[i];}
}
__global__ void silu_quantize_int8_kernel(const float* g,const float* u,int8_t* q,float* scale,int n){
    __shared__ float sh[256];
    int tid=threadIdx.x;
    float m=0.0f;
    for(int i=tid;i<n;i+=blockDim.x){
        float x=g[i];
        float v=silu_value(x)*u[i];
        m=fmaxf(m,fabsf(v));
    }
    sh[tid]=m;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]=fmaxf(sh[tid],sh[tid+stride]);
        __syncthreads();
    }
    float max_abs=sh[0];
    float inv=max_abs>0.0f ? 127.0f/max_abs : 0.0f;
    if(tid==0) scale[0]=max_abs>0.0f ? max_abs/127.0f : 1.0f;
    __syncthreads();
    for(int i=tid;i<n;i+=blockDim.x){
        float x=g[i];
        float y=silu_value(x)*u[i];
        int v=max_abs>0.0f ? (int)lrintf(y*inv) : 0;
        v=v<-127 ? -127 : (v>127 ? 127 : v);
        q[i]=(int8_t)v;
    }
}
__global__ void rope_kernel(float* x,int heads,int pos,float rope_theta){
    int pair=blockIdx.x*blockDim.x+threadIdx.x;
    int total=heads*(HEAD_DIM/2); if(pair>=total) return;
    int h=pair/(HEAD_DIM/2), p=pair%(HEAD_DIM/2);
    int d0=p,d1=p+(HEAD_DIM/2),base=h*HEAD_DIM;
    float inv=powf(rope_theta,-(float)(2*p)/HEAD_DIM);
    float a=pos*inv,c=cosf(a),s=sinf(a);
    float v0=x[base+d0],v1=x[base+d1];
    x[base+d0]=v0*c-v1*s; x[base+d1]=v0*s+v1*c;
}
__global__ void store_kv_kernel(float* cache,const float* x,int pos,int dim){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<dim) cache[(size_t)pos*dim+i]=x[i];
}
__global__ void rope_store_kv_kernel(float* q,float* k,const float* v,KvCacheT* kc,KvCacheT* vc,const float* rope_cos,const float* rope_sin,int pos){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int q_pairs=N_HEADS*(HEAD_DIM/2);
    int k_pairs=N_KV_HEADS*(HEAD_DIM/2);
    if(i<q_pairs){
        int h=i/(HEAD_DIM/2);
        int p=i%(HEAD_DIM/2);
        int d0=p;
        int d1=p+(HEAD_DIM/2);
        int base=h*HEAD_DIM;
        int rp=pos*(HEAD_DIM/2)+p;
        float c=__ldg(rope_cos+rp);
        float s=__ldg(rope_sin+rp);
        float v0=q[base+d0],v1=q[base+d1];
        q[base+d0]=v0*c-v1*s;
        q[base+d1]=v0*s+v1*c;
    }
    if(i<k_pairs){
        int h=i/(HEAD_DIM/2);
        int p=i%(HEAD_DIM/2);
        int d0=p;
        int d1=p+(HEAD_DIM/2);
        int base=h*HEAD_DIM;
        int rp=pos*(HEAD_DIM/2)+p;
        float c=__ldg(rope_cos+rp);
        float s=__ldg(rope_sin+rp);
        float v0=k[base+d0],v1=k[base+d1];
        float r0=v0*c-v1*s;
        float r1=v0*s+v1*c;
        kc[(size_t)pos*KV_DIM+base+d0]=kv_cache_store(r0);
        kc[(size_t)pos*KV_DIM+base+d1]=kv_cache_store(r1);
    }
    if(i<KV_DIM) vc[(size_t)pos*KV_DIM+i]=kv_cache_store(v[i]);
}
__global__ void attention_scores_kernel(const float* q,const KvCacheT* kc,float* scores,int pos,int max_seq){
    int h=blockIdx.x;
    int tid=threadIdx.x;
    int group=N_HEADS/N_KV_HEADS, kh=h/group;
    const float scale=rsqrtf((float)HEAD_DIM);
    for(int t=tid;t<=pos;t+=blockDim.x){
        float dot=0.0f;
        const float* qh=q+h*HEAD_DIM;
        const KvCacheT* row=kc+(size_t)t*KV_DIM+kh*HEAD_DIM;
        for(int r=0;r<HEAD_DIM;r++) dot+=qh[r]*kv_cache_load(row[r]);
        scores[(size_t)h*max_seq+t]=dot*scale;
    }
}
__global__ void attention_softmax_kernel(float* scores,int pos,int max_seq){
    __shared__ float sh[256];
    int h=blockIdx.x;
    int tid=threadIdx.x;
    float mx=-INFINITY;
    for(int t=tid;t<=pos;t+=blockDim.x){
        mx=fmaxf(mx,scores[(size_t)h*max_seq+t]);
    }
    sh[tid]=mx;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]=fmaxf(sh[tid],sh[tid+stride]);
        __syncthreads();
    }
    mx=sh[0];
    float den=0.0f;
    for(int t=tid;t<=pos;t+=blockDim.x){
        float e=expf(scores[(size_t)h*max_seq+t]-mx);
        scores[(size_t)h*max_seq+t]=e;
        den+=e;
    }
    sh[tid]=den;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]+=sh[tid+stride];
        __syncthreads();
    }
    den=sh[0];
    for(int t=tid;t<=pos;t+=blockDim.x){
        scores[(size_t)h*max_seq+t]/=den;
    }
}
__global__ void attention_apply_kernel(const float* probs,const KvCacheT* vc,float* ctx,int pos,int max_seq){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=HIDDEN) return;
    int d=idx%HEAD_DIM, h=idx/HEAD_DIM;
    int group=N_HEADS/N_KV_HEADS, kh=h/group;
    float out=0.0f;
    for(int t=0;t<=pos;t++){
        float p=probs[(size_t)h*max_seq+t];
        out+=p*kv_cache_load(vc[(size_t)t*KV_DIM+kh*HEAD_DIM+d]);
    }
    ctx[idx]=out;
}
__global__ void attention_fused_kernel(const float* q,const KvCacheT* kc,const KvCacheT* vc,float* scores,float* ctx,int pos,int max_seq){
#if USE_ATTENTION_SHM
    (void)scores;
    extern __shared__ float smem[];
    float* sh=smem;
    float* qsh=sh+blockDim.x;
    float* score_sh=qsh+HEAD_DIM;
#else
    __shared__ float sh[256];
    __shared__ float qsh[HEAD_DIM];
#endif
    int h=blockIdx.x;
    int tid=threadIdx.x;
    int group=N_HEADS/N_KV_HEADS;
    int kh=h/group;
    const float* qh=q+h*HEAD_DIM;
    if(tid<HEAD_DIM) qsh[tid]=qh[tid];
    __syncthreads();
    const float scale=rsqrtf((float)HEAD_DIM);
    float mx=-INFINITY;
    for(int t=tid;t<=pos;t+=blockDim.x){
        const KvCacheT* row=kc+(size_t)t*KV_DIM+kh*HEAD_DIM;
        float dot=0.0f;
#pragma unroll
        for(int r=0;r<HEAD_DIM;r++) dot=fmaf(qsh[r],kv_cache_load(row[r]),dot);
        float s=dot*scale;
#if USE_ATTENTION_SHM
        score_sh[t]=s;
#else
        scores[(size_t)h*max_seq+t]=s;
#endif
        mx=fmaxf(mx,s);
    }
    sh[tid]=mx;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]=fmaxf(sh[tid],sh[tid+stride]);
        __syncthreads();
    }
    mx=sh[0];
    float den=0.0f;
    for(int t=tid;t<=pos;t+=blockDim.x){
#if USE_ATTENTION_SHM
        float e=expf(score_sh[t]-mx);
        score_sh[t]=e;
#else
        float e=expf(scores[(size_t)h*max_seq+t]-mx);
        scores[(size_t)h*max_seq+t]=e;
#endif
        den+=e;
    }
    sh[tid]=den;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]+=sh[tid+stride];
        __syncthreads();
    }
    float inv_den=1.0f/sh[0];
#if USE_ATTENTION_SHM
    __syncthreads();
#endif
    for(int d=tid;d<HEAD_DIM;d+=blockDim.x){
        float out=0.0f;
        for(int t=0;t<=pos;t++){
#if USE_ATTENTION_SHM
            float p=score_sh[t]*inv_den;
#else
            float p=scores[(size_t)h*max_seq+t]*inv_den;
#endif
            out=fmaf(p,kv_cache_load(vc[(size_t)t*KV_DIM+kh*HEAD_DIM+d]),out);
        }
        ctx[h*HEAD_DIM+d]=out;
    }
}
__global__ void attention_rope_fused_kernel(
    const float* q,const float* k,const float* v,
    KvCacheT* kc,KvCacheT* vc,float* scores,float* ctx,
    const float* rope_cos,const float* rope_sin,
    int pos,int max_seq
){
    __shared__ float sh[256];
    __shared__ float qsh[HEAD_DIM];
    int h=blockIdx.x;
    int tid=threadIdx.x;
    int group=N_HEADS/N_KV_HEADS;
    int kh=h/group;
    int rp_base=pos*(HEAD_DIM/2);
    const float* qh=q+h*HEAD_DIM;
    const float* kk=k+kh*HEAD_DIM;
    const float* vv=v+kh*HEAD_DIM;
    KvCacheT* kc_row=kc+(size_t)pos*KV_DIM+kh*HEAD_DIM;
    KvCacheT* vc_row=vc+(size_t)pos*KV_DIM+kh*HEAD_DIM;
    if(tid<HEAD_DIM/2){
        int d0=tid;
        int d1=tid+(HEAD_DIM/2);
        float c=__ldg(rope_cos+rp_base+tid);
        float s=__ldg(rope_sin+rp_base+tid);
        float q0=qh[d0],q1=qh[d1];
        float k0=kk[d0],k1=kk[d1];
        qsh[d0]=q0*c-q1*s;
        qsh[d1]=q0*s+q1*c;
        kc_row[d0]=kv_cache_store(k0*c-k1*s);
        kc_row[d1]=kv_cache_store(k0*s+k1*c);
    }
    if(tid<HEAD_DIM) vc_row[tid]=kv_cache_store(vv[tid]);
    __syncthreads();
    const float scale=rsqrtf((float)HEAD_DIM);
    float mx=-INFINITY;
    for(int t=tid;t<=pos;t+=blockDim.x){
        const KvCacheT* row=kc+(size_t)t*KV_DIM+kh*HEAD_DIM;
        float dot=0.0f;
#pragma unroll
        for(int r=0;r<HEAD_DIM;r++) dot=fmaf(qsh[r],kv_cache_load(row[r]),dot);
        float s=dot*scale;
        scores[(size_t)h*max_seq+t]=s;
        mx=fmaxf(mx,s);
    }
    sh[tid]=mx;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]=fmaxf(sh[tid],sh[tid+stride]);
        __syncthreads();
    }
    mx=sh[0];
    float den=0.0f;
    for(int t=tid;t<=pos;t+=blockDim.x){
        float e=expf(scores[(size_t)h*max_seq+t]-mx);
        scores[(size_t)h*max_seq+t]=e;
        den+=e;
    }
    sh[tid]=den;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]+=sh[tid+stride];
        __syncthreads();
    }
    float inv_den=1.0f/sh[0];
    for(int d=tid;d<HEAD_DIM;d+=blockDim.x){
        float out=0.0f;
        for(int t=0;t<=pos;t++){
            float p=scores[(size_t)h*max_seq+t]*inv_den;
            out=fmaf(p,kv_cache_load(vc[(size_t)t*KV_DIM+kh*HEAD_DIM+d]),out);
        }
        ctx[h*HEAD_DIM+d]=out;
    }
}
__global__ void mark_token_seen_kernel(unsigned char* seen,int token){
    if(token>=0 && token<VOCAB_SIZE) seen[token]=1;
}
__global__ void argmax_with_penalty_kernel(const float* logits,const unsigned char* seen,float repetition_penalty,int* out){
    __shared__ float best_vals[256];
    __shared__ int best_ids[256];
    int tid=threadIdx.x;
    float best=-INFINITY;
    int best_id=0;
    for(int i=tid;i<VOCAB_SIZE;i+=blockDim.x){
        float v=logits[i];
        if(repetition_penalty>1.0f && seen[i]){
            v = v>0 ? v/repetition_penalty : v*repetition_penalty;
        }
        if(v>best || (v==best && i<best_id)){
            best=v;
            best_id=i;
        }
    }
    best_vals[tid]=best;
    best_ids[tid]=best_id;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            float other=best_vals[tid+stride];
            int other_id=best_ids[tid+stride];
            if(other>best_vals[tid] || (other==best_vals[tid] && other_id<best_ids[tid])){
                best_vals[tid]=other;
                best_ids[tid]=other_id;
            }
        }
        __syncthreads();
    }
    if(tid==0) *out=best_ids[0];
}
__global__ void argmax_stage1_kernel(const float* logits,const unsigned char* seen,float repetition_penalty,float* block_vals,int* block_ids){
    __shared__ float best_vals[256];
    __shared__ int best_ids[256];
    int tid=threadIdx.x;
    int start=blockIdx.x*blockDim.x+tid;
    int stride=blockDim.x*gridDim.x;
    float best=-INFINITY;
    int best_id=0;
    for(int i=start;i<VOCAB_SIZE;i+=stride){
        float v=logits[i];
        if(repetition_penalty!=1.0f && seen[i]){
            v = v>0.0f ? v/repetition_penalty : v*repetition_penalty;
        }
        if(v>best || (v==best && i<best_id)){
            best=v;
            best_id=i;
        }
    }
    best_vals[tid]=best;
    best_ids[tid]=best_id;
    __syncthreads();
    for(int stride2=blockDim.x/2;stride2>0;stride2>>=1){
        if(tid<stride2){
            float other=best_vals[tid+stride2];
            int other_id=best_ids[tid+stride2];
            if(other>best_vals[tid] || (other==best_vals[tid] && other_id<best_ids[tid])){
                best_vals[tid]=other;
                best_ids[tid]=other_id;
            }
        }
        __syncthreads();
    }
    if(tid==0){
        block_vals[blockIdx.x]=best_vals[0];
        block_ids[blockIdx.x]=best_ids[0];
    }
}
__global__ void argmax_stage2_kernel(const float* block_vals,const int* block_ids,int* out){
    __shared__ float best_vals[256];
    __shared__ int best_ids[256];
    int tid=threadIdx.x;
    float best=(tid<ARGMAX_BLOCKS) ? block_vals[tid] : -INFINITY;
    int best_id=(tid<ARGMAX_BLOCKS) ? block_ids[tid] : 0;
    best_vals[tid]=best;
    best_ids[tid]=best_id;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            float other=best_vals[tid+stride];
            int other_id=best_ids[tid+stride];
            if(other>best_vals[tid] || (other==best_vals[tid] && other_id<best_ids[tid])){
                best_vals[tid]=other;
                best_ids[tid]=other_id;
            }
        }
        __syncthreads();
    }
    if(tid==0) *out=best_ids[0];
}

struct Layer{
    float *ln1=nullptr,*ln2=nullptr,*bq=nullptr,*bk=nullptr,*bv=nullptr;
    WeightMatrix wq,wk,wv,wo,wgate,wup,wdown;
    KvCacheT *kc=nullptr,*vc=nullptr;
};
struct Model{
    WeightMatrix emb,lm;
    float *norm=nullptr,*rope_cos=nullptr,*rope_sin=nullptr;
    float rms_norm_eps=DEFAULT_RMS_NORM_EPS,rope_theta=DEFAULT_ROPE_THETA;
    Layer layers[N_LAYERS];
};
struct Work{
    float *x=nullptr,*n=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*ctx=nullptr,*ao=nullptr;
    float *gate=nullptr,*up=nullptr,*mid=nullptr,*mo=nullptr,*logits=nullptr;
    float *attn_scores=nullptr;
    float *argmax_vals=nullptr;
    float *xq_scale=nullptr;
    int *argmax_ids=nullptr;
    int8_t *xq=nullptr;
    half *wmma_x=nullptr;
};
struct Engine{
    Model m;
    Work w;
    int max_seq=0,pos=0;
    float repetition_penalty=1.0f;
    unsigned char* seen=nullptr;
    int* next_token=nullptr;
    double prefill_ms=0.0;
    double decode_ms=0.0;
    double forward_ms=0.0;
    double sample_ms=0.0;
    int prefill_tokens=0;
    int decode_tokens=0;
};

template <typename T>
static void freep(T*& p){if(p){cudaFree(p);p=nullptr;}}
static void free_weight(WeightMatrix& w){freep(w.data);freep(w.scale);}
static std::string lname(int i,const std::string& s){return "model.layers."+std::to_string(i)+"."+s;}

static void init_rope_table(Model& m,int max_seq){
    size_t n=(size_t)max_seq*(HEAD_DIM/2);
    std::vector<float> hc(n),hs(n);
    for(int pos=0;pos<max_seq;pos++){
        for(int p=0;p<HEAD_DIM/2;p++){
            float inv=std::pow(m.rope_theta,-(float)(2*p)/HEAD_DIM);
            float a=pos*inv;
            hc[(size_t)pos*(HEAD_DIM/2)+p]=std::cos(a);
            hs[(size_t)pos*(HEAD_DIM/2)+p]=std::sin(a);
        }
    }
    CK(cudaMalloc(&m.rope_cos,n*sizeof(float)));
    CK(cudaMalloc(&m.rope_sin,n*sizeof(float)));
    CK(cudaMemcpy(m.rope_cos,hc.data(),n*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(m.rope_sin,hs.data(),n*sizeof(float),cudaMemcpyHostToDevice));
    std::cout<<"[C++] initialized RoPE table, max_seq="<<max_seq
             <<", entries="<<n<<"\n";
}

static Work make_work(int max_seq){
    Work w;
    CK(cudaMalloc(&w.x,HIDDEN*sizeof(float))); CK(cudaMalloc(&w.n,HIDDEN*sizeof(float)));
    CK(cudaMalloc(&w.q,HIDDEN*sizeof(float))); CK(cudaMalloc(&w.k,KV_DIM*sizeof(float))); CK(cudaMalloc(&w.v,KV_DIM*sizeof(float)));
    CK(cudaMalloc(&w.ctx,HIDDEN*sizeof(float))); CK(cudaMalloc(&w.ao,HIDDEN*sizeof(float)));
    CK(cudaMalloc(&w.gate,INTERMEDIATE*sizeof(float))); CK(cudaMalloc(&w.up,INTERMEDIATE*sizeof(float)));
    CK(cudaMalloc(&w.mid,INTERMEDIATE*sizeof(float))); CK(cudaMalloc(&w.mo,HIDDEN*sizeof(float)));
    CK(cudaMalloc(&w.logits,VOCAB_SIZE*sizeof(float)));
    CK(cudaMalloc(&w.attn_scores,(size_t)N_HEADS*max_seq*sizeof(float)));
    CK(cudaMalloc(&w.argmax_vals,ARGMAX_BLOCKS*sizeof(float)));
    CK(cudaMalloc(&w.xq_scale,sizeof(float)));
    CK(cudaMalloc(&w.argmax_ids,ARGMAX_BLOCKS*sizeof(int)));
    CK(cudaMalloc(&w.xq,INTERMEDIATE*sizeof(int8_t)));
    CK(cudaMalloc(&w.wmma_x,INTERMEDIATE*sizeof(half)));
    return w;
}
static void free_work(Work& w){
    freep(w.x);freep(w.n);freep(w.q);freep(w.k);freep(w.v);freep(w.ctx);freep(w.ao);
    freep(w.gate);freep(w.up);freep(w.mid);freep(w.mo);freep(w.logits);freep(w.attn_scores);
    freep(w.argmax_vals);freep(w.xq_scale);freep(w.argmax_ids);freep(w.xq);freep(w.wmma_x);
}
static Model load_model(const std::string& dir,int max_seq){
    ModelConfig cfg=load_config(dir);
    auto metas=scan_safetensors(dir);
    Model m;
    m.rms_norm_eps=cfg.rms_norm_eps;
    m.rope_theta=cfg.rope_theta;
    init_rope_table(m,max_seq);
    m.emb=load_weight_tensor(metas,"model.embed_tokens.weight");
    m.norm=load_tensor(metas,"model.norm.weight");
    m.lm=load_weight_tensor(metas,"lm_head.weight");
    for(int i=0;i<N_LAYERS;i++){
        std::cout<<"\n[C++] loading layer "<<i<<"\n";
        Layer& l=m.layers[i];
        l.ln1=load_tensor(metas,lname(i,"input_layernorm.weight"));
        l.ln2=load_tensor(metas,lname(i,"post_attention_layernorm.weight"));
        l.wq=load_weight_tensor(metas,lname(i,"self_attn.q_proj.weight"));
        l.wk=load_weight_tensor(metas,lname(i,"self_attn.k_proj.weight"));
        l.wv=load_weight_tensor(metas,lname(i,"self_attn.v_proj.weight"));
        l.wo=load_weight_tensor(metas,lname(i,"self_attn.o_proj.weight"));
        l.bq=load_tensor(metas,lname(i,"self_attn.q_proj.bias"));
        l.bk=load_tensor(metas,lname(i,"self_attn.k_proj.bias"));
        l.bv=load_tensor(metas,lname(i,"self_attn.v_proj.bias"));
        l.wgate=load_weight_tensor(metas,lname(i,"mlp.gate_proj.weight"));
        l.wup=load_weight_tensor(metas,lname(i,"mlp.up_proj.weight"));
        l.wdown=load_weight_tensor(metas,lname(i,"mlp.down_proj.weight"));
        CK(cudaMalloc(&l.kc,(size_t)max_seq*KV_DIM*sizeof(KvCacheT)));
        CK(cudaMalloc(&l.vc,(size_t)max_seq*KV_DIM*sizeof(KvCacheT)));
        CK(cudaMemset(l.kc,0,(size_t)max_seq*KV_DIM*sizeof(KvCacheT)));
        CK(cudaMemset(l.vc,0,(size_t)max_seq*KV_DIM*sizeof(KvCacheT)));
    }
    return m;
}
static void free_model(Model& m){
    free_weight(m.emb); freep(m.norm); freep(m.rope_cos); freep(m.rope_sin); free_weight(m.lm);
    for(int i=0;i<N_LAYERS;i++){
        Layer& l=m.layers[i];
        freep(l.ln1);freep(l.ln2);free_weight(l.wq);free_weight(l.wk);free_weight(l.wv);free_weight(l.wo);
        freep(l.bq);freep(l.bk);freep(l.bv);free_weight(l.wgate);free_weight(l.wup);free_weight(l.wdown);
        freep(l.kc);freep(l.vc);
    }
}
static void mark_seen(unsigned char* seen,int token){
    mark_token_seen_kernel<<<1,1>>>(seen,token);
}
static void prepare_wmma_x(const float* x,half* xh,int n){
    int B=256;
    float_to_half_kernel<<<(n+B-1)/B,B>>>(x,xh,n);
}
static void launch_wmma_linear(const half* xh,const WeightMatrix& W,const float* b,float* y,int IN,int OUT){
    wmma_linear_kernel<<<(OUT+WMMA_TILE-1)/WMMA_TILE,32>>>(xh,W.data,b,y,IN,OUT);
}
static void prepare_int8_x(const float* x,int8_t* xq,float* xq_scale,int n){
    quantize_int8_kernel<<<1,256>>>(x,xq,xq_scale,n);
}
static void prepare_silu_int8_x(const float* gate,const float* up,int8_t* xq,float* xq_scale,int n){
    silu_quantize_int8_kernel<<<1,256>>>(gate,up,xq,xq_scale,n);
}
static void launch_linear_from_float(const float* x,half* xh,int8_t* xq,float* xq_scale,const WeightMatrix& W,const float* b,float* y,int IN,int OUT){
#if USE_WMMA_LINEAR && !USE_INT8_WEIGHTS && !USE_INT4_WEIGHTS
    prepare_wmma_x(x,xh,IN);
    launch_wmma_linear(xh,W,b,y,IN,OUT);
#elif USE_INT4_WEIGHTS && USE_INT4_DP4A
    prepare_int8_x(x,xq,xq_scale,IN);
    int B=LINEAR_THREADS;
#if USE_LINEAR_I4_DP4A4
    linear_i4_dp4a4_kernel<<<(OUT+3)/4,B>>>(xq,xq_scale,W.data,W.scale,b,y,IN,OUT);
#elif USE_LINEAR_I4_DP4A2
    linear_i4_dp4a2_kernel<<<(OUT+1)/2,B>>>(xq,xq_scale,W.data,W.scale,b,y,IN,OUT);
#else
    linear_i4_dp4a_kernel<<<OUT,B>>>(xq,xq_scale,W.data,W.scale,b,y,IN,OUT);
#endif
#else
    int B=LINEAR_THREADS;
    linear_kernel<<<OUT,B>>>(x,W.data,W.scale,b,y,IN,OUT);
#endif
}
static void launch_qkv_from_float(
    const float* x,half* xh,int8_t* xq,float* xq_scale,
    const WeightMatrix& Wq,const WeightMatrix& Wk,const WeightMatrix& Wv,
    const float* bq,const float* bk,const float* bv,
    float* q,float* k,float* v,
    int IN
){
#if USE_WMMA_LINEAR && !USE_INT8_WEIGHTS && !USE_INT4_WEIGHTS
    prepare_wmma_x(x,xh,IN);
    launch_wmma_linear(xh,Wq,bq,q,IN,HIDDEN);
    launch_wmma_linear(xh,Wk,bk,k,IN,KV_DIM);
    launch_wmma_linear(xh,Wv,bv,v,IN,KV_DIM);
#elif USE_INT4_WEIGHTS && USE_INT4_DP4A
    prepare_int8_x(x,xq,xq_scale,IN);
    int B=LINEAR_THREADS;
#if USE_QKV_GATE_I4_DP4A4
    qkv_i4_dp4a4_kernel<<<(HIDDEN+2*KV_DIM+3)/4,B>>>(xq,xq_scale,Wq.data,Wk.data,Wv.data,Wq.scale,Wk.scale,Wv.scale,bq,bk,bv,q,k,v,IN);
#elif USE_QKV_GATE_I4_DP4A2
    qkv_i4_dp4a2_kernel<<<(HIDDEN+2*KV_DIM+1)/2,B>>>(xq,xq_scale,Wq.data,Wk.data,Wv.data,Wq.scale,Wk.scale,Wv.scale,bq,bk,bv,q,k,v,IN);
#else
    qkv_i4_dp4a_kernel<<<HIDDEN+2*KV_DIM,B>>>(xq,xq_scale,Wq.data,Wk.data,Wv.data,Wq.scale,Wk.scale,Wv.scale,bq,bk,bv,q,k,v,IN);
#endif
#else
    int B=LINEAR_THREADS;
    qkv_linear_kernel<<<HIDDEN+2*KV_DIM,B>>>(x,Wq.data,Wk.data,Wv.data,Wq.scale,Wk.scale,Wv.scale,bq,bk,bv,q,k,v,IN);
#endif
}
static void launch_gate_up_from_float(
    const float* x,half* xh,int8_t* xq,float* xq_scale,
    const WeightMatrix& Wgate,const WeightMatrix& Wup,
    float* gate,float* up,
    int IN
){
#if USE_WMMA_LINEAR && !USE_INT8_WEIGHTS && !USE_INT4_WEIGHTS
    prepare_wmma_x(x,xh,IN);
    launch_wmma_linear(xh,Wgate,nullptr,gate,IN,INTERMEDIATE);
    launch_wmma_linear(xh,Wup,nullptr,up,IN,INTERMEDIATE);
#elif USE_INT4_WEIGHTS && USE_INT4_DP4A
    prepare_int8_x(x,xq,xq_scale,IN);
    int B=LINEAR_THREADS;
#if USE_QKV_GATE_I4_DP4A4
    gate_up_i4_dp4a4_kernel<<<(2*INTERMEDIATE+3)/4,B>>>(xq,xq_scale,Wgate.data,Wup.data,Wgate.scale,Wup.scale,gate,up,IN);
#elif USE_QKV_GATE_I4_DP4A2
    gate_up_i4_dp4a2_kernel<<<(2*INTERMEDIATE+1)/2,B>>>(xq,xq_scale,Wgate.data,Wup.data,Wgate.scale,Wup.scale,gate,up,IN);
#else
    gate_up_i4_dp4a_kernel<<<2*INTERMEDIATE,B>>>(xq,xq_scale,Wgate.data,Wup.data,Wgate.scale,Wup.scale,gate,up,IN);
#endif
#else
    int B=LINEAR_THREADS;
    gate_up_linear_kernel<<<2*INTERMEDIATE,B>>>(x,Wgate.data,Wup.data,Wgate.scale,Wup.scale,gate,up,IN);
#endif
}
static void forward_token(const Model& m,Work& w,int token,int pos,int max_seq){
    if(token<0||token>=VOCAB_SIZE) throw std::runtime_error("bad token id "+std::to_string(token));
    int B=256;
    embedding_kernel<<<(HIDDEN+B-1)/B,B>>>(token,m.emb.data,m.emb.scale,w.x);
    rmsnorm_kernel<<<1,B,B*sizeof(float)>>>(w.x,m.layers[0].ln1,w.n,HIDDEN,m.rms_norm_eps);
    for(int i=0;i<N_LAYERS;i++){
        const Layer& l=m.layers[i];
        launch_qkv_from_float(w.n,w.wmma_x,w.xq,w.xq_scale,l.wq,l.wk,l.wv,l.bq,l.bk,l.bv,w.q,w.k,w.v,HIDDEN);
#if USE_FUSED_ROPE_ATTENTION
        attention_rope_fused_kernel<<<N_HEADS,B>>>(w.q,w.k,w.v,l.kc,l.vc,w.attn_scores,w.ctx,m.rope_cos,m.rope_sin,pos,max_seq);
#else
        rope_store_kv_kernel<<<(N_HEADS*(HEAD_DIM/2)+B-1)/B,B>>>(w.q,w.k,w.v,l.kc,l.vc,m.rope_cos,m.rope_sin,pos);
#if USE_ATTENTION_SHM
        attention_fused_kernel<<<N_HEADS,B,(B+HEAD_DIM+max_seq)*sizeof(float)>>>(w.q,l.kc,l.vc,w.attn_scores,w.ctx,pos,max_seq);
#else
        attention_fused_kernel<<<N_HEADS,B>>>(w.q,l.kc,l.vc,w.attn_scores,w.ctx,pos,max_seq);
#endif
#endif
        launch_linear_from_float(w.ctx,w.wmma_x,w.xq,w.xq_scale,l.wo,nullptr,w.ao,HIDDEN,HIDDEN);
        add_rmsnorm_kernel<<<1,B,B*sizeof(float)>>>(w.x,w.ao,l.ln2,w.n,HIDDEN,m.rms_norm_eps);
        launch_gate_up_from_float(w.n,w.wmma_x,w.xq,w.xq_scale,l.wgate,l.wup,w.gate,w.up,HIDDEN);
#if USE_INT4_WEIGHTS && USE_INT4_DP4A && USE_FUSED_MLP_QUANT
        prepare_silu_int8_x(w.gate,w.up,w.xq,w.xq_scale,INTERMEDIATE);
        linear_i4_dp4a_kernel<<<HIDDEN,LINEAR_THREADS>>>(w.xq,w.xq_scale,l.wdown.data,l.wdown.scale,nullptr,w.mo,INTERMEDIATE,HIDDEN);
#else
        silu_mul_kernel<<<(INTERMEDIATE+B-1)/B,B>>>(w.gate,w.up,w.mid,INTERMEDIATE);
        launch_linear_from_float(w.mid,w.wmma_x,w.xq,w.xq_scale,l.wdown,nullptr,w.mo,INTERMEDIATE,HIDDEN);
#endif
        if(i+1<N_LAYERS){
            add_rmsnorm_kernel<<<1,B,B*sizeof(float)>>>(w.x,w.mo,m.layers[i+1].ln1,w.n,HIDDEN,m.rms_norm_eps);
        }else{
            add_kernel<<<(HIDDEN+B-1)/B,B>>>(w.x,w.mo,HIDDEN);
        }
    }
    rmsnorm_kernel<<<1,B,B*sizeof(float)>>>(w.x,m.norm,w.n,HIDDEN,m.rms_norm_eps);
    launch_linear_from_float(w.n,w.wmma_x,w.xq,w.xq_scale,m.lm,nullptr,w.logits,HIDDEN,VOCAB_SIZE);
    CK(cudaDeviceSynchronize());
}
static int argmax_gpu_to_cpu(float* logits,const unsigned char* seen,float repetition_penalty,float* block_vals,int* block_ids,int* next_token){
    argmax_stage1_kernel<<<ARGMAX_BLOCKS,256>>>(logits,seen,repetition_penalty,block_vals,block_ids);
    argmax_stage2_kernel<<<1,256>>>(block_vals,block_ids,next_token);
    int h=0;
    CK(cudaMemcpy(&h,next_token,sizeof(int),cudaMemcpyDeviceToHost));
    return h;
}

extern "C" {
const char* llm_last_error(){return g_err.c_str();}
void* llm_create(const char* model_dir,int max_seq){
    try{
        if(!model_dir) throw std::runtime_error("model_dir is null");
        Engine* e=new Engine(); e->max_seq=max_seq; e->pos=0;
        std::cout<<"[C++] create engine, model="<<model_dir<<", max_seq="<<max_seq<<"\n";
        reset_time_log();
        {
            std::ostringstream os;
            os<<"[C++][time] create engine, model="<<model_dir<<", max_seq="<<max_seq;
            time_log(os.str());
        }
        e->m=load_model(model_dir,max_seq); e->w=make_work(max_seq);
        CK(cudaMalloc(&e->seen,VOCAB_SIZE*sizeof(unsigned char)));
        CK(cudaMalloc(&e->next_token,sizeof(int)));
        CK(cudaMemset(e->seen,0,VOCAB_SIZE*sizeof(unsigned char)));
        return e;
    }catch(const std::exception& ex){g_err=ex.what(); return nullptr;}
}
void llm_destroy(void* h){
    if(!h) return; Engine* e=(Engine*)h; freep(e->next_token); freep(e->seen); free_work(e->w); free_model(e->m); delete e;
}
int llm_set_repetition_penalty(void* h,float penalty){
    try{
        if(!h) throw std::runtime_error("handle null");
        if(penalty<1.0f) throw std::runtime_error("repetition penalty must be >= 1.0");
        ((Engine*)h)->repetition_penalty=penalty;
        return 0;
    }catch(const std::exception& ex){g_err=ex.what(); return -1;}
}
int llm_prefill(void* h,const int* tokens,int n){
    try{
        if(!h) throw std::runtime_error("handle null");
        if(!tokens) throw std::runtime_error("tokens null");
        Engine* e=(Engine*)h; if(n<=0||n>e->max_seq) throw std::runtime_error("bad prefill length");
        e->pos=0;
        e->prefill_ms=0.0;
        e->decode_ms=0.0;
        e->forward_ms=0.0;
        e->sample_ms=0.0;
        e->prefill_tokens=0;
        e->decode_tokens=0;
        CK(cudaMemset(e->seen,0,VOCAB_SIZE*sizeof(unsigned char)));
        auto prefill_start=Clock::now();
        for(int i=0;i<n;i++){
            std::cout<<"[C++] prefill pos="<<e->pos<<", token="<<tokens[i]<<"\n";
            auto forward_start=Clock::now();
            forward_token(e->m,e->w,tokens[i],e->pos,e->max_seq);
            double forward_ms=elapsed_ms(forward_start,Clock::now());
            e->forward_ms+=forward_ms;
            mark_seen(e->seen,tokens[i]);
            e->pos++;
            e->prefill_tokens++;
            {
                std::ostringstream os;
                os<<"[C++][time] prefill token "<<i<<" forward_ms="<<forward_ms;
                time_log(os.str());
            }
        }
        e->prefill_ms+=elapsed_ms(prefill_start,Clock::now());
        double tps=e->prefill_ms>0.0 ? (1000.0*e->prefill_tokens/e->prefill_ms) : 0.0;
        {
            std::ostringstream os;
            os<<"[C++][time] prefill total_ms="<<e->prefill_ms
              <<", tokens="<<e->prefill_tokens
              <<", tokens_per_s="<<tps;
            time_log(os.str());
        }
        return 0;
    }catch(const std::exception& ex){g_err=ex.what(); return -1;}
}
int llm_decode_one(void* h,int* next){
    try{
        if(!h) throw std::runtime_error("handle null");
        if(!next) throw std::runtime_error("next null");
        Engine* e=(Engine*)h; if(e->pos>=e->max_seq) throw std::runtime_error("pos >= max_seq");
        auto decode_start=Clock::now();
        auto sample_start=Clock::now();
        int t=argmax_gpu_to_cpu(e->w.logits,e->seen,e->repetition_penalty,e->w.argmax_vals,e->w.argmax_ids,e->next_token);
        double sample_ms=elapsed_ms(sample_start,Clock::now());
        e->sample_ms+=sample_ms;
        *next=t;
        std::cout<<"[C++] decode pos="<<e->pos<<", token="<<t<<"\n";
        mark_seen(e->seen,t);
        auto forward_start=Clock::now();
        forward_token(e->m,e->w,t,e->pos,e->max_seq);
        double forward_ms=elapsed_ms(forward_start,Clock::now());
        e->forward_ms+=forward_ms;
        e->pos++;
        e->decode_tokens++;
        double decode_ms=elapsed_ms(decode_start,Clock::now());
        e->decode_ms+=decode_ms;
        double tps=e->decode_ms>0.0 ? (1000.0*e->decode_tokens/e->decode_ms) : 0.0;
        {
            std::ostringstream os;
            os<<"[C++][time] decode step_ms="<<decode_ms
              <<", sample_ms="<<sample_ms
              <<", forward_ms="<<forward_ms
              <<", decode_tokens="<<e->decode_tokens
              <<", decode_tokens_per_s="<<tps;
            time_log(os.str());
        }
        return 0;
    }catch(const std::exception& ex){g_err=ex.what(); return -1;}
}
int llm_get_pos(void* h){return h?((Engine*)h)->pos:-1;}
}
