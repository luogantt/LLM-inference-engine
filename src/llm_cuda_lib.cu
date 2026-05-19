#include <cuda_runtime.h>
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

#define CK(x) do { cudaError_t _cuda_err=(x); if(_cuda_err!=cudaSuccess) throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(_cuda_err)); } while(0)

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

static thread_local std::string g_err;
using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
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

__global__ void embedding_kernel(int token,const float* emb,float* x){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<HIDDEN) x[i]=emb[(size_t)token*HIDDEN+i];
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
__global__ void linear_kernel(const float* x,const float* W,const float* b,float* y,int IN,int OUT){
    __shared__ float sh[256];
    int o=blockIdx.x;
    int tid=threadIdx.x;
    if(o>=OUT) return;
    const float* row=W+(size_t)o*IN;
    float s=0;
    for(int i=tid;i<IN;i+=blockDim.x) s+=row[i]*x[i];
    sh[tid]=s;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride) sh[tid]+=sh[tid+stride];
        __syncthreads();
    }
    if(tid==0) y[o]=b ? sh[0]+b[o] : sh[0];
}
__global__ void add_kernel(float* x,const float* y,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=y[i];
}
__global__ void silu_mul_kernel(const float* g,const float* u,float* o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){float x=g[i]; o[i]=(x/(1.f+expf(-x)))*u[i];}
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
__global__ void attn_kernel(const float* q,const float* kc,const float* vc,float* ctx,int pos){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=HIDDEN) return;
    int d=idx%HEAD_DIM, h=idx/HEAD_DIM;
    int group=N_HEADS/N_KV_HEADS, kh=h/group;
    float mx=-1e30f;
    for(int t=0;t<=pos;t++){
        float dot=0;
        for(int r=0;r<HEAD_DIM;r++) dot+=q[h*HEAD_DIM+r]*kc[(size_t)t*KV_DIM+kh*HEAD_DIM+r];
        mx=fmaxf(mx,dot/sqrtf((float)HEAD_DIM));
    }
    float den=0,out=0;
    for(int t=0;t<=pos;t++){
        float dot=0;
        for(int r=0;r<HEAD_DIM;r++) dot+=q[h*HEAD_DIM+r]*kc[(size_t)t*KV_DIM+kh*HEAD_DIM+r];
        float e=expf(dot/sqrtf((float)HEAD_DIM)-mx);
        den+=e; out+=e*vc[(size_t)t*KV_DIM+kh*HEAD_DIM+d];
    }
    ctx[idx]=out/den;
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

struct Layer{
    float *ln1=nullptr,*ln2=nullptr,*wq=nullptr,*wk=nullptr,*wv=nullptr,*wo=nullptr;
    float *bq=nullptr,*bk=nullptr,*bv=nullptr,*wgate=nullptr,*wup=nullptr,*wdown=nullptr;
    float *kc=nullptr,*vc=nullptr;
};
struct Model{float *emb=nullptr,*norm=nullptr,*lm=nullptr; float rms_norm_eps=DEFAULT_RMS_NORM_EPS,rope_theta=DEFAULT_ROPE_THETA; Layer layers[N_LAYERS];};
struct Work{
    float *x=nullptr,*n=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*ctx=nullptr,*ao=nullptr;
    float *gate=nullptr,*up=nullptr,*mid=nullptr,*mo=nullptr,*logits=nullptr;
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
static std::string lname(int i,const std::string& s){return "model.layers."+std::to_string(i)+"."+s;}

static Work make_work(){
    Work w;
    CK(cudaMalloc(&w.x,HIDDEN*sizeof(float))); CK(cudaMalloc(&w.n,HIDDEN*sizeof(float)));
    CK(cudaMalloc(&w.q,HIDDEN*sizeof(float))); CK(cudaMalloc(&w.k,KV_DIM*sizeof(float))); CK(cudaMalloc(&w.v,KV_DIM*sizeof(float)));
    CK(cudaMalloc(&w.ctx,HIDDEN*sizeof(float))); CK(cudaMalloc(&w.ao,HIDDEN*sizeof(float)));
    CK(cudaMalloc(&w.gate,INTERMEDIATE*sizeof(float))); CK(cudaMalloc(&w.up,INTERMEDIATE*sizeof(float)));
    CK(cudaMalloc(&w.mid,INTERMEDIATE*sizeof(float))); CK(cudaMalloc(&w.mo,HIDDEN*sizeof(float)));
    CK(cudaMalloc(&w.logits,VOCAB_SIZE*sizeof(float)));
    return w;
}
static void free_work(Work& w){
    freep(w.x);freep(w.n);freep(w.q);freep(w.k);freep(w.v);freep(w.ctx);freep(w.ao);
    freep(w.gate);freep(w.up);freep(w.mid);freep(w.mo);freep(w.logits);
}
static Model load_model(const std::string& dir,int max_seq){
    ModelConfig cfg=load_config(dir);
    auto metas=scan_safetensors(dir);
    Model m;
    m.rms_norm_eps=cfg.rms_norm_eps;
    m.rope_theta=cfg.rope_theta;
    m.emb=load_tensor(metas,"model.embed_tokens.weight");
    m.norm=load_tensor(metas,"model.norm.weight");
    m.lm=load_tensor(metas,"lm_head.weight");
    for(int i=0;i<N_LAYERS;i++){
        std::cout<<"\n[C++] loading layer "<<i<<"\n";
        Layer& l=m.layers[i];
        l.ln1=load_tensor(metas,lname(i,"input_layernorm.weight"));
        l.ln2=load_tensor(metas,lname(i,"post_attention_layernorm.weight"));
        l.wq=load_tensor(metas,lname(i,"self_attn.q_proj.weight"));
        l.wk=load_tensor(metas,lname(i,"self_attn.k_proj.weight"));
        l.wv=load_tensor(metas,lname(i,"self_attn.v_proj.weight"));
        l.wo=load_tensor(metas,lname(i,"self_attn.o_proj.weight"));
        l.bq=load_tensor(metas,lname(i,"self_attn.q_proj.bias"));
        l.bk=load_tensor(metas,lname(i,"self_attn.k_proj.bias"));
        l.bv=load_tensor(metas,lname(i,"self_attn.v_proj.bias"));
        l.wgate=load_tensor(metas,lname(i,"mlp.gate_proj.weight"));
        l.wup=load_tensor(metas,lname(i,"mlp.up_proj.weight"));
        l.wdown=load_tensor(metas,lname(i,"mlp.down_proj.weight"));
        CK(cudaMalloc(&l.kc,(size_t)max_seq*KV_DIM*sizeof(float)));
        CK(cudaMalloc(&l.vc,(size_t)max_seq*KV_DIM*sizeof(float)));
        CK(cudaMemset(l.kc,0,(size_t)max_seq*KV_DIM*sizeof(float)));
        CK(cudaMemset(l.vc,0,(size_t)max_seq*KV_DIM*sizeof(float)));
    }
    return m;
}
static void free_model(Model& m){
    freep(m.emb); freep(m.norm); freep(m.lm);
    for(int i=0;i<N_LAYERS;i++){
        Layer& l=m.layers[i];
        freep(l.ln1);freep(l.ln2);freep(l.wq);freep(l.wk);freep(l.wv);freep(l.wo);
        freep(l.bq);freep(l.bk);freep(l.bv);freep(l.wgate);freep(l.wup);freep(l.wdown);
        freep(l.kc);freep(l.vc);
    }
}
static void mark_seen(unsigned char* seen,int token){
    mark_token_seen_kernel<<<1,1>>>(seen,token);
}
static void forward_token(const Model& m,Work& w,int token,int pos){
    if(token<0||token>=VOCAB_SIZE) throw std::runtime_error("bad token id "+std::to_string(token));
    int B=256;
    embedding_kernel<<<(HIDDEN+B-1)/B,B>>>(token,m.emb,w.x);
    for(int i=0;i<N_LAYERS;i++){
        const Layer& l=m.layers[i];
        rmsnorm_kernel<<<1,B,B*sizeof(float)>>>(w.x,l.ln1,w.n,HIDDEN,m.rms_norm_eps);
        linear_kernel<<<HIDDEN,B>>>(w.n,l.wq,l.bq,w.q,HIDDEN,HIDDEN);
        linear_kernel<<<KV_DIM,B>>>(w.n,l.wk,l.bk,w.k,HIDDEN,KV_DIM);
        linear_kernel<<<KV_DIM,B>>>(w.n,l.wv,l.bv,w.v,HIDDEN,KV_DIM);
        rope_kernel<<<(N_HEADS*(HEAD_DIM/2)+B-1)/B,B>>>(w.q,N_HEADS,pos,m.rope_theta);
        rope_kernel<<<(N_KV_HEADS*(HEAD_DIM/2)+B-1)/B,B>>>(w.k,N_KV_HEADS,pos,m.rope_theta);
        store_kv_kernel<<<(KV_DIM+B-1)/B,B>>>(l.kc,w.k,pos,KV_DIM);
        store_kv_kernel<<<(KV_DIM+B-1)/B,B>>>(l.vc,w.v,pos,KV_DIM);
        attn_kernel<<<(HIDDEN+B-1)/B,B>>>(w.q,l.kc,l.vc,w.ctx,pos);
        linear_kernel<<<HIDDEN,B>>>(w.ctx,l.wo,nullptr,w.ao,HIDDEN,HIDDEN);
        add_kernel<<<(HIDDEN+B-1)/B,B>>>(w.x,w.ao,HIDDEN);
        rmsnorm_kernel<<<1,B,B*sizeof(float)>>>(w.x,l.ln2,w.n,HIDDEN,m.rms_norm_eps);
        linear_kernel<<<INTERMEDIATE,B>>>(w.n,l.wgate,nullptr,w.gate,HIDDEN,INTERMEDIATE);
        linear_kernel<<<INTERMEDIATE,B>>>(w.n,l.wup,nullptr,w.up,HIDDEN,INTERMEDIATE);
        silu_mul_kernel<<<(INTERMEDIATE+B-1)/B,B>>>(w.gate,w.up,w.mid,INTERMEDIATE);
        linear_kernel<<<HIDDEN,B>>>(w.mid,l.wdown,nullptr,w.mo,INTERMEDIATE,HIDDEN);
        add_kernel<<<(HIDDEN+B-1)/B,B>>>(w.x,w.mo,HIDDEN);
    }
    rmsnorm_kernel<<<1,B,B*sizeof(float)>>>(w.x,m.norm,w.n,HIDDEN,m.rms_norm_eps);
    linear_kernel<<<VOCAB_SIZE,B>>>(w.n,m.lm,nullptr,w.logits,HIDDEN,VOCAB_SIZE);
    CK(cudaDeviceSynchronize());
}
static int argmax_gpu_to_cpu(float* logits,const unsigned char* seen,float repetition_penalty,int* next_token){
    argmax_with_penalty_kernel<<<1,256>>>(logits,seen,repetition_penalty,next_token);
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
        e->m=load_model(model_dir,max_seq); e->w=make_work();
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
            forward_token(e->m,e->w,tokens[i],e->pos);
            double forward_ms=elapsed_ms(forward_start,Clock::now());
            e->forward_ms+=forward_ms;
            mark_seen(e->seen,tokens[i]);
            e->pos++;
            e->prefill_tokens++;
            std::cout<<"[C++][time] prefill token "<<i<<" forward_ms="<<forward_ms<<"\n";
        }
        e->prefill_ms+=elapsed_ms(prefill_start,Clock::now());
        double tps=e->prefill_ms>0.0 ? (1000.0*e->prefill_tokens/e->prefill_ms) : 0.0;
        std::cout<<"[C++][time] prefill total_ms="<<e->prefill_ms
                 <<", tokens="<<e->prefill_tokens
                 <<", tokens_per_s="<<tps<<"\n";
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
        int t=argmax_gpu_to_cpu(e->w.logits,e->seen,e->repetition_penalty,e->next_token);
        double sample_ms=elapsed_ms(sample_start,Clock::now());
        e->sample_ms+=sample_ms;
        *next=t;
        std::cout<<"[C++] decode pos="<<e->pos<<", token="<<t<<"\n";
        mark_seen(e->seen,t);
        auto forward_start=Clock::now();
        forward_token(e->m,e->w,t,e->pos);
        double forward_ms=elapsed_ms(forward_start,Clock::now());
        e->forward_ms+=forward_ms;
        e->pos++;
        e->decode_tokens++;
        double decode_ms=elapsed_ms(decode_start,Clock::now());
        e->decode_ms+=decode_ms;
        double tps=e->decode_ms>0.0 ? (1000.0*e->decode_tokens/e->decode_ms) : 0.0;
        std::cout<<"[C++][time] decode step_ms="<<decode_ms
                 <<", sample_ms="<<sample_ms
                 <<", forward_ms="<<forward_ms
                 <<", decode_tokens="<<e->decode_tokens
                 <<", decode_tokens_per_s="<<tps<<"\n";
        return 0;
    }catch(const std::exception& ex){g_err=ex.what(); return -1;}
}
int llm_get_pos(void* h){return h?((Engine*)h)->pos:-1;}
}
