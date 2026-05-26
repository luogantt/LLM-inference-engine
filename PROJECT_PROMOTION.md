# 比华为官方 torch_npu 更快：我用 C++/Ascend 写了一个 7B 推理引擎

项目地址：

[https://github.com/luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)

最近我在做一个实验性项目：**LLM-inference-engine**。

它不是基于 PyTorch，也不是基于 transformers 推理栈，而是直接用 C++ 写推理核心，并针对 Ascend NPU 做了一条 direct `.so` 推理路径。

这次在 `Ascend-super` 分支上，我把 DeepSeek-R1-Distill-Qwen-7B 的单 batch decode 跑到了一个很有意思的结果：

**同一个 prompt，同样生成 128 tokens，我的 direct `.so` 稳态速度已经超过华为官方 torch_npu baseline。**

当前公开 tag：

```bash
ascend-super-36tok
```

这意味着：在这个单 batch、低延迟 decode 算例里，一条手写 C++ / AscendCL / ACLNN direct runtime 路径，已经跑过了官方通用 PyTorch NPU 路径。下一步就是继续对标并挑战 vLLM-Ascend。

## 下载项目指定 tag

直接下载这次性能记录对应的 tag：

```bash
git clone --branch ascend-super-36tok --depth 1 https://github.com/luogantt/LLM-inference-engine.git
cd LLM-inference-engine
```

如果你已经 clone 过项目：

```bash
cd ~/LLM-inference-engine
git fetch origin --tags
git checkout ascend-super-36tok
```

## 下载模型

模型使用：

```text
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

中国大陆网络建议使用 ModelScope：

```bash
pip install -U modelscope

python download_model.py \
  --source modelscope \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir ./deepseek-r1-7b
```

如果你的 ModelScope 下载目录缺少 HuggingFace 风格的校验文件，可以跳过下载后的文件名检查：

```bash
python download_model.py \
  --source modelscope \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir ./deepseek-r1-7b \
  --skip-check
```

也可以使用 HuggingFace Hub：

```bash
pip install -U huggingface_hub

python download_model.py \
  --source huggingface \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir ./deepseek-r1-7b
```

## 编译 Ascend direct .so

```bash
cd ~/LLM-inference-engine

make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-ascend ASCEND_HOME=/usr/local/Ascend/cann-8.5.1
```

## 测试模型和任务

模型：

```text
DeepSeek-R1-Distill-Qwen-7B
```

prompt：

```text
黑格尔的哲学思想可以概括为
```

生成长度：

```text
max_new_tokens = 128
max_seq = 800
```

## 1. torch_npu baseline

先跑华为官方 torch_npu 路径：

```bash
cd ~/LLM-inference-engine

export ASCEND_VISIBLE_DEVICES=4

python python_infer_ascend.py \
  --model ./deepseek-r1-7b \
  --prompt "黑格尔的哲学思想可以概括为" \
  --max-new-tokens 128 \
  --max-seq 800 \
  --device npu:0 \
  --dtype float16 \
  2>&1 | tee torch_npu_128.log
```

实测结果：

```text
generated_tokens=128
elapsed_s=3.696
tokens_per_s=34.627
```

也就是：

```text
torch_npu = 34.627 tok/s
```

## 2. Ascend-super direct .so

然后跑我自己的 direct `.so` 推理路径：

```bash
cd ~/LLM-inference-engine

export ASCEND_VISIBLE_DEVICES=4
export ASCEND_DEVICE_ID=0

export ASCEND_LOAD_WEIGHTS=all
export ASCEND_WEIGHT_LOAD_LOG=0
export ASCEND_HOST_RAW_CACHE=0
export ASCEND_RUN_EMBED=1
export ASCEND_DIRECT_DECODE=all_layers_ref
export ASCEND_REF_CACHE_WEIGHTS=1
export ASCEND_REF_CACHE_LOG=0
export ASCEND_REF_KV_CACHE=1
export ASCEND_REF_U16_WEIGHTS=1
export ASCEND_REF_FAST_DOT=1
export ASCEND_REF_DOT4=0
export ASCEND_REF_NEON_DOT=0
export ASCEND_ATTN_BACKEND=cpu
export ASCEND_QKV_BACKEND=aclnn
export ASCEND_QKV_FALLBACK=0
export ASCEND_QKV_LOG=0
export ASCEND_ATTN_PROJ_BACKEND=aclnn
export ASCEND_ATTN_PROJ_FALLBACK=0
export ASCEND_ATTN_PROJ_LOG=0
export ASCEND_MLP_BACKEND=aclnn
export ASCEND_MLP_FALLBACK=0
export ASCEND_MLP_LOG=0
export ASCEND_LM_HEAD_BACKEND=aclnn
export ASCEND_LM_HEAD_FALLBACK=0
export ASCEND_LM_HEAD_LOG=0
export ASCEND_ACLNN_CUBE_MATH_TYPE=0
export ASCEND_REF_LINEAR_THREADS=16
export ASCEND_REF_ATTN_LINEAR_THREADS=16
export ASCEND_REF_ATTN_THREADS=16
export ASCEND_REF_ATTN_THREAD_MIN_SEQ=32
export ASCEND_REF_MLP_THREADS=24
export ASCEND_REF_DOWN_THREADS=24
export ASCEND_LM_HEAD_THREADS=16
export ASCEND_REF_PROFILE_LAYERS=0
export ASCEND_REF_PROFILE_TOKEN_LIMIT=0

python python_infer.py \
  --model ./deepseek-r1-7b \
  --lib ./build/libllm_ascend.so \
  --prompt "黑格尔的哲学思想可以概括为" \
  --max-new-tokens 128 \
  --max-seq 800 \
  --tokenizer-backend tokenizers \
  --no-chat-template \
  2>&1 | tee ascend_super_128.log
```

查看尾部 decode 速度：

```bash
grep "decode all_layers_ref finished" ascend_super_128.log | tail -20
```

查看最后一个 token：

```bash
grep "decode all_layers_ref finished" ascend_super_128.log | tail -1
```

其中一个稳定尾部结果是：

```text
elapsed_ms=27.985649
```

换算：

```text
1000 / 27.985649 = 35.73 tok/s
```

也就是：

```text
Ascend-super direct .so ≈ 35.7 tok/s
```

## 结果对比

```text
torch_npu baseline       : 34.627 tok/s
Ascend-super direct .so  : 35.73 tok/s
```

提升比例：

```text
(35.73 / 34.627 - 1) * 100% ≈ 3.2%
```

这不是一个靠 batch 堆出来的吞吐数字，而是单 batch decode 的低延迟结果。

这也不是一个黑盒框架调参结果，而是把推理链路拆开之后，一步一步把 QKV、MLP、attention projection、lm_head 等路径迁到 AscendCL / ACLNN 后得到的结果。

## 为什么这个结果有意义

很多人做 Ascend 推理，第一反应是直接上 torch_npu，或者用 vLLM-Ascend 这样的上层推理框架。

但这个算例说明了一件事：

**在单 batch decode 场景下，手写 C++ direct runtime 路径仍然有非常明确的优化空间，甚至可以跑过官方 torch_npu baseline。**

当前 `Ascend-super` 路径里已经做了这些优化：

```text
QKV:               ACLNN BF16
O projection:      ACLNN BF16
MLP:               ACLNN BF16
LM Head:           ACLNN BF16
KV Cache:          enabled
weight cache:      enabled
CPU attention:     optimized fallback
RoPE cache:        enabled
```

当前尾部瓶颈大概是：

```text
all_layers ≈ 25.5 ms
lm_head    ≈ 2.2 ms
decode     ≈ 27.9 ms
```

后面还有继续优化的空间，尤其是：

```text
1. 把 lm_head argmax/top1 继续搬到 device 上
2. 写真正融合的 Ascend attention kernel
3. 减少同步点和 D2H/H2D 数据搬运
4. 进一步融合 decode 路径
```

## 项目定位

这个项目不是为了包装一个黑盒框架，而是为了把大模型推理路径拆开：

```text
embedding
RMSNorm
QKV
RoPE
attention
MLP
KV cache
lm_head
decode loop
```

每一层都尽量透明，每一个耗时都能看到。

它适合：

```text
1. 学习 LLM 推理底层实现
2. 研究 Ascend NPU 上的 decode 优化
3. 对比 torch_npu / vLLM-Ascend / 自定义 runtime 的性能边界
4. 做单 batch、低延迟推理实验
```

## 一句话总结

在 `ascend-super-36tok` 这个 tag 上，LLM-inference-engine 已经用 direct `.so` 路径，在 DeepSeek-R1-Distill-Qwen-7B 的 128-token decode 算例中，跑出了约 **35.7 tok/s** 的稳态速度，超过了同机 torch_npu baseline 的 **34.6 tok/s**。

项目地址：

[https://github.com/luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)

欢迎 star、fork、复现 benchmark，也欢迎拿 vLLM-Ascend 的同 prompt、同 128 tokens、同单 batch 配置来正面对比。
