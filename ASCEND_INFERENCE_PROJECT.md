# Ascend 大模型推理项目介绍：从 safetensors 到自研 `.so` 推理引擎

项目地址：[https://github.com/luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)

## 项目简介

`LLM-inference-engine` 是一个从底层实现大模型推理流程的实验型项目。它最初面向 CUDA / GPU 推理，后来增加了 Ascend 分支，用来探索在华为 Ascend 910 / CANN 环境下运行 DeepSeek-R1-Distill-Qwen-7B 这类大语言模型。

这个项目的重点不是简单调用现成推理框架，而是把大模型推理拆成可以观察、可以调试、可以逐步优化的底层模块。当前 Ascend 路径已经支持通过 C++ 动态库 `libllm_ascend.so` 加 Python 外壳的方式完成完整模型推理。Python 主要负责 tokenizer、参数解析和调用 `.so`，真正的模型目录扫描、权重加载、AscendCL 初始化、HBM 内存管理、prefill 和 decode 流程都放在 C++ 侧实现。

也就是说，这不是单纯的 `torch_npu + transformers` 推理脚本，而是在向“自研 Ascend 大模型推理引擎”推进。

## 项目目标

这个项目希望实现一条尽量清晰、可控的大模型推理链路：

- 读取 HuggingFace / ModelScope 格式的 safetensors 权重；
- 解析模型配置，例如层数、hidden size、head 数、KV head 数、词表大小等；
- 使用 AscendCL / CANN 初始化 NPU 设备；
- 将 embedding、Transformer 层权重、final norm、lm_head 等权重加载到 Ascend HBM；
- 通过 Python tokenizer 得到输入 token；
- 通过 `ctypes` 调用 C++ `.so` 动态库；
- 在 `.so` 内部完成 prefill 和逐 token decode；
- 打印每个阶段的耗时，方便定位性能瓶颈；
- 后续逐步把 reference 计算替换成 AscendC / ACL 高性能算子。

当前版本已经可以跑通完整模型的 `.so` 推理路径，能够生成中文文本，并支持 KV cache、权重缓存、线程数配置、profile 开关等功能。

## 当前 Ascend 推理架构

当前 direct AscendCL 路径大致如下：

```text
用户 prompt
  -> Python tokenizer
  -> Python ctypes
  -> build/libllm_ascend.so
  -> AscendCL / CANN runtime
  -> Ascend HBM 权重与缓存
  -> prefill
  -> decode one token
  -> lm_head argmax
  -> tokenizer decode 输出文本
```

Python 层入口是：

```text
python_infer.py
```

Ascend C++ 动态库源码是：

```text
src/llm_ascend_lib.cpp
```

编译后生成：

```text
build/libllm_ascend.so
```

## 环境要求

推荐在 Ascend 910 机器或云平台 Ascend 容器中运行。环境中需要有：

- Ascend 910 NPU；
- CANN / AscendCL；
- `g++`；
- Python 3；
- `tokenizers`；
- `modelscope` 或 `huggingface_hub`；
- 足够的磁盘空间和 HBM 显存。

可以先检查 NPU：

```bash
npu-smi info
```

如果看到类似下面的信息，说明设备可见：

```text
Name: Ascend910
HBM-Usage: ... / 65536 MB
Health: OK
```

## 获取项目代码

克隆项目：

```bash
git clone -b Ascend https://github.com/luogantt/LLM-inference-engine.git
cd LLM-inference-engine
```

如果已经有仓库，直接拉取 Ascend 分支：

```bash
cd ~/LLM-inference-engine
git pull --ff-only origin Ascend
```

## 下载模型

项目默认测试模型是：

```text
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

仓库提供了 `download_model.py`，可以下载 safetensors 格式模型。国内网络通常建议使用 ModelScope：

```bash
pip install -U modelscope

python download_model.py \
  --source modelscope \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir ./deepseek-r1-7b
```

如果使用 HuggingFace：

```bash
pip install -U huggingface_hub

python download_model.py \
  --source huggingface \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir ./deepseek-r1-7b
```

下载完成后，模型目录中应包含：

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors.index.json
model-00001-of-000002.safetensors
model-00002-of-000002.safetensors
```

如果模型目录不是 `./deepseek-r1-7b`，推理时把 `--model` 参数改成自己的模型路径即可。

## 安装 Python 依赖

direct `.so` 推理路径建议使用 `tokenizers` 后端，这样可以避免在同一个进程中引入 `torch_npu` 和 transformers：

```bash
pip install -U tokenizers
```

如果只运行 `.so` 推理，不需要安装 PyTorch、transformers 或 torch_npu。

## 编译 Ascend 动态库

在 Ascend 机器上执行：

```bash
cd ~/LLM-inference-engine

make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-ascend ASCEND_HOME=/usr/local/Ascend/cann-8.5.1
```

编译完成后会生成：

```text
build/libllm_ascend.so
```

如果机器上的 CANN 有标准 `latest` 软链接，也可以使用默认路径：

```bash
make -f Makefile.cuda_lib lib-ascend
```

## 完整 `.so` 推理流程

下面是一套当前推荐的 Ascend `.so` 推理命令。它使用 direct AscendCL 动态库，不走 torch。

```bash
cd ~/LLM-inference-engine

git pull --ff-only origin Ascend

make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-ascend ASCEND_HOME=/usr/local/Ascend/cann-8.5.1

mkdir -p ~/ascend/log

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

export ASCEND_REF_LINEAR_THREADS=16
export ASCEND_REF_ATTN_LINEAR_THREADS=8
export ASCEND_REF_MLP_THREADS=16
export ASCEND_REF_DOWN_THREADS=16
export ASCEND_LM_HEAD_THREADS=16

export ASCEND_REF_PROFILE_LAYERS=0

python python_infer.py \
  --model ./deepseek-r1-7b \
  --lib ./build/libllm_ascend.so \
  --prompt "黑格尔的哲学思想可以概括为" \
  --max-new-tokens 128 \
  --max-seq 800 \
  --tokenizer-backend tokenizers \
  --no-chat-template
```

其中比较重要的参数含义如下：

```text
ASCEND_VISIBLE_DEVICES=4
```

表示使用物理 NPU 4。如果平台已经把设备映射成逻辑 0，则程序内部使用：

```text
ASCEND_DEVICE_ID=0
```

```text
ASCEND_LOAD_WEIGHTS=all
```

表示把完整模型权重加载到 Ascend HBM。

```text
ASCEND_DIRECT_DECODE=all_layers_ref
```

表示启用完整 Transformer 层的 reference decode 路径。

```text
ASCEND_REF_KV_CACHE=1
```

表示启用 KV cache，后续 token 只处理新增 token。

```text
ASCEND_REF_CACHE_WEIGHTS=1
```

表示缓存 reference 阶段需要使用的权重，减少重复转换和加载开销。

```text
ASCEND_HOST_RAW_CACHE=0
```

表示不缓存 host raw 权重，避免内存占用过大导致进程被系统 kill。

## 运行结果示例

运行后会看到类似日志：

```text
[Python] backend: ascend-direct-acl
[Ascend] config loaded: layers=28, hidden=3584, heads=28, kv_heads=4, intermediate=18944, vocab=152064
[Ascend][time] requested weights loaded, mode=all, count=339

========== ascend-direct-acl prefill ==========
[Ascend][time] embedding lookup D2D finished ...

========== ascend-direct-acl decode ==========
[0] token=101124, text_so_far='三个'
[1] token=99558, text_so_far='三个主要'
[2] token=99659, text_so_far='三个主要部分'
...
```

最终会输出：

```text
========== generated text ==========
三个主要部分：辩证法、逻辑学和德意志唯心论。
```

如果把 `--max-new-tokens` 设置得更大，就可以生成更长的段落。

## 与 torch_npu 推理的区别

项目里也有一个 `python_infer_ascend.py`，它是基于 `torch_npu + transformers` 的验证脚本。这个脚本适合快速确认模型、tokenizer 和 Ascend 环境是否正常。

但是 direct `.so` 路径不同：

```text
python_infer.py + build/libllm_ascend.so
```

这条路径的重点是自研推理引擎。它通过 C ABI 动态库直接管理 AscendCL 运行时和模型推理流程，不依赖 torch 执行模型 forward。

简单来说：

- `python_infer_ascend.py`：框架验证路径，使用 torch_npu；
- `python_infer.py --lib ./build/libllm_ascend.so`：自研 `.so` 推理路径，不走 torch。

## 当前性能情况

当前 `all_layers_ref` 还是 reference 实现，主要目标是先跑通完整模型和验证正确性。因此它不是最终高性能版本。

从日志可以看到，首 token 通常比较慢，因为需要处理 prompt、加载或转换 reference 权重，并完成完整 prefill。后续 token 会明显快很多，因为启用了 KV cache，只需要处理新增 token。

目前主要耗时集中在：

- Q/K/V/O 线性层；
- MLP 的 gate/up/down 投影；
- lm_head argmax；
- host reference 计算和 H2D / D2H 数据搬运。

这些日志对于后续优化非常重要，因为它能告诉我们真正慢在哪里。

## 常见问题

### 1. 为什么程序卡在 decode 前面？

首次运行时可能正在加载完整权重到 HBM，或者正在进行 reference 权重转换。7B 模型权重较大，第一次启动需要等待几十秒是正常的。

可以观察日志里是否出现：

```text
[Ascend][time] requested weights loaded, mode=all
```

如果已经进入：

```text
========== ascend-direct-acl decode ==========
```

首 token 仍然可能比较慢，等第一条：

```text
[0] token=...
```

出来后，后续 token 会更快。

### 2. 为什么进程被 Killed？

一般是内存占用过高。建议使用：

```bash
export ASCEND_HOST_RAW_CACHE=0
export ASCEND_WEIGHT_LOAD_LOG=0
export ASCEND_REF_CACHE_LOG=0
```

同时不要把线程数开得过大。

### 3. 为什么输出像是在思考？

DeepSeek-R1 系列模型本身是 reasoning 模型，默认容易输出思考过程。如果想让它更直接，可以使用更明确的 prompt，例如：

```text
请直接给出最终答案，不要展示思考过程，用一段完整中文介绍黑格尔的哲学思想。
```

也可以使用 `--no-chat-template`，让输入更接近原始补全文本。

### 4. 为什么不用 transformers？

这个项目的目标是研究和实现底层推理引擎。transformers 很适合快速使用模型，但它隐藏了很多底层细节。这里希望把权重加载、内存管理、decode、KV cache、算子耗时等都拆开，方便学习和优化。

## 后续优化方向

目前 reference 路径已经能完整跑通模型，后续优化重点会集中在把热点计算迁移到 Ascend 侧：

1. 将 RMSNorm 从 host reference 替换为 AscendC / ACL 实现；
2. 将 Q/K/V/O 线性层替换为 Ascend 高性能算子；
3. 将 MLP 的 gate/up/down 投影迁移到 Ascend 侧；
4. 将 lm_head argmax 从 host 扫描改成设备侧 kernel；
5. 减少 H2D / D2H 数据搬运；
6. 继续优化 KV cache；
7. 进一步压缩首 token 延迟；
8. 后续尝试量化、算子融合和更高效的 decode pipeline。

## 总结

这个 Ascend 推理项目已经从“能否在 Ascend 上加载模型”推进到了“能否通过自研 `.so` 跑完整大模型推理”。它不是一个黑盒调用脚本，而是一个可以逐层拆解、逐步优化的大模型推理实验平台。

对于想学习大模型底层推理、AscendCL / CANN、safetensors 权重加载、KV cache、decode 性能分析的人来说，这个项目很适合作为实践入口。

项目地址：

[https://github.com/luogantt/LLM-inference-engine](https://github.com/luogantt/LLM-inference-engine)
