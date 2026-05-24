# LLM Inference Engine

一个从零实现的单 batch LLM 推理引擎，当前主要面向
`DeepSeek-R1-Distill-Qwen-7B` 的 HuggingFace safetensors 权重目录。

项目目标是尽量少依赖外部推理框架，用 C++ / GPU kernel 手写核心推理路径，便于学习、实验和性能优化。

## Features

- 不依赖 PyTorch 推理图、vLLM、llama.cpp 等外部推理框架。
- 支持 HuggingFace safetensors 模型目录。
- Python 负责 tokenizer 和调用动态库，C++ 动态库负责模型加载和 GPU 推理。
- NVIDIA CUDA 路径输出 `build/libllm_cuda.so`。
- Moore Threads MUSA 路径输出 `build/libllm_musa.so`。

## Supported Model

当前源码里的常量匹配 DeepSeek-R1-Distill-Qwen-7B：

```text
N_LAYERS = 28
HIDDEN = 3584
N_HEADS = 28
N_KV_HEADS = 4
INTERMEDIATE = 18944
VOCAB_SIZE = 152064
```

注意：本项目当前加载的是 HuggingFace safetensors 目录，不加载 GGUF 文件。
如果你手里是 `DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf`，请用 llama.cpp。

## NVIDIA CUDA

在 NVIDIA GPU 上编译：

```bash
make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib A=sm_80
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python python_infer.py \
  --model /data/project/deepseek-r1-7b \
  --lib ./build/libllm_cuda.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 512 \
  --max-seq 800
```

Jetson AGX Orin 示例：

```bash
make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib A=sm_87
```

## Moore Threads MUSA

`moore` 分支提供了摩尔线程 MUSA 的第一版适配，已在 AutoDL 的 MTT S4000 环境上验证过基础编译和推理路径。

已验证环境：

```text
GPU: MTT S4000
Memory: 49152 MiB
Driver: 2.7.0
MUSA Toolkit: 3.1.0
mcc: /usr/local/musa/bin/mcc
```

检查设备：

```bash
mthreads-gmi
musa_version_query
ls /usr/local/musa
```

下载 safetensors 模型，国内源推荐 ModelScope：

```bash
pip install -U modelscope

modelscope download \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local_dir /root/autodl-tmp/deepseek-r1-7b
```

编译 MUSA 动态库：

```bash
export PATH=/usr/local/musa/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH

make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-musa
```

运行：

```bash
MUSA_VISIBLE_DEVICES=0 python python_infer.py \
  --model /root/autodl-tmp/deepseek-r1-7b \
  --lib ./build/libllm_musa.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 128 \
  --max-seq 800
```

查看 GPU 状态：

```bash
watch -n 1 mthreads-gmi
```

如果显存和 GPU 利用率一直是 0，说明程序没有走到 MUSA 设备路径。

## Moore Threads Notes

- MUSA 不是 NVIDIA CUDA，也不是 Huawei Ascend。
- 这个分支使用 `mcc -x musa -mtgpu` 编译。
- MUSA 构建使用原生头文件 `musa_runtime.h` 和 `musa_fp16.h`。
- CUDA runtime 调用在源码中映射到 MUSA runtime，例如 `musaMalloc`、`musaMemcpy`、`musaMemset`、`musaDeviceSynchronize`。
- WMMA 路径在 MUSA 构建中关闭，因为 `nvcuda::wmma` 是 NVIDIA 专用路径。
- 当前 MUSA 版本是兼容优先，不是最终优化版。

## Main Files

```text
src/llm_cuda_lib.cu          推理核心，CUDA/MUSA 共用源码
python_infer.py              Python 调用入口
Makefile.cuda_lib            动态库编译入口
MOORE_MUSA.md                摩尔线程 MUSA 详细说明
log.txt                      性能日志
```

## Troubleshooting

如果 `mcc` 找不到：

```bash
export PATH=/usr/local/musa/bin:$PATH
```

如果运行时找不到 `libmusart.so`：

```bash
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH
```

如果 tokenizer 报 PyTorch 版本提示，只要后续能加载 tokenizer，通常可以忽略；本项目推理不依赖 PyTorch 模型执行。

如果你在摩尔线程机器上误用了 NVIDIA 编译命令：

```bash
make -f Makefile.cuda_lib lib A=sm_80
```

请改用：

```bash
make -f Makefile.cuda_lib lib-musa
```
