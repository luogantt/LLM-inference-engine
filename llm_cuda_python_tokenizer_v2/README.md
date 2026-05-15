# Python Tokenizer + CUDA Dynamic Library Inference

结构：

```text
Python tokenizer
    ↓ token ids
CUDA shared library: libllm_cuda.so
    ↓ generated token ids
Python detokenizer
    ↓ text
```

这个版本不用 torch / transformers 做 forward，只用 Python tokenizer。CUDA/C++ 负责模型前向和 greedy decode。

## 编译

A800 / A100:

```bash
make lib A=sm_80
```

RTX 4090:

```bash
make lib A=sm_89
```

默认：

```bash
make lib
```

生成：

```text
build/libllm_cuda.so
```

## 运行

```bash
pip install transformers tokenizers sentencepiece
python python_infer.py --model "/home/lg/推理/推理引擎/deepseek-r1-7b" --lib ./build/libllm_cuda.so --prompt "你好 deepseek" --max-new-tokens 16 --max-seq 256
```

## 说明

- CUDA 动态库直接读取 safetensors。
- 输入文本由 Python tokenizer 编码。
- 生成 token id 由 Python tokenizer 解码。
- 当前 CUDA kernel 是 naive 教学版，不用 cuBLAS、不用 Tensor Core、不用 vLLM、不用 llama.cpp。
- BF16/F16 权重会转成 FP32 放 GPU，7B 约需要 30GB+ 显存，推荐 A800/A100 80GB。
