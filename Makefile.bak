NVCC ?= nvcc
A ?= sm_80

COMMON_FLAGS = -O2 -std=c++17 -lineinfo

.PHONY: all toy deepseek7b clean

all: toy deepseek7b

toy:
	$(NVCC) $(COMMON_FLAGS) -arch=$(A) src/toy_cuda_infer.cu -o toy_cuda_infer

deepseek7b:
	$(NVCC) $(COMMON_FLAGS) -arch=$(A) src/deepseek7b_token_cuda_infer.cu -o deepseek7b_token_cuda_infer

clean:
	rm -f toy_cuda_infer deepseek7b_token_cuda_infer
