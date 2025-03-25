#!/bin/bash
uv sync --offline
cp debug/utils.py .venv/lib/python3.10/site-packages/transformers/generation/utils.py
cp debug/modeling_llama.py .venv/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py

export LLM_PATH=meta-llama/Llama-2-7b-hf

# 获取/usr/local下所有cuda-目录（末尾可能带/，需要去掉）
cuda_dirs=(/usr/local/cuda-*/)
if [ ${#cuda_dirs[@]} -eq 0 ]; then
    echo "未在 /usr/local 找到任何 cuda 文件夹"
    exit 1
fi

# 去除末尾的斜杠，并利用sort -V按版本排序，选取最后一个作为最新版本
latest_cuda=$(for dir in "${cuda_dirs[@]}"; do echo "${dir%/}"; done | sort -V | tail -n 1)

echo "最新的 CUDA 文件夹为: $latest_cuda"

# 构造新的LD_LIBRARY_PATH
# 依次加入以下路径：
#   1. ${latest_cuda}/lib64/stubs
#   2. ${latest_cuda}/lib64
#   3. ${latest_cuda}/cudnn/lib
export LD_LIBRARY_PATH="${latest_cuda}/lib64/stubs:${latest_cuda}/lib64:${latest_cuda}/cudnn/lib:${LD_LIBRARY_PATH}"

sh train_movielens.sh
