#!/bin/bash

set -ex

rm -rf logs
export SUDNN_KERNEL_CACHE_CAPACITY=30000
export SUDNN_KERNEL_CACHE_EXCLUDE_UID=1
export SUDNN_KERNEL_CACHE_FOLDER=./kernel_cache # SUDNN算子编译缓存路径

export SUDNN_KERNEL_CACHE_DISK_LEVEL=3
export SUDNN_KERNEL_CACHE_MAX_SIZE_MB=10240 # SUDNN算子编译缓存大小上限

if [ ! -d $SUDNN_KERNEL_CACHE_FOLDER ]; then
    mkdir -p $SUDNN_KERNEL_CACHE_FOLDER
fi


# # ------------------------------
# # common enviromental variables
# # ------------------------------
export PYTORCH_SUPA_ALLOC_CONF=max_split_size_mb:30 # pytorch显存分配器的最大拆分块大小设置

# # br-pytorch envs
export BRTB_DISABLE_ZERO_REORDER=1      # 关闭reorder清零
export BRTB_DISABLE_ZERO_OUTPUT_NUMA=1  # 关闭output numa格式清零
export BRTB_DISABLE_ZERO_OUTPUT_UMA=1   # 关闭output uma格式清零
export BRTB_DISABLE_ZERO_WS=1           # 关闭workspace清零
export BRTB_ENABLE_NCDHW=1              # 支持NCDHW
export BRTB_ENABLE_EAGER_ADV_API=1      # SUEAGER高性能计算算子开关
export BRTB_ENABLE_FORCE_UMA=1          # 强制使用UMA
export BRTB_ENABLE_FORCE_EAGER_CONV2D=1 # 使用SUEAGER CONV2D算子
export BRTB_DISABLE_L2_FLUSH=1          # 关闭L2 flush功能
export BRTB_ENABLE_SUBLAS_API=1         # SUBLAS API启动开关
export BRTB_ENABLE_WEIGHT_BYPASS=1
export BRTB_ENABLE_REGISTER_BEFORE_D2H=1

export OMP_NUM_THREADS=16   # 指定并行程序运行时可使用的最大线程数

python3 api_inference.py
