#!/bin/bash

# install CUDA Toolkit v9.0
# instructions from https://developer.nvidia.com/cuda-downloads (linux -> x86_64 -> Ubuntu -> 16.04 -> deb)
CUDA_REPO_PKG="cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb"
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/${CUDA_REPO_PKG}
dpkg -i ${CUDA_REPO_PKG}
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
apt-get update
apt-get -y install cuda-9-0

CUDA_PATCH1="cuda-repo-ubuntu1604-9-0-local-cublas-performance-update_1.0-1_amd64-deb"
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/1/${CUDA_PATCH1}
dpkg -i ${CUDA_PATCH1}
apt-get update

# install cuDNN v7.0
CUDNN_PKG="libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb"
wget https://github.com/ashokpant/cudnn_archive/raw/master/v7.0/${CUDNN_PKG}
dpkg -i ${CUDNN_PKG}
apt-get update

# install NVIDIA CUDA Profile Tools Interface ( libcupti-dev v9.0)
apt-get install cuda-command-line-tools-9-0

# set environment variables
export PATH=${PATH}:/usr/local/cuda-9.0/bin
export CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-9.0
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
