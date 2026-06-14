#!/bin/bash

# This script is a part of the setup process for the hnsw-auto-tuner project.
# It is intended to be run on a old Ubuntu 18.04 and AMD machine

# Issue1: CMake
# Due to Ubuntu 18.04
if [ ! -d "$HOME/local/cmake" ]; then
    wget https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6-linux-x86_64.tar.gz
    tar -xvzf cmake-3.27.6-linux-x86_64.tar.gz
    mv cmake-3.27.6-linux-x86_64 $HOME/local/cmake
    echo 'export PATH=$HOME/local/cmake/bin:$PATH' >> ./venv/bin/activate
    source ~/.bashrc
    cmake --version
fi

# Issue2: CUDA
# Due to Ubuntu 18.04
CUDA_VERSION="11.8.0"
CUDA_INSTALL_DIR="$HOME/local/cuda"

if [ ! -d "$CUDA_INSTALL_DIR" ]; then
    wget https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/cuda_${CUDA_VERSION}_520.61.05_linux.run -O cuda_installer.run
    sh cuda_installer.run --silent --toolkit --installpath=$CUDA_INSTALL_DIR
    echo 'export PATH=$HOME/local/cuda/bin:$PATH' >> ./venv/bin/activate
    echo 'export LD_LIBRARY_PATH=$HOME/local/cuda/lib64:$LD_LIBRARY_PATH' >> ./venv/bin/activate
    echo 'export LD_LIBRARY_PATH=$HOME/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    nvcc --version
    GPU_ARCH="86"
    export CMAKE_CUDA_ARCHITECTURES=$GPU_ARCH
    echo 'export CMAKE_CUDA_ARCHITECTURES=86' >> ./venv/bin/activate
    echo 'export CMAKE_CUDA_ARCHITECTURES=86' >> ~/.bashrc
fi

# Issue3: OpenBLAS
# Due to AMD CPU
OPENBLAS_DIR="$HOME/local/openblas"

if [ ! -d "$OPENBLAS_DIR" ]; then
    mkdir -p $HOME/local
    pushd $HOME/local
    git clone https://github.com/xianyi/OpenBLAS.git
    pushd OpenBLAS
    make -j$(nproc)
    make install PREFIX=$OPENBLAS_DIR
    popd
    popd
    echo 'export OPENBLASROOT=$HOME/local/openblas' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$OPENBLASROOT/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo 'export LIBRARY_PATH=$OPENBLASROOT/lib:$LIBRARY_PATH' >> ~/.bashrc
    echo 'export CPATH=$OPENBLASROOT/include:$CPATH' >> ~/.bashrc
    echo 'export OPENBLASROOT=$HOME/local/openblas' >> ./venv/bin/activate
    echo 'export LD_LIBRARY_PATH=$OPENBLASROOT/lib:$LD_LIBRARY_PATH' >> ./venv/bin/activate
    echo 'export LIBRARY_PATH=$OPENBLASROOT/lib:$LIBRARY_PATH' >> ./venv/bin/activate
    echo 'export CPATH=$OPENBLASROOT/include:$CPATH' >> ./venv/bin/activate
    source ~/.bashrc
fi
