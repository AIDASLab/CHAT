#!/bin/bash
set -e

conda create -n conda_env python=3.10.6 pytorch faiss-cpu=1.10.0 -c pytorch -c conda-forge -y
source ~/.bashrc
echo $(conda info --base)/envs/conda_env/lib/python3.10/site-packages >> venv/lib/python3.10/site-packages/conda.pth

# pip3 install numpy swig

# TARGET_DIR="libs"
# mkdir -p $TARGET_DIR
# if [ ! -d "$TARGET_DIR/faiss" ]; then
#     git clone https://github.com/facebookresearch/faiss.git $TARGET_DIR/faiss
# fi

# pushd $TARGET_DIR/faiss

# cmake -B build . \
#     -DCMAKE_CUDA_ARCHITECTURES=86 \
#     -DFAISS_ENABLE_GPU=OFF \
#     -DFAISS_ENABLE_PYTHON=ON \
#     -DBUILD_TESTING=OFF \
#     -DBUILD_SHARED_LIBS=ON \
#     -DFAISS_OPT_LEVEL=avx2 \
#     -DPython_EXECUTABLE=$(which python3) \
#     -BLA_VENDOR=OpenBLAS \
#     -DMKL_LIBRARIES=$HOME/local/openblas/lib/libopenblas.so

# # make -C build -j$NUM_CORES faiss_avx2
# # make -C build -j$NUM_CORES swigfaiss
# build_dir='LA_VENDOR=OpenBLAS'
# make -C $build_dir -j faiss_avx2
# make -C $build_dir -j swigfaiss
# (cd $build_dir/faiss/python && python3 setup.py install)
# popd
