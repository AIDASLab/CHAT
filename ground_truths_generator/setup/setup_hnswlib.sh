#!/bin/bash

pip3 install pybind11 numpy setuptools

TARGET_DIR="libs"
mkdir -p $TARGET_DIR

if [ ! -d "$TARGET_DIR/hnswlib" ]; then
    git clone https://github.com/nmslib/hnswlib.git $TARGET_DIR/hnswlib
fi
pushd $TARGET_DIR/hnswlib
python3 setup.py install
popd