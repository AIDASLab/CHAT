#!/bin/bash

# if [ ! -d "venv" ]; then
#     python3.10 -m venv venv
# fi
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O anaconda.sh
bash anaconda.sh -b -p $HOME/anaconda3
echo 'export PATH=$HOME/anaconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
conda --version
# 1) PRE-SETUP
# source ./setup/pre_setup.sh

# source ~/.bashrc
# source ./venv/bin/activate

# 2) INSTALL DEPENDENCIES
conda create -n conda_env python=3.10.6 pytorch faiss-cpu=1.10.0 -c pytorch -c conda-forge -y
conda activate conda_env
pip3 install -r requirements.txt

# 3) INSTALL LIBRARIES
# source ./setup/setup_hnswlib.sh
# source ./setup/setup_faiss.sh
pip3 install hnswlib

# TODO : SET UP ENVIRONMENT VARIABLES
ROOT_DIR=$(pwd)
AUTO_TUNER_DIR=$ROOT_DIR/auto_tuner
RESULTS_DIR=$ROOT_DIR/results
DATA_DIR=$ROOT_DIR/data
echo "export ROOT_DIR=$ROOT_DIR" >> ~/.bashrc
echo "export AUTO_TUNER_DIR=$AUTO_TUNER_DIR" >> ~/.bashrc
echo "export RESULTS_DIR=$RESULTS_DIR" >> ~/.bashrc
echo "export DATA_DIR=$DATA_DIR" >> ~/.bashrc
echo "export OMP_WAIT_POLICY=PASSIVE" >> ~/.bashrc
export ROOT_DIR=$ROOT_DIR
export AUTO_TUNER_DIR=$AUTO_TUNER_DIR
export RESULTS_DIR=$RESULTS_DIR
export DATA_DIR=$DATA_DIR
export OMP_WAIT_POLICY=PASSIVE
echo "export PYTHONPATH=$AUTO_TUNER_DIR" >> ~/.bashrc
export PYTHONPATH=$PYTHONPATH