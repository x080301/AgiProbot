#!/usr/bin/env bash
# command to install this enviroment: source init.sh

# install miniconda3 if not installed yet.
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh
#source ~/.bashrc

export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0"   # a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1
module purge
module load cuda/11.3.1
module load gcc/7.5.0
# make sure local cuda version is 11.1

# download openpoints
# git submodule add git@github.com:guochengqian/openpoints.git
git submodule update --init --recursive
git submodule update --remote --merge # update to the latest version

# install PyTorch
conda deactivate
conda env remove --name openpoints
conda create -n openpoints -y python=3.7 numpy=1.20 numba
conda activate openpoints

conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch -c nvidia

