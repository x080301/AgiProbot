#!/usr/bin/env bash
# command to install this enviroment: source init.sh

# install miniconda3 if not installed yet.
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh
#source ~/.bashrc

export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0"   # a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1
module purge
module load cuda/11.7.0
module load gcc/9.3.0
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

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# install relevant packages
# torch-scatter is a must, and others are optional
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_scatter-2.1.0%2Bpt113cu117-cp37-cp37m-win_amd64.whl


# pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install -r requirements.txt

export CUDA_HOME="C:/rogram Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"
# install cpp extensions, the pointnet++ library
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

# grid_subsampling library. necessary only if interested in S3DIS_sphere
cd subsampling
python setup.py build_ext --inplace
cd ..


# # point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
cd pointops/
python setup.py install
cd ..


# Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
cd chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user
cd ../../../

echo 'end'