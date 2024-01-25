# APESv2

<p>
<a href="https://arxiv.org/pdf/2302.14673.pdf">
    <img src="https://img.shields.io/badge/PDF-arXiv-brightgreen" /></a>
<a href="https://junweizheng93.github.io/publications/APES/APES.html">
    <img src="https://img.shields.io/badge/Project-Homepage-red" /></a>
<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange" /></a>
</p>

# APESv1
https://github.com/JunweiZheng93/APES#apes-attention-based-point-cloud-edge-sampling


# Setup
Python version: 3.9
```bash
conda create -n apesv2 python=3.9 -y
conda activate apesv2
```
To install all dependencies, enter:
```bash
pip install -r requirements.txt
```
Be careful, please install PyTorch using this command:
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch -y
```
Install Pytorch3D:
```bash
conda install -c fvcore -c iopath -c conda-forge fvcore=0.1.5 iopath=0.1.9 -y
conda install -c pytorch3d pytorch3d=0.7.0 -y
```


# How to run

## Classification
```bash
bash cls_experiments.sh
``` 
or
When using `AnTao420M` dataset, run: 
train:
```bash
python train_modelnet.py datasets=modelnet_AnTao350M usr_config=YOUR/USR/CONFIG/PATH
```
test:
```bash
python test_modelnet.py datasets=modelnet_AnTao350M usr_config=YOUR/USR/CONFIG/PATH
```

## Segmentation
```bash
bash seg_experiments.sh
``` 
or
When using `AnTao350M` dataset, run:  
train:
```bash
python train_shapenet.py datasets=shapenet_AnTao350M usr_config=YOUR/USR/CONFIG/PATH
```
test:
```bash
python test_shapenet.py datasets=shapenet_AnTao350M usr_config=YOUR/USR/CONFIG/PATH
```

When using `Yi650M` dataset, run: 
train: 
```bash
python train_shapenet.py datasets=shapenet_Yi650M usr_config=YOUR/USR/CONFIG/PATH
```
test: 
```bash
python test_shapenet.py datasets=shapenet_Yi650M usr_config=YOUR/USR/CONFIG/PATH
```

## Comparison
```bash
bash comp_experiments.sh
``` 
or
train:
```bash
python train_comparison.py datasets=modelnet_Alignment1024 usr_config=YOUR/USR/CONFIG/PATH
```
test:
```bash
python test_comparison.py datasets=modelnet_Alignment1024 usr_config=YOUR/USR/CONFIG/PATH
```
