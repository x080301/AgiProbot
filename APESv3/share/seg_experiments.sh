#!/bin/bash
experiments_path="./configs/experiments/1"

for config_experiment in "$experiments_path"/*
do
    python train_shapenet.py datasets=shapenet_AnTao350M usr_config=$config_experiment
    python test_shapenet.py datasets=shapenet_AnTao350M usr_config=$config_experiment
done