#!/bin/bash
experiments_path="./configs/experiments/0"

for config_experiment in "$experiments_path"/*
do
    python train_modelnet.py usr_config=$config_experiment
    python test_modelnet.py usr_config=$config_experiment
done