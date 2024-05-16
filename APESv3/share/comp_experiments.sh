#!/bin/bash
experiments_path="./configs/experiments/2"

for config_experiment in "$experiments_path"/*
do
    python train_comparison.py datasets=modelnet_Alignment1024 usr_config=$config_experiment
    python test_comparison.py datasets=modelnet_Alignment1024 usr_config=$config_experiment
done