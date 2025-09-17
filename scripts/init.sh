#!/bin/bash

mkdir -m 777 -p dataset
mkdir -m 777 -p slurm_logs
mkdir -m 777 -p nsight_logs
mkdir -m 777 -p runs
mkdir -m 777 -p runs/default
mkdir -m 777 -p runs/dp_2
mkdir -m 777 -p runs/dp_4

pip install -r requirements.txt

python my_lib/init_dataset.py --seed 42 --dataset_dir dataset