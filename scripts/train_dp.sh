#!/bin/bash

NSYS_CMD="nsys profile --force-overwrite true -o nsight_logs/1_gpu_${TIMESTAMP}.nsys-rep"
CUDA_VISIBLE_DEVICES=0 ${NSYS_CMD} python train_cifar.py --data dataset/cifar10 --ckpt runs/default/${TIMESTAMP}

NSYS_CMD="nsys profile --force-overwrite true -o nsight_logs/2_gpu_${TIMESTAMP}.nsys-rep"
CUDA_VISIBLE_DEVICES=0,1 ${NSYS_CMD} python train_cifar.py --dp --data dataset/cifar10 --ckpt runs/dp_2/${TIMESTAMP}
