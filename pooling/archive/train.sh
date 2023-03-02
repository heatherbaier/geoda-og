#!/bin/bash
clear
COUNTRIES=$(ls ../../imagery/)
GPUS="567"
GPU_FILE=gpus_in_use.txt

rm $GPU_FILE
touch $GPU_FILE

python3 train_all.py --gpus_avail $GPUS --gpu_file $GPU_FILE

