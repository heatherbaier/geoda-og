#!/bin/bash
clear
COUNTRIES=$(ls ../../imagery/)
GPUS=("5" "6" "7")
GPU_FILE=gpus_in_use.txt

rm $GPU_FILE
touch $GPU_FILE


for i in ${COUNTRIES[@]}; do

    FOLDER_NAME="${i}_v1"
    echo $FOLDER_NAME
        
    if [ -s $GPU_FILE ]
    then
        OUT=$(awk '{print}' $GPU_FILE)
        echo $OUT
    else
        echo ${GPUS[0]} | tee -a $GPU_FILE
    fi
    
done