#!/bin/bash
echo '----------------------------------------'
dataset='/media/igofed/SSD_2T/DATASETS/weather_3'
model='densenet121'



echo '[PARSER]  Source Data Folder  :' $'\t' $dataset
echo '[PARSER]  CNN model           :' $'\t' $model

python3 src/train.py --dataset $dataset --model $model #--split