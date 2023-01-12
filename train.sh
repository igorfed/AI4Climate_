#!/bin/bash
echo '----------------------------------------'
dataset='weather_2'
model='efficientnet_b0'
path='/media/igofed/SSD_2T/DATASETS/'


echo '[PARSER]  Source Data Folder  :' $'\t' $path
echo '[PARSER]              DATA    :' $'\t' $dataset
echo '[PARSER]  CNN model           :' $'\t' $model

python3 src/train.py --dataset $dataset --model $model --path $path #--split