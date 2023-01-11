#!/bin/bash
echo '----------------------------------------'
dataset='weather_2'
model='efficientnet_b0'
path='/media/igofed/SSD_2T/DATASETS/'
pth='/home/igofed/LiU/AI4Climate_version_updated/AI4Climate_/outputs/efficientnet_b0_weather_2_last_checkpoint.bin'

echo '[PARSER]  Source Data Folder  :' $'\t' $path
echo '[PARSER]              DATA    :' $'\t' $dataset
echo '[PARSER]  CNN model           :' $'\t' $model
echo '[PARSER]  Path to trained MDL :' $'\t' $pth

python3 src/inference.py --dataset $dataset --model $model --path $path --pth $pth #--split