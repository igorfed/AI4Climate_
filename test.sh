#!/bin/bash
echo '----------------------------------------'
dataset='weather_2'
model='efficientnet_b0'
#model='densenet121'
path='/media/igofed/SSD_2T/DATASETS/'
pth='/home/igofed/LiU/AI4Climate_version_updated/AI4Climate_/outputs/efficientnet_b0_weather_2_last_checkpoint.bin'
#pth='/home/igofed/LiU/AI4Climate_version_updated/AI4Climate_/outputs/densenet121_weather_2_last_checkpoint.bin'
plot='True'


echo '[PARSER]  Source Data Folder  :' $'\t' $path
echo '[PARSER]              DATA    :' $'\t' $dataset
echo '[PARSER]  CNN model           :' $'\t' $model
echo '[PARSER]  Path to trained MDL :' $'\t' $pth
echo '[PARSER]  Do you want to plot :' $'\t' $plot

python3 src/inference_test.py --dataset $dataset --model $model --path $path --pth $pth --plot $plot 
#python3 src/inference1.py --dataset $dataset --model $model --path $path --pth $pth --plot $plot 