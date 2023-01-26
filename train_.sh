#!/bin/bash
echo '----------------------------------------'
dataset='weather_2'
model='densenet161'
path='/mnt/symphony/igofe52_/ai4ci'


echo '[PARSER]  Source Data Folder  :' $'\t' $path
echo '[PARSER]              DATA    :' $'\t' $dataset
echo '[PARSER]  CNN model           :' $'\t' $model

python3 src/train.py --dataset $dataset --model $model --path $path #--split