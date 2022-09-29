#!/bin/bash


source='/home/igofed/LiU/AI4Climate_/temp_2022_09_28_20_14_02/image/'
csv='/home/igofed/LiU/AI4Climate_/temp_2022_09_28_20_14_02/2022_09_28_20_14_02.csv' 
echo 'Source Data Folder  :' $'\t' $source
echo 'CSV filem           :' $'\t' $csv
python3 src/random_plot.py -source $source -csv $csv 