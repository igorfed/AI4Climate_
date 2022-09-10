#!/bin/bash
source='/media/igofed/SSD_1T/AI4CI/DATA/mydata/euflood2013_flood/out/1'
csv='/media/igofed/SSD_1T/AI4CI/DATA/mydata/euflood2013_flood/out/eu2013.csv' 
echo 'Source Data Folder  :' $'\t' $source
echo 'CSV filem           :' $'\t' $csv
python3 src/eu_flood_2013/random_plot.py -source $source -csv $csv 