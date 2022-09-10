#!/bin/bash
source='/media/igofed/SSD_1T/AI4CI/DATA/mydata/european-flood-2013_imgs_small/imgs_small'
flooding='/media/igofed/SSD_1T/AI4CI/EUFlood2013Dataset/eu-flood-dataset/queries/flooding.txt' 
#flooding='/media/igofed/SSD_1T/AI4CI/EUFlood2013Dataset/eu-flood-dataset/queries/pollution.txt'  
#flooding='/media/igofed/SSD_1T/AI4CI/EUFlood2013Dataset/eu-flood-dataset/queries/flooding.txt'  
#irrelevant='/media/igofed/SSD_1T/AI4CI/DATA/mydata/european-flood-2013_imgs_small/irrelevant.txt' 
#metadata='/media/igofed/SSD_1T/AI4CI/DATA/mydata/european-flood-2013_imgs_small/metadata.json' 
out='/media/igofed/SSD_1T/AI4CI/DATA/mydata/euflood2013'
echo 'Source Data Folder  :' $'\t' $source
echo 'Flooding txt file   :' $'\t' $flooding
echo 'Irrelevant txt file :' $'\t' $irrelevant
echo 'Metadata json file  :' $'\t' $metadata
echo 'Output folder       :' $'\t' $out
python3 src/eu_flood_2013/parser.py -source $source -flooding $flooding -out $out #-irrelevant $irrelevant -metadata $metadata 