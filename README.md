# AI4Climate project


conda env create -f environment2.yml

The repository contains Torch implementation to detect flooded water on the image based on well known EfficientNet and DenseNet architecture achieving at the selected pictures.

## Abstract

The moder analysys of nature disasters such as floods offten sufers from limited data due to a coarse distribution of sensors. The limitation cpuld be alleviated  by using information ontained images of event posted on social network. In this work analysed new dataset of .... images, annotated by our experts regarding their relevance with respect to three tasks:

flooded area = {0: "non-flooded area", 1 : "flooded area"}

ground details = {0:'road', 1:'pavement', 2: 'cycle path​', 3:'paved area', 4:'gravel area', 5:'green area', 6:'riverbank', 7:'other', 8:'gravel road', 9:'walk path​'}

water detail =  {0:'non obstructive', 1:'slightly obstructive​', 2:'obstructive​'}

waterDetails2 = {0: 'less than 2 $m^2$​', 1:'less than 2-5 $m^2$​', 2: 'less than 5-10 $m^2$​​', 3: 'over 10 $m^2$'}

waterDetails3 = {0:'less than 5​ cm', 1:'5-10 cm​', 2:'10-30 cm​​', 3:'over 30 cm'};


## Notes

For a moment it contains only two classes: Class 0 - No water, Class 1 - has water. Soon, it will update for water categories. </br> Repository contains scripts to read the annotation, provide common structure of existed images and split it into the datasets to train and validate results. It also contain configuration file, and models implemented from scratch.

## Table of Contents

1. [Used Datasets](doc/datasets.md)</br>
1. [Test, Train and Validation sets](doc/trainset.md)</br>
1. [Used Models](doc/efficientNet.md)</br>
1. [Results](doc/data.md)</br>
1. [Quick start](doc/gui.md)</br>
