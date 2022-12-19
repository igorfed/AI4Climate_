# AI4Climate project

The repository contains Torch implementation to detect flooded water on the image based on well known EfficientNet and DenseNet architecture achieving at the selected pictures.

## Abstract

The moder analysys of nature disasters such as floods offten sufers from limited data due to a coarse distribution of sensors. The limitation cpuld be alleviated  by using information ontained images of event posted on social network. In this work analysed new dataset of .... images, annotated by our experts regarding their relevance with respect to three tasks:
-determining the flooded area,
-water details - less then 2 $m^2$ 
-water details - less ...

## Notes

For a moment it contains only two classes: Class 0 - No water, Class 1 - has water. Soon, it will update for water categories. </br> Repository contains scripts to read the annotation, provide common structure of existed images and split it into the datasets to train and validate results. It also contain configuration file, and models implemented from scratch.

## Table of Contents

1. [Used Datasets](doc/datasets.md)</br>
1. [Test, Train and Validation sets](doc/trainset.md)</br>
1. [Used Models](doc/efficientNet.md)</br>
1. [Results](doc/data.md)</br>
1. [Quick start](doc/gui.md)</br>
