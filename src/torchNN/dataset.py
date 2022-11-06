from turtle import color
import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import sys
import os

import argparse
from glob import glob
import random
import numpy as np
from config import arg_parser_dataset
from config import CFG
import matplotlib.pyplot as plt
#############
from com.colors import COLOR
from com.common_packages import check_if_dir_existed
from config import show_image, show_grid, show_image_grid, slice_plot
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

#########Define Transforms#######################################
def get_train_transform(pretrained):
    train_transform = transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.RandomRotation(degrees=(-10,10)), 
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        normalize_transform(pretrained)
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
    return train_transform

def get_test_transform(pretrained):
    test_transform = transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        #transforms.RandomRotation(degrees=(-10,10)), 
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        normalize_transform(pretrained)
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
    return test_transform

def get_valid_transform(pretrained):
    valid_transform = transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.ToTensor(),
        normalize_transform(pretrained)
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.2258]) # Normalize by 3 means 2 STDs of the image net, 3 channel
    ])
    return valid_transform
    
def normalize_transform(pretrained):
    if pretrained: # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    else: # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize

#########Define Dataset#######################################
def get_datasets(train_path, test_path, valid_path, pretrained):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along 
    with the class names.
    """
    train_set=datasets.ImageFolder(train_path,transform=(get_train_transform(pretrained))
    )
    train_set_size = len(train_set)

    test_set=datasets.ImageFolder(test_path,transform=(get_test_transform(pretrained))
    )
    test_set_size = len(test_set)

    valid_set=datasets.ImageFolder(valid_path,transform=(get_valid_transform(pretrained))
    )
    valid_set_size = len(valid_set)


    # Validation transforms

    return train_set, test_set, valid_set


def get_data_loaders(train_set, test_set, valid_set):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(train_set, batch_size=CFG.batch_size,  shuffle=True, num_workers=CFG.num_workers)
    print (40*"-")
    print(COLOR.Green,"[INFO]:Train set Size: \t\t{}".format(len(train_set)))
    print("\tNum. of batches: \t\t{}".format(len(train_loader)))
    print("\tNum. of total examples: \t{}".format(len(train_loader.dataset)))
    print(COLOR.END,39*"-")
    test_loader = DataLoader(test_set, batch_size=CFG.batch_size,  shuffle=True, num_workers=CFG.num_workers)
    print (40*"-")
    print(COLOR.Green,"[INFO]:Test set Size: \t\t\t{}".format(len(test_set)))
    print("\tNum. of batches: \t\t{}".format(len(test_loader)))
    print("\tNum. of total examples: \t{}".format(len(test_loader.dataset)))
    print(COLOR.END,40*"-")

    valid_loader = DataLoader(valid_set, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
    print(COLOR.Green,"[INFO]:Valid set Size: \t\t{}".format(len(valid_set)))
    print("\tNum. of batches: \t\t{}".format(len(valid_loader))) 
    print("\tNum. of total examples: \t{}".format(len(valid_loader.dataset)))
    print(COLOR.END,40*"-")
    return train_loader, test_loader, valid_loader 




def main(args):
    
    train_path = os.path.join(args.dataset, 'train')
    valid_path = os.path.join(args.dataset, 'valid')
    train_set, valid_set = get_datasets(CFG, train_path, valid_path)
    #listdirs(args.dataset)         
    train_loader, valid_loader  = get_data_loaders(train_set, valid_set)
    #dataiter = iter(train_loader)
    #images, labels = dataiter.next()
    
    #out = make_grid(images, nrow =4,  padding=5)
    
    slice_plot(data_loader= train_loader, nrow = 4, ncol= 4, fig_title= "train plot")
    slice_plot(data_loader= valid_loader, nrow = 4, ncol= 4, fig_title= "valid plot")
    #show_image_grid(out, labels)
    #show_grid(out, title = [CFG.class_name[x] for x in labels ])
    plt.show()

if __name__ == '__main__':
    args = arg_parser_dataset()
    check_if_dir_existed(args.dataset)
    main(args)
    print('done')

    
    #train_path = '/home/igofed/LiU/AI4Climate_/weather_1_0/train'
    #validate_path = '/home/igofed/LiU/AI4Climate_/weather_1_0/valid'
    #test_path = '/home/igofed/LiU/AI4Climate_/weather_1_0/test'

