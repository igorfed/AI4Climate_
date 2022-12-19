from turtle import color
import torch
from torchvision import datasets
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import sys
import os
import glob
import numpy as np
import com.config as config
import cv2
import matplotlib.pyplot as plt
#############
from com.colors import COLOR
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from typing import Tuple, List, Dict
from PIL import Image
from com.common_packages import check_if_dir_existed
from com.common_packages import check_if_file_existed
import argparse
import com.utils as utils

import random


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=False, help="path to the dataset")
    parser.add_argument("-m", "--mode", type=str, required=False, help="train  / valid / test")
    parser.add_argument("-p", "--plot", action='store_true', help="Random plot batch of images")
    parser.add_argument("-s", "--save", action='store_true', help="Save figure")
    parser.add_argument("-b", "--balance", action='store_true', help="Balance classes in source dataset")
    return vars(parser.parse_args())

class CustomDataset(torch.utils.data.Dataset):

    """Define training/valid dataset loading methods.
    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): Image size.
        image rotate (int)
        mean value list [float, float, float]
        std value list [float, float, float]
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """
    def __init__(self, 
                image_dir   : str, 
                image_size  : int,  # from config CFG
                rotate      : int,  # rotation int
                mean        : list, # mean value List[int, int, int]
                std         : list, # std value List[int, int, int]
                mode        : str   # 'train' or 'test' or 'valid'
                ) -> None:
        super(CustomDataset, self).__init__()

        self.paths = list(glob.glob(f"{image_dir}/*/*"))
        self.image_size = image_size
        self.rotate = rotate
        self.mode = mode
        self.delimiter = config.delimiter
        self.mean = mean
        self.std = std
        self.img_extension = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
        print(COLOR.Green, f"[INFO]: Custom Loader for :\t {self.mode} data", COLOR.END)
        self.classes, self.class_to_idx = self.find_classes(image_dir)
        print(COLOR.Green + 'Classes', self.class_to_idx ,  COLOR.END)
        self.transform_dataset()

    def find_classes(
                    self, 
                    directory: str
                    ) -> Tuple[List[str], Dict[str, int]]:

        """Finds the class folders in a dataset.
         directory: Root directory path with image dir
         directory/
            ├── class_0
            │   ├── xxx.png
            │   ├── xxy.png
            │   └── ...
            └── class_y
                ├── xx0.png
                ├── xx1.png
                └── ...
        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        ATTENTION configuration of files saved only for 2 classes has Water or not
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
   
   
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        #image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        #image = cv2.cvtColor(cv2.imread(image_path))
        image = cv2.imread(image_path)
        return Image.fromarray(image)


    def transform_dataset(self):
        self.transform = []
        if self.mode == 'train':
            self.transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomRotation(degrees=(-self.rotate, self.rotate)), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.mean, 
                        std=self.std)
            ])
        elif self.mode == 'valid':
            self.transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.mean, 
                        std=self.std)
            ])
        elif self.mode == 'test':
            self.transform=transforms.Compose([
                                transforms.Resize((self.image_size,self.image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=self.mean, 
                                    std=self.std)
            ])

        else:
            raise ValueError(COLOR.Red, "Unsupported data read type. Please use `train` or `valid` or `test`", COLOR.END)

    def __getitem__(self, index: int)-> Tuple[torch.Tensor, int]:

        image_dir, image_name = self.paths[index].split(self.delimiter)[-2:]
        img = self.load_image(index)
        class_idx = self.class_to_idx[image_dir]
        
        return self.transform(img), class_idx

######### Load Custom Dataset #####################################

def load_dataset(data_path, mode, balance_dataset=False):
    """Creates training and testing DataLoaders.

    """
    def prnt(dataset, dataloader, mode):
        print(COLOR.Green)
        print(f"[INFO] : Original {mode} Set Size: \t{len(dataset)}")
        print(f"\tNum. of batches: \t\t{len(dataloader)}")
        print(f"\tNum. of total examples: \t{len(dataloader.dataset)}")
        print(COLOR.END,39*"-")

    def class_balancing(dataset, dataset_path):
        '''
        arguments:
        dataset - results of datasets.ImageFolder
        dataset_path - train_path or valid_path or test path
        '''
        class_weights = []
        for subdir, _, files in os.walk(dataset_path):
            if len(files) > 0:
                class_weights.append(1/len(files))
                print(COLOR.Green, '[INFO] In', subdir, len(files) ,' Images', COLOR.END)
        sample_weights = [0]*len(dataset)
        for i, (_, label) in enumerate(dataset):
            class_weight = class_weights[label]
            sample_weights[i] = class_weight

        return torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


    def check_balance(dataset, name ):
        class_names = dataset.classes
        total_classes = len(class_names)
        total_per_class = [0]*total_classes
        for _, label in dataset:
            total_per_class[label] += 1
        print(f'[INFO]: {name} Set:\t{sum(total_per_class)}')
        for i, classSum in enumerate (total_per_class): 
            print(f'\t[INFO]: {classSum}:\t{class_names[i]} {classSum/sum(total_per_class)*100:0.1f}%')


    dataset = CustomDataset(
                                image_dir=data_path,
                                image_size = config.CFG.img_size,
                                rotate=config.CFG.angle,
                                mean = config.CFG.mean,
                                std = config.CFG.std,
                                mode = mode)
    # Get class names
    check_balance(dataset=dataset, name=mode)                                                
    if mode == 'train' or mode =='valid':
        shuffle = True
    else:
        shuffle = False

    if balance_dataset==True:
        sampler = class_balancing(dataset=dataset, dataset_path=data_path)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=config.CFG.batch_size,
                                                shuffle=False,
                                                sampler=sampler,
                                                num_workers=config.CFG.num_workers)
    else:
        data_loader = torch.utils.data.DataLoader(
                                dataset=dataset,   
                                batch_size=config.CFG.batch_size,
                                num_workers=config.CFG.num_workers,
                                shuffle=shuffle)

    prnt(dataset=dataset, dataloader=data_loader, mode=mode)                                                

    return dataset, data_loader


def main(args):
    
    
    print('Figure save', args['save'])
    print('Mode', args['mode'])
    print('Balance', args['balance'])
    balance =  args['balance']

    if args['mode']=='train':
        train_path = os.path.join(args['dataset'], 'train')
        check_if_dir_existed(train_path, create=False)
        train_set, train_loader = load_dataset(train_path, mode=args['mode'], balance_dataset=balance)
        if args['plot']:
            utils.show_batch_grid(train_loader, balance, args['mode'], figure_save=args['save'])
    
    elif args['mode']=='valid':
        valid_path = os.path.join(args['dataset'], 'valid')
        check_if_dir_existed(valid_path, create=False)
        valid_set, valid_loader = load_dataset(train_path, mode=args['mode'], balance_dataset=balance)
        if args['plot']:
            utils.show_batch_grid(valid_loader, balance, args['mode'], figure_save=args['save'])
    
    elif args['mode']=='test':
        test_path = os.path.join(args['dataset'], 'test')
        check_if_dir_existed(test_path, create=False)
        test_set, test_loader = load_dataset(test_path, mode=args['mode'], balance_dataset=balance)
        if args['plot']:
            utils.show_batch_grid(test_loader, balance, args['mode'], figure_save=args['save'])
    
    #utils.show_batch_grid(train_loader, 'Valid')


    plt.show()

if __name__ == '__main__':
    args = arg_parser()
    print(args)
    check_if_dir_existed(args['dataset'])
    main(args)

