## Configuration
import torch
import os
import numpy as np
from skimage import io
from com.colors import COLOR
from PIL import Image
from com.common_packages import check_if_dir_existed
from com.common_packages import check_if_file_existed
import pandas as pd
import matplotlib.pyplot as plt
import os

class CFG:
    # number of epoch to train model
    epochs =20 
    # learning rate (How NN updates the gradient)
    lr = 0.001 
     # 16 images in one batch
    batch_size = 16
     #(The model imported from timm)
    model_name = 'tf_efficientnet_b4_ns'
    # Crop all images to be 224x224 after resizing
    img_crop = 224
     # Resize all images to be 256x256 
    img_size = 256
    
    train_path = '/media/igofed/SSD_1T/AI4CI/chest_xray/train'
    validate_path = '/media/igofed/SSD_1T/AI4CI/chest_xray/val'
    test_path = '/media/igofed/SSD_1T/AI4CI/chest_xray/test'
    class_name = ['No-Flooded Image', 'Flooded Image']

class DS:
    train = []
    validation = []
    test = []




def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(COLOR.BBlue, 'my device :', device, COLOR.END)
    return device



class Flood_NonFlood_dataset(torch.utils.data.Dataset):
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """        
        self.root_dir = root_dir
        self.transfrom = transform
        # Read the csv file
        csv_file = os.path.join(root_dir, csv_file)
        self.annotations = pd.read_csv(csv_file)
        self.img_names = []
        self.labels = []
        for i in range(len(self.annotations)):
            annotations = self.annotations.iloc[i,:]
            img_name = f'{annotations[1]}_{int(annotations[2]):04d}_{annotations[4]}.png'
            label = int(annotations[4])
            self.img_names.append(img_name)
            self.labels.append(label)

    def __len__(self):
        return  len(self.img_names)

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()
        
        img_dir = os.path.join(self.root_dir, "image")
        img_path = os.path.join(img_dir, self.img_names[index])
        image = io.imread(img_path)
        image = Image.fromarray(image)
        label = self.labels[index]

        if self.transfrom:
            image = self.transfrom(image)

        return {'image': image, 'label': label}    
