import os
from tkinter import END

import torch
import torchvision
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import random 
import timm
import torch.nn.functional as F
import sys
from com.colors import COLOR
from com.config import CFG
from skimage import io
import pandas as pd
from com.config import CFG, DS, get_device, Flood_NonFlood_dataset, random_plot
#sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

print(80*"-")
print(COLOR.Green + 'Python VERSION: \t', sys.version, COLOR.END)
print(COLOR.Green + 'pyTorch VERSION:\t', torch.__version__, COLOR.END)
print(COLOR.Green +'torchvision VERSION:\t', torchvision.__version__, COLOR.END)
print(80*"-")
print(COLOR.Green +'CUDNN VERSION:\t\t', torch.backends.cudnn.version(), COLOR.END)
print(COLOR.Green +'Cuda is availabel:\t', torch.cuda.is_available(), COLOR.END)
print(COLOR.Green +'Cuda device number:\t', torch.cuda.current_device(), COLOR.END)
print(COLOR.Green +'Cuda count:\t\t', torch.cuda.device_count(), COLOR.END)
print(COLOR.Green +'Cuda device name:\t', torch.cuda.get_device_name(0), COLOR.END)
print(COLOR.Green +'Torch Cuda Version: \t', torch.version.cuda,  COLOR.END)
print(80*"-")
plt.ion()   # interactive mode
from com.common_packages import read_csv, check_if_file_existed





if __name__ == '__main__':
    device = get_device()
    csv_file = "annotation.csv"
    root_dir =  '/media/igofed/SSD_1T/AI4CI/FULLDATASET/FULLDATASET'
    __dataset = Flood_NonFlood_dataset(csv_file=csv_file, root_dir=root_dir, transform=None)
    
    print('done')