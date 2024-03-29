## Configuration
import torch
import os
from torchvision import transforms
class CFG:
    epochs =300# number epoch to train model
    learning_rate = 1e-3    # learning rate (How NN updates the gradient)
    batch_size = 32 	    # 16 images in one batchs
    img_size = 224  	    # Resize all images to be 256x256
    num_workers = os.cpu_count()
    class_name = ['Class0', 'Class1']
    angle = 5
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

img_extension = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

loss_label_smoothing = 0.1

normalisation = transforms.Normalize(mean=CFG.mean, std= CFG.std)                
denormalisation = transforms.Normalize(mean=[-CFG.mean[0]/CFG.std[0],-CFG.mean[1]/CFG.std[1], -CFG.mean[2]/CFG.std[2]], 
                                       std= [1/CFG.std[0], 1/CFG.std[1], 1/CFG.std[2] ])                
                        

import sys

if sys.platform == "win32":
    delimiter = "\\"
else:
    delimiter = "/"
