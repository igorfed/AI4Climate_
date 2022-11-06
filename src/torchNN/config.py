from curses import noraw
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

class CFG:
    
    epochs =20 # nomber epoch to train model
    lr = 0.001 # learning rate (How NN updates the gradient)
    batch_size = 16 # 16 images in one batchs
    model_name = 'tf_efficientnet_b4_ns' #(The model imported from timm)
    img_crop = 224 # Crop all images to be 224x224 after resizing
    img_size = 256 # Resize all images to be 256x256
    num_workers = 4
    class_name = ['No-Flooded Image', 'Flooded Image']
    
    

def show_image(image, label, N,  get_denormalize = True):
    '''
    Helper Gunction for showing the image
    '''
    image = image.permute(1,2,0)
    mean, std = torch.FloatTensor([0.485, 0.456, 0.406]), torch.FloatTensor([0.229, 0.224, 0.225])
    
    if get_denormalize:
        image = image*std + mean
        image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.title(str(N) + "____" + label)


def listdirs(rootdir):
    for it in os.scandir(rootdir):
        if it.is_dir():
            print(it.path)

import argparse
def arg_parser_dataset():
    parser = argparse.ArgumentParser(description = 'copy all dataset into one. Merge CSV')
    parser.add_argument("-d", "--dataset", type=str, required=False, help="path to the dataset")
    #parser.add_argument("-m", "--model", type=str, required=False, help="path to output trained model")
    #parser.add_argument("-p", "--plot", type=str, required=False, help="path to output loss/accuracy plot")
    return parser.parse_args()

def show_image(image, label, get_denormalize = True):
    '''
    Helper Gunction for showing the image
    '''
    image = image.permute(1,2,0)
    mean, std = torch.FloatTensor([0.485, 0.456, 0.406]), torch.FloatTensor([0.229, 0.224, 0.225])
    
    if get_denormalize:
        image = image*std + mean
        image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.title(label)


def show_grid(image, title = None):
    '''
    Helper Function for showing the image in the grid
    '''
    image = image.permute(1,2,0)
    mean, std = torch.FloatTensor([0.485, 0.456, 0.406]), torch.FloatTensor([0.229, 0.224, 0.225])
    image = image*std + mean
    image = np.clip(image, 0, 1)
    plt.figure(figsize = [15,15])
    plt.imshow(image)
    if title != None:
        plt.title(title)
        
import torchvision.transforms.functional as F
import torchvision

def show_image_grid(images, labels):
    #images = images.permute(1,2,0)
        
    #if not isinstance(images, list):
    #    images = [images]
    #images = np.clip(images, 0, 1)
    mean, std = torch.FloatTensor([0.485, 0.456, 0.406]), torch.FloatTensor([0.229, 0.224, 0.225])
    fig, axs = plt.subplots(ncols=len(images), squeeze=False)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    images = images.permute(1,2,0)
    mean, std = torch.FloatTensor([0.485, 0.456, 0.406]), torch.FloatTensor([0.229, 0.224, 0.225])
    images = images*std + mean
    images = np.clip(images, 0, 1)
    plt.figure(figsize = [15,15])
    plt.imshow(images)


def slice_plot(data_loader, nrow, ncol, fig_title):
        mean, std = torch.FloatTensor([0.485, 0.456, 0.406]), torch.FloatTensor([0.229, 0.224, 0.225])
        fig = plt.figure(fig_title, figsize=(30, 20), dpi=80)
        data_iter = iter(data_loader)
        print(len (data_iter.next()))
        print('Data length', len(data_iter))
        images, labels = data_iter.next()
        k=0
        s = 'Data length'
        for i in range(nrow):
            #print(i, labels[i])
            for j in range(ncol):
                k = k+1
                print(i+1, j+1, k)
                ax = fig.add_subplot(nrow, ncol, k)
                img = images[i+j].permute(1,2,0)*std + mean
                
                ax.imshow(np.asarray(img))
                ax.set_aspect('equal')
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                textstr = '\n'.join((
                    r'Batch size    : %s' % (CFG.batch_size,),        
                    r'Image in batch: %s' % (k-1),
                    r'Label         : %s' % (labels[i+j].numpy()),
                    ))
                ax.set_title(CFG.class_name[labels[i+j].numpy()], fontsize=10, color='red' if labels[i+j].numpy() == 1 else 'green')
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
