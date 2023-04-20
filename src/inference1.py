import torch
import torchvision
from tqdm import tqdm
import argparse
import os
import sys
from torch.autograd import Variable as V
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from com.common_packages import check_if_dir_existed, check_if_file_existed
from com.colors import COLOR
import matplotlib.pyplot as plt
import cv2
import numpy as np
from models import models
import com.config as config
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False
import glob as glob

from dataset import load_dataset
import numpy as np
import com.utils as utils
print(config.CFG.epochs)
import engine
from pathlib import Path
import random
from torchvision import transforms
from PIL import ImageFile, Image
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from com.places365 import load_labels, indoor_outdoor, scene_attribute,returnTF
import wideresnet
import warnings 
warnings.filterwarnings("ignore")
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",   "--path",       type=str, required=False, help="path to the dataset")
    parser.add_argument("-d",   "--dataset",    type=str, required=False, help="type of the dataset")
    parser.add_argument("-md",  "--model",      type=str, required=False, help="type of model")
    parser.add_argument("-pth",  "--pth",       type=str, required=False, help="path to pretrained model")
    parser.add_argument("-plot", "--plot",      type=str, required=False, help="Do you want to plot")

    return vars(parser.parse_args())

def save_images(model, test_path, num_images_to_plot=10):
    test_image_path_list = list(Path(test_path).glob("*/*.png")) # get list all image paths from test data 
    test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                       k=num_images_to_plot) # randomly select 'k' image paths to pred and plot
    i = 0

    image_transform = transforms.Compose([
                        transforms.Resize((config.CFG.img_size, config.CFG.img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                        mean=config.CFG.mean, 
                        std=config.CFG.std)                
                        ]
                    )
    for image_path in test_image_path_sample:
        print(i, image_path)
        # Open image
        img = Image.open(image_path)
        # Turn on model evaluation mode and inference mode
    
        original_img = img.copy()
        model.eval()
        with torch.no_grad():
            # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
            transformed_image = image_transform(img).unsqueeze(dim=0)
            target_image_pred = model(transformed_image.to(device))
            # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
            target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
            # Convert prediction probabilities -> prediction labels
            target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
            # Plot image with predicted label and probability
            #cv2.putText(
            #            img, f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}",
            #        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            #        1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA
            #    )
            #   
            image_path = f"{output_images}/{model_name}_{os.path.basename(image_path)}.png"
            #cv2.imwrite(image_path, img)
            #original_img.save("img1.png")
            plt.figure()
            plt.imshow(original_img)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            textstr = '\n'.join((
                            r'filename    : %s' % (os.path.basename(image_path),),        
                            r'Prediction  : %s' % (config.CFG.class_name[target_image_pred_label]),
                            r'Probability : %s' % (target_image_pred_probs.max()),

                            ))
            plt.text(10, 50, textstr, fontsize=8, verticalalignment='top', bbox=props)
        
            plt.axis(False)
            plt.savefig(image_path)
            i = i +1

def correct_no_correct(images, labels, probs, pred_labels):
    corrects = torch.eq(labels, pred_labels)
    incorrect_examples = []
    correct_examples = []
    for image, label, prob, correct in zip(images, labels, probs, corrects):
        if not correct:
            incorrect_examples.append((image, label, prob))
        if correct:
            correct_examples.append((image, label, prob))
    incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)
    #correct_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)
    return correct_examples, incorrect_examples






def get_predictions(model, iterator, device):
    # Examining the Model
    model.eval()
    images = []
    labels = []
    probs = []
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y_pred = model(x)
            y_prob = F.softmax(y_pred, dim=-1)
            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs

def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return 
def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))


def load_model(model_file):
    # this model has a last conv feature map as 14x14

    #model_file = 'wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    useGPU = 1
    if useGPU == 1:
        model = torch.load(model_file)
    else:
        model = torch.load(model_file, map_location=lambda storage, loc: storage) # allow cpu

    # the following is deprecated, everything is migrated to python36

    ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
    #from functools import partial
    #import pickle
    #pickle.load = partial(pickle.load, encoding="latin1")
    #pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    #model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

    model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model


if __name__ == '__main__':
    '''
    '''
    print('--------------------------------')
    args = arg_parser()
    print(args)
    print('--------------------------------')
    DATA = args['dataset']
    path = args['path']
    model_name = args['model']
    model_path = args['pth']
    enable_plot= bool(args['plot'])
    checkpoint = torch.load(model_path)
    dataset = f'{path}/{DATA}'
    #print(f'dataset: {dataset}')
    test_path = os.path.join(dataset, 'test')
    check_if_dir_existed(test_path)
    device = 'cpu'
    Color = (COLOR.Blue if torch.cuda.is_available() else COLOR.Red)
    print(f'device{device}')
    test_set, test_loader   = load_dataset(test_path, mode='test', balance_dataset=True)
    #utils.show_batch_grid(test_loader, True, 'test', enable_plot=False, figure_save=True)
    #     STEP 1. SORT by indoor, outdoor

    # load the labels
    classes = load_labels(file_name_category='categories_places365.txt')
    labels_IO = indoor_outdoor(file_name_IO = 'IO_places365.txt')
    labels_attribute, W_attribute = scene_attribute(file_name_attribute = 'labels_sunattribute.txt')
    #print(classes)
    #print(labels_IO)
    #print(len(classes), len(labels_IO))
    #print(labels_attribute)
    #print(W_attribute)
    

    # LOAD PLACES CNN  
    features_blobs = []
    model_places = load_model(model_file = 'whole_wideresnet18_places365_python36.pth.tar')
    tf = returnTF()
    # get the softmax weight
    params = list(model_places.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax<0] = 0


    #img_url = 'http://places.csail.mit.edu/demo/6.jpg'
    #os.system('wget %s -q -O test.jpg' % img_url)
    #img = Image.open('test.jpg')
    #input_img = V(tf(img).unsqueeze(0))
    #print(type(input_img), input_img.shape)
    
    i = 0
    for (x, y) in test_loader:
        print(type(x),x[0].shape)#, x[0][:][:][:], y )
        if i >0:
            break
        i =i +1
        plt.imshow(x[i].permute(1, 2, 0))
        plt.title(f"Training example #{i}")
        plt.axis('off')
    plt.show()
    
   #print(model_places)
    # my pretrained model
    #model, total_params = models.build_model(model_name, device)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #train_loss = checkpoint['train_loss']
    #train_acc = checkpoint['train_acc']
    #valid_loss = checkpoint['valid_loss']
    #valid_acc = checkpoint['valid_acc']
    #model.eval()
    

    #plt.show()
    print('done')
    
    

