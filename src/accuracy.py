import torch
import torchvision
from tqdm import tqdm
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from com.common_packages import check_if_dir_existed
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

__models = ['densenet121', 'densenet169', 'densenet201', 'densenet161', 'efficientnet_b0', 'efficientnet_b1']

model_paths = {
    'densenet121': '/home/igofed/LiU/AI4Climate_version_updated/AI4Climate_/outputs/densenet121_weather_2_last_checkpoint.bin',
    'densenet161': '/home/igofed/LiU/AI4Climate_version_updated/AI4Climate_/outputs/densenet161_weather_2_last_checkpoint.bin',
    'densenet169': '/home/igofed/LiU/AI4Climate_version_updated/AI4Climate_/outputs/densenet169_weather_2_last_checkpoint.bin',
    'densenet201': '/home/igofed/LiU/AI4Climate_version_updated/AI4Climate_/outputs/densenet201_weather_2_last_checkpoint.bin',
    'efficientnet_b0': '/home/igofed/LiU/AI4Climate_version_updated/AI4Climate_/outputs/efficientnet_b0_weather_2_last_checkpoint.bin',
    'efficientnet_b1': '/home/igofed/LiU/AI4Climate_version_updated/AI4Climate_/outputs/efficientnet_b1_weather_2_last_checkpoint.bin',
}

dataset='weather_2'
path='/media/igofed/SSD_2T/DATASETS/'


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



def prnt(tp, tn, fp, fn):
    print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn} ' )
    acc = (tp+tn)/(tp+tn+fp+fn)
    print(f'Overal accuracy {acc:.3f}')
    print(f'Miscalculation {1 - acc:.3f}')


if __name__ == '__main__':
    '''
    '''
    print('--------------------------------')
    device = 'cpu'
    Color = (COLOR.Blue if torch.cuda.is_available() else COLOR.Red)
    print(f'device{device}')

    # pretrained
    checkpoint0 = torch.load(model_paths[__models[0]])
    checkpoint1 = torch.load(model_paths[__models[1]])
    checkpoint2 = torch.load(model_paths[__models[2]])
    checkpoint3 = torch.load(model_paths[__models[3]])
    checkpoint4 = torch.load(model_paths[__models[4]])
    checkpoint5 = torch.load(model_paths[__models[5]])


    model0, total_params0 = models.build_model(__models[0], device)
    model0.load_state_dict(checkpoint0['model_state_dict'])
    

    model1, total_params1 = models.build_model(__models[1], device)
    model1.load_state_dict(checkpoint1['model_state_dict'])
    
    model2, total_params2 = models.build_model(__models[2], device)
    model2.load_state_dict(checkpoint2['model_state_dict'])
    
    model3, total_params3 = models.build_model(__models[3], device)
    model3.load_state_dict(checkpoint3['model_state_dict'])

    model4, total_params4 = models.build_model(__models[4], device)
    model4.load_state_dict(checkpoint4['model_state_dict'])

    model5, total_params5 = models.build_model(__models[5], device)
    model5.load_state_dict(checkpoint5['model_state_dict'])


    #train_loss = checkpoint['train_loss']
    #train_acc = checkpoint['train_acc']
    #valid_loss = checkpoint['valid_loss']
    #valid_acc = checkpoint['valid_acc']
    

    fig = plt.figure('losses', figsize=(30, 20), dpi=80)
    ax0 = fig.add_subplot(2, 4, 1)
    ax0.plot(checkpoint0['train_loss'], label='train loss')
    ax0.plot(checkpoint0['valid_loss'], label='valid loss')
    ax0.plot(checkpoint0['train_acc'], label='train acc')
    ax0.plot(checkpoint0['valid_acc'], label='valid acc')
    ax0.set_title(__models[0], fontsize=10)

    ax1 = fig.add_subplot(2, 4, 2)
    ax1.plot(checkpoint1['train_loss'], label='train loss')
    ax1.plot(checkpoint1['valid_loss'], label='valid loss')
    ax1.plot(checkpoint1['train_acc'], label='train acc')
    ax1.plot(checkpoint1['valid_acc'], label='valid acc')
    ax1.set_title(__models[1], fontsize=10)
    
    ax2 = fig.add_subplot(2, 4, 3)
    ax2.plot(checkpoint2['train_loss'], label='train loss')
    ax2.plot(checkpoint2['valid_loss'], label='valid loss')
    ax2.plot(checkpoint2['train_acc'], label='train acc')
    ax2.plot(checkpoint2['valid_acc'], label='valid acc')
    ax2.set_title(__models[2], fontsize=10)
    
    ax3 = fig.add_subplot(2, 4, 4)
    ax3.plot(checkpoint3['train_loss'], label='train loss')
    ax3.plot(checkpoint3['valid_loss'], label='valid loss')
    ax3.plot(checkpoint3['train_acc'], label='train acc')
    ax3.plot(checkpoint3['valid_acc'], label='valid acc')
    ax3.set_title(__models[3], fontsize=10)

    ax4 = fig.add_subplot(2, 4, 5)
    ax4.plot(checkpoint4['train_loss'], label='train loss')
    ax4.plot(checkpoint4['valid_loss'], label='valid loss')
    ax4.plot(checkpoint4['train_acc'], label='train acc')
    ax4.plot(checkpoint4['valid_acc'], label='valid acc')
    ax4.set_title(__models[4], fontsize=10)

    ax5 = fig.add_subplot(2, 4, 6)
    ax5.plot(checkpoint5['train_loss'], label='train loss')
    ax5.plot(checkpoint5['valid_loss'], label='valid loss')
    ax5.plot(checkpoint5['train_acc'], label='train acc')
    ax5.plot(checkpoint5['valid_acc'], label='valid acc')
    ax5.set_title(__models[5], fontsize=10)


    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    
    #####################################################################
    dataset = f'{path}/{dataset}'
    test_path = os.path.join(dataset, 'test')
    check_if_dir_existed(test_path)
    test_set, test_loader   = load_dataset(test_path, mode='test', balance_dataset=True)
    ##################save_images(model, test_path, num_images_to_plot=16)

    images0, labels0, probs0 = get_predictions(model0, test_loader, device)
    images1, labels1, probs1 = get_predictions(model1, test_loader, device)
    images2, labels2, probs2 = get_predictions(model2, test_loader, device)
    images3, labels3, probs3 = get_predictions(model3, test_loader, device)
    images4, labels4, probs4 = get_predictions(model4, test_loader, device)
    images5, labels5, probs5 = get_predictions(model5, test_loader, device)
    ########################################################################
    acc = []
    TN = []
    FP = []
    FN = []
    TP = []
    cm0 = confusion_matrix(labels0, torch.argmax(probs0, 1))
    cm0 = ConfusionMatrixDisplay(cm0, display_labels=config.CFG.class_name)
    tn, fp, fn, tp = confusion_matrix(labels0, torch.argmax(probs0, 1), labels=[0,1]).ravel()
    prnt(tp, tn, fp, fn)
    acc.append((tp+tn)/(tp+tn+fp+fn))
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    TP.append(tp)
    
    cm1 = confusion_matrix(labels1, torch.argmax(probs1, 1))
    cm1 = ConfusionMatrixDisplay(cm1, display_labels=config.CFG.class_name)
    tn, fp, fn, tp = confusion_matrix(labels1, torch.argmax(probs1, 1), labels=[0,1]).ravel()
    acc.append((tp+tn)/(tp+tn+fp+fn))
    prnt(tp, tn, fp, fn)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    TP.append(tp)

    cm2 = confusion_matrix(labels2, torch.argmax(probs2, 1))
    cm2 = ConfusionMatrixDisplay(confusion_matrix(labels2, torch.argmax(probs2, 1)), display_labels=config.CFG.class_name)
    tn, fp, fn, tp = confusion_matrix(labels2, torch.argmax(probs2, 1), labels=[0,1]).ravel()
    acc.append((tp+tn)/(tp+tn+fp+fn))
    prnt(tp, tn, fp, fn)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    TP.append(tp)

    cm3 = confusion_matrix(labels3, torch.argmax(probs3, 1))
    cm3 = ConfusionMatrixDisplay(confusion_matrix(labels3, torch.argmax(probs3, 1)), display_labels=config.CFG.class_name)
    tn, fp, fn, tp = confusion_matrix(labels3, torch.argmax(probs3, 1), labels=[0,1]).ravel()
    acc.append((tp+tn)/(tp+tn+fp+fn))
    prnt(tp, tn, fp, fn)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    TP.append(tp)

    cm4 = confusion_matrix(labels4, torch.argmax(probs4, 1))
    cm4 = ConfusionMatrixDisplay(confusion_matrix(labels4, torch.argmax(probs4, 1)), display_labels=config.CFG.class_name)
    tn, fp, fn, tp = confusion_matrix(labels4, torch.argmax(probs4, 1), labels=[0,1]).ravel()
    acc.append((tp+tn)/(tp+tn+fp+fn))
    prnt(tp, tn, fp, fn)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    TP.append(tp)

    cm5 = confusion_matrix(labels5, torch.argmax(probs5, 1))
    cm5 = ConfusionMatrixDisplay(confusion_matrix(labels5, torch.argmax(probs5, 1)), display_labels=config.CFG.class_name)
    tn, fp, fn, tp = confusion_matrix(labels5, torch.argmax(probs5, 1), labels=[0,1]).ravel()
    acc.append((tp+tn)/(tp+tn+fp+fn))
    prnt(tp, tn, fp, fn)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    TP.append(tp)



    fig1 = plt.figure('confusion', figsize=(30, 20), dpi=80)
    ax10 = fig1.add_subplot(2, 4, 1)
    ax20 = fig1.add_subplot(2, 4, 2)
    ax30 = fig1.add_subplot(2, 4, 3)
    ax40 = fig1.add_subplot(2, 4, 4)
    ax50 = fig1.add_subplot(2, 4, 5)
    ax60 = fig1.add_subplot(2, 4, 6)
    ax70 = fig1.add_subplot(2, 4, 7)
    
    cm0.plot(values_format='d', cmap='Blues', ax=ax10)
    cm1.plot(values_format='d', cmap='Blues', ax=ax20)
    cm2.plot(values_format='d', cmap='Blues', ax=ax30)
    cm3.plot(values_format='d', cmap='Blues', ax=ax40)
    cm4.plot(values_format='d', cmap='Blues', ax=ax50)
    cm5.plot(values_format='d', cmap='Blues', ax=ax60)
    ax70.plot(acc, label='Accuracy')
    ax70.legend(loc='lower left')

    #ax2.set_ylim([-0.1, 1.1])
    

    plt.xticks(rotation=20)     




    plt.show()
    print('done')
    
    

