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
from com.places365 import load_labels, indoor_outdoor, scene_attribute

def load_efficientnet_model(pretrained=True, fine_tune=True, num_classes=2):
    """
    Load the pre-trained EfficientNetB0 model.
    pretrained : It will be a boolean value indicating whether we want to load the ImageNet weights or not.
    fine_tune: It is also a boolean value. When it is True, all the intermediate layers will also be trained.
    num_classes: Number of classes in the dataset.
    """
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = torchvision.models.efficientnet_b0(pretrained=pretrained)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
        # Change the final classification head.
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
    return model


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



if __name__ == '__main__':
    print('INFERENCE--------------------------------')
    args = arg_parser()
    print('--------------------------------')
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
    utils.show_batch_grid(test_loader, True, 'test', enable_plot=False, figure_save=True)
    #     STEP 1. SORT by indoor, outdoor

    # load the labels
    #classes, labels_IO, labels_attribute, W_attribute = load_labels()


    # pretrained
    #model, total_params = models.build_model(model_name, device)
    model = load_efficientnet_model()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    train_loss = checkpoint['train_loss']
    train_acc = checkpoint['train_acc']
    valid_loss = checkpoint['valid_loss']
    valid_acc = checkpoint['valid_acc']
    model.eval()
    utils.save_plots(train_acc, valid_acc, train_loss, valid_loss, enable_plot=True, figure_name=model_path)
    output_paths = os.path.join(os.getcwd(), 'outputs')
    output_images = os.path.join(output_paths, 'images_pretrained')
    print(40*"-")
    check_if_dir_existed(output_images, True)
    ##################save_images(model, test_path, num_images_to_plot=16)
    images, labels, probs = get_predictions(model, test_loader, device)
    
    # for each prediction we get the predicted class.

    pred_labels = torch.argmax(probs, 1)

    print('labels', labels)
    print('pred_labels', pred_labels, len(pred_labels), len(probs), len(images))
    
    # plot the confusion matrix

    utils.plot_confusion_matrix(labels, pred_labels, config.CFG.class_name, model_name, enable_plot=False, figure_save=True)

    tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0,1]).ravel()
    from sklearn.metrics import classification_report
    print(classification_report(labels, pred_labels))

    print('True Positive', tp)
    print('True Negative', tn)
    print('False Positive', fp)
    print('False Negative', fn)
    acc = (tp+tn)/(tp+tn+fp+fn)
    print(f'Overal accuracy {acc:.3f}')
    print(f'Miscalculation {1 - acc:.3f}')

    # Correct vs non-correct
    
    correct_examples, incorrect_examples = correct_no_correct(images, labels, probs, pred_labels)

    
    utils.plot_most_incorrect('incorrect_examples', incorrect_examples, config.CFG.class_name, 16, model_name, True, figure_save=True)
    utils.plot_most_incorrect('correct_examples', correct_examples,config.CFG.class_name, 36, model_name, True, figure_save=True)
    print("ENABLE_plot", enable_plot, type(enable_plot))
    if enable_plot==True:
        plt.show()
    


    print('done')
    
    

