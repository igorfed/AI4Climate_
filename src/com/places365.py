import os
import numpy as np
import torch
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import wideresnet
from com.common_packages import check_if_file_existed
import cv2
def load_labels(file_name_category):
    '''
    read txt labels from /classes
    '''
    classes = list()
    check_if_file_existed(file_name_category)
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)
    return classes

def indoor_outdoor(file_name_IO):
    #file_name_IO = 'IO_places365.txt'
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO, labels_F = [],[]
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-2]) -1)
            labels_F.append(int(items[-1]))

    labels_IO = np.array(labels_IO)
    labels_F = np.array(labels_F)
    return labels_IO, labels_F


def scene_attribute(file_name_attribute, file_name_W):
    # scene attribute relevant
    #file_name_attribute = 'labels_sunattribute.txt'
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    W_attribute = np.load(file_name_W)
    return labels_attribute, W_attribute


def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def load_model(model_file):
    # this model has a last conv feature map as 14x14

    #model_file = 'wideresnet18_places365.pth.tar'
    #if not os.access(model_file, os.W_OK):
    #    os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
    #    os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')
    #check_if_file_existed(model_file)
    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model, features_blobs


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf
