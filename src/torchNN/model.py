import torchvision.models as models
import torch.nn as nn
import timm
import torch

'''
EfficientNet is a CNN architecture and scaling method that uniformly scales all 
dimensions of depth/width/resolution using a compound coefficient. 
Unlike conventional practice that arbitrary scales these factors, 
the EfficientNet scaling method uniformly scales network width, depth, 
and resolution with a set of fixed scaling coefficients
'''
def build_efficient_net(pretrained=True, fine_tune=True):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')

    model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)
    
    #if fine_tune:
    #    print('[INFO]: Fine-tuning all layers...')
    #    for params in model.parameters():
    #        params.requires_grad = True
    #elif not fine_tune:
    #    print('[INFO]: Freezing hidden layers...')
    #    for params in model.parameters():
    #        params.requires_grad = False

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=1792, out_features=625), #1792 is the orginal in_features
        torch.nn.ReLU(), #ReLu to be the activation function
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(in_features=625, out_features=256),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=256, out_features=2), 
    )


    

    return model

def build_dense_net(pretrained=True):
    model = timm.create_model('densenet121', pretrained=True)
    return model

