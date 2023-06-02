import torch
import argparse
import os
import sys
from typing import Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from com.common_packages import check_if_dir_existed, trunc, check_if_file_existed
from com.colors import COLOR
import matplotlib.pyplot as plt
import cv2
import numpy as np
from models import models
import com.config as config
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False
import glob as glob
import numpy as np
from torchvision import transforms
from PIL import ImageFile, Image
import torch.nn.functional as F
import com.places365 as places 

#import load_labels, indoor_outdoor, scene_attribute
import warnings 
warnings.filterwarnings("ignore")
from torch.autograd import Variable as V
from torchvision import transforms as trn
import time
import csv 

timestr = time.strftime("%Y%m%d-%H%M%S")
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",   "--path",       type=str, required=False, help="path to the dataset")
    parser.add_argument("-d",   "--dataset",    type=str, required=False, help="type of the dataset")
    parser.add_argument("-md",  "--model",      type=str, required=False, help="type of model")
    parser.add_argument("-pth",  "--pth",       type=str, required=False, help="path to pretrained model")
    parser.add_argument("-plot", "--plot",      type=str, required=False, help="Do you want to plot")

    return vars(parser.parse_args())
def returnTF():
    # load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf

class CustomTestLoader(torch.utils.data.Dataset):

    """Define test dataset loading methods.
    Args:
        image_dir (str) : test dataset address.
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
                #rotate      : int,  # rotation int
                mean        : list, # mean value List[int, int, int]
                std         : list, # std value List[int, int, int]
                ) -> None:
        super(CustomTestLoader, self).__init__()

        self.paths = list(glob.glob(f"{image_dir}/*"))
        self.image_size = image_size
        self.delimiter = config.delimiter
        self.mean = mean
        self.std = std
        self.img_extension = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
        self.transform = transforms.Compose([
                        transforms.Resize((config.CFG.img_size, config.CFG.img_size)),
                        transforms.ToTensor(), 
                        transforms.Normalize(
                            mean=config.CFG.mean, 
                            std=config.CFG.std)                
                        ])
        
        ########### Classes not Defined #####################
        #self.classes, self.class_to_idx = self.find_classes(image_dir)
        #print(COLOR.Green + 'Classes', self.class_to_idx ,  COLOR.END)

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
   
   
    def load_image(self, index: int):
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        image = cv2.imread(image_path)[...,::-1]
        return Image.fromarray(image), image
 


    def __getitem__(self, index: int)-> Tuple[torch.Tensor, str, Image.Image]:#Image.Image
        
        if torch.is_tensor(index):
            index = index.tolist()

        self.image_dir, self.image_name = self.paths[index].split(self.delimiter)[-2:]
        image, original_image = self.load_image(index)
        img = self.transform(image)
        return img, os.path.splitext(self.image_name)[0], original_image
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
    return model

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    print(nc, h, w)
    print(feature_conv.reshape((nc, h*w)).shape)
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

if __name__ == '__main__':
    print('--------------------------------')
    args = arg_parser()
    print('--------------------------------')
    DATA = args['dataset']
    path = args['path']
    check_if_dir_existed(path)
    model_name = args['model']
    model_path = args['pth']
    enable_plot= bool(args['plot'])
    dataset = f'{path}/{DATA}'
    #test_path = os.path.join(dataset, 'test__')
    test_path = os.path.join(dataset, 'test_flood_noflood/class1')
    model_path = f'./outputs/{model_path}'
    check_if_file_existed(model_path)
    device = 'cpu'
    print(model_path)
    check_if_file_existed(model_path)
    checkpoint = torch.load(model_path)    
    model, total_params = models.build_model(model_name, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    ###################
    output_paths = os.path.join(os.getcwd(), 'outputs')
    output_images = os.path.join(output_paths, 'images_test_')

    check_if_dir_existed(output_images, True)
    output_csv = output_images
    output_images = os.path.join(output_images, timestr)
    check_if_dir_existed(output_images, True)

    dataset = CustomTestLoader(
                                image_dir=test_path,
                                image_size = config.CFG.img_size,
                                mean = config.CFG.mean,
                                std = config.CFG.std)
   

    # PLACES 
    file_name_category='categories_places365.txt'
    file_name_category = f'./src/classes/{file_name_category}'
    classes = places.load_labels(file_name_category)

    file_name_IO='IO_places365_F.txt'
    file_name_IO = f'./src/classes/{file_name_IO}'

    labels_IO, labels_F = places.indoor_outdoor(file_name_IO)
    file_name_attribute = 'labels_sunattribute.txt'
    file_name_attribute = f'./src/classes/{file_name_attribute}'
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    file_name_W = f'./src/classes/{file_name_W}'
    labels_attribute, W_attribute = places.scene_attribute(file_name_attribute, file_name_W)
    # load the model

    model_file='wideresnet18_places365.pth.tar'
    model_file = f'./src/models/{model_file}'
    check_if_file_existed(model_file)
    
    features_blobs = []
    places_model = load_model(model_file)

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax<0] = 0

    
    classes = np.array(classes)
    with open(f'{output_csv}/output_{timestr}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["i", 
                         "fname", 
                         "classes", 
                         "indoor_prob", 
                         "indoor", 
                         "relevance_prob",
                         "relevance",
                         "water_prob",
                         "water",
                         "flooded",
                         "flooded_prob",
                         "attributes(option)" # not shure how to use
                         ])
        
        for i in range(len(dataset)):
        #for i in range(10):
            with torch.no_grad():
                x, image_name, original_image = dataset[i]
                X = x.view(1,3,224,224)
                # places365
                places_model.eval()
                # forward pass
                logit = places_model.forward(X)
                h_x = F.softmax(logit, 1).data.squeeze()
                probs, idx = h_x.sort(0, True)
                probs = probs.numpy()
                idx = idx.numpy()
                io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
                f_image = np.mean(labels_F[idx[:10]]) # vote for can be  flooded or not 
                
                # output the prediction of scene category
                print(f'{i}:{image_name}:{original_image.shape}')
                if io_image < 0.5:
                    indoor = 0
                    print(COLOR.Red,'\t--TYPE OF ENVIRONMENT: indoor', COLOR.END)
                else:
                    indoor = 1
                    print(COLOR.Green,'\t--TYPE OF ENVIRONMENT: outdoor', COLOR.END)
                
                probs_indor_outdoor = trunc(np.mean(probs[:10]),3)
                
                if f_image < 0.5:
                    relevant = 0
                    print(COLOR.Red, '\t--Not RELEVANT', COLOR.END)
                else:
                    relevant = 1
                    print(COLOR.Green, '\t--RELEVANT', COLOR.END)

                for i in range(0, 5):
                    probs[i] = round(probs[i],2)
                    print('\t{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
                
                

                # output the scene attributes
                responses_attribute = W_attribute.dot(features_blobs[1])
                idx_a = np.argsort(responses_attribute)
                idx_a_reverse = np.array(idx_a[::-1])
                attributes = []
                relevances = []
                for i in range(10):
                    print(f'\t{i} {idx_a_reverse[i]} -> {labels_attribute[idx_a_reverse[i]]}')
                    attributes.append(labels_attribute[idx_a_reverse[i]])
                    relevances.append(labels_F[idx[i]])
                model.eval()
                x = x.unsqueeze(dim=0)
                x = x.to(device)
                y_pred = model(x)
                y_prob = F.softmax(y_pred, dim=-1)
                images = x.to(device)
                myprobs = y_prob.to(device)

                pred_labels = torch.argmax(myprobs, 1)
                myprobs = myprobs.tolist()
                pred_labels = pred_labels.tolist()
                pred_labels = pred_labels[0]
                probability = trunc(myprobs[0][pred_labels],3)
                print(f"water {pred_labels}, {probability}")
                
                if pred_labels==0:
                    probability_of_water = trunc(1-probability,3)
                else:
                    probability_of_water = probability
                
                if indoor==1:
                    fooded_prob = trunc(probability_of_water, 3)
                else:
                    fooded_prob = 0
                    
                if fooded_prob < 0.5:
                    flooded = 0
                    print('\t--No Flooded')
                else:
                    flooded = 1
                    print(COLOR.Green, '\t--Flooded', COLOR.END)


                writer.writerow([i, # num
                                 image_name,  #f_name
                                 classes[(idx[:10])],
                                 probs_indor_outdoor,
                                 indoor, 
                                 f_image, 
                                 relevances, #relevant,           
                                 probability_of_water,
                                 pred_labels,
                                 flooded,
                                 fooded_prob,
                                 attributes])
                
                image_path = f"{output_images}/{flooded}_{image_name}_{fooded_prob}.png"           
                
            
                print(f"image_path: {image_path}")
                cv2.imwrite(image_path, cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

        #print(i, image_path)

        #if i >k:
        #    k = k
            
            
        #else:
        #    k = k+1            
            
        
        #plt.show()
        
    
        #    pass       #cv2.imshow(window_name, cv2.cvtColor(cv2.resize(original_image, None , interpolation=cv2.INTER_CUBIC,  fx=4, fy=4,) , cv2.COLOR_BGR2RGB))
        #cv2.imwrite(image_path, cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                           
    #cv2.waitKey(0)
  
    # closing all open windows
    #cv2.destroyAllWindows()

    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #textstr = '\n'.join((
    #            f'Image name : {image_name}',
    #            f'Probability : {probability}',
    #            f'Class Name : {pred_labels}',
    #            ))
    
    #img = config.denormalisation(images[0]).permute(1,2,0)
    
    #plt.imshow(original_image) 
    #plt.text(0.05, 0.95, textstr, fontsize=10, verticalalignment='top', bbox=props)
    #plt.grid(False)
    #plt.show()
    
    file.close()
    print('done')
    
    

