import torch
import argparse
import os
import sys
from typing import Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from com.common_packages import check_if_dir_existed, trunc
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
 


if __name__ == '__main__':
    print('--------------------------------')
    args = arg_parser()
    print('--------------------------------')
    DATA = args['dataset']
    path = args['path']
    model_name = args['model']
    model_path = args['pth']
    enable_plot= bool(args['plot'])
    dataset = f'{path}/{DATA}'
    test_path = os.path.join(dataset, 'test_')
    check_if_dir_existed(test_path)
    device = 'cpu'
    checkpoint = torch.load(model_path)    
    model, total_params = models.build_model(model_name, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    ###################
    output_paths = os.path.join(os.getcwd(), 'outputs')
    output_images = os.path.join(output_paths, 'images_test_')
    check_if_dir_existed(output_images, True)

    dataset = CustomTestLoader(
                                image_dir=test_path,
                                image_size = config.CFG.img_size,
                                mean = config.CFG.mean,
                                std = config.CFG.std)
   

    #data_loader = torch.utils.data.DataLoader(
    #                            dataset=dataset,   
    #                            batch_size=config.CFG.batch_size,
    #                            num_workers=config.CFG.num_workers,
    #                            shuffle=True)

    #mean, std = torch.FloatTensor(config.CFG.mean), torch.FloatTensor(config.CFG.mean)
    
    #images = []
    #probs = []

    # show images
    # PLACES 
    classes = places.load_labels(file_name_category='categories_places365.txt')
    labels_IO = places.indoor_outdoor(file_name_IO = 'IO_places365.txt')
    labels_attribute, W_attribute = places.scene_attribute(file_name_attribute = 'labels_sunattribute.txt', file_name_W = 'W_sceneattribute_wideresnet18.npy')
    # load the model
    
    places_model = places.load_model(model_file = 'wideresnet18_places365.pth.tar')

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax<0] = 0
    #img_url = 'http://places.csail.mit.edu/demo/6.jpg'
    #os.system('wget %s -q -O test.jpg' % img_url)
    img = Image.open('12.jpg')
    tf = returnTF() # image transformer
    input_img = V(tf(img).unsqueeze(0))
    window_name = 'Test opencv '
    figure_name =  f'Type of environment /Indor-Outdoor/ Images in [Class0, Class1]'
    fig = plt.figure(figure_name, figsize=(30, 20), dpi=80)
    M, N = 4, 4
    k = 1
    for i in range(len(dataset)):
        with torch.no_grad():
            x, image_name, original_image = dataset[i]
            print('Image name', image_name)
            X = x.view(1,3,224,224)
            print(type(x), x.shape)
            print(type(X), X.shape)
            print(type(input_img), input_img.shape)
            # places365
            places_model.eval()
            # forward pass
            
            logit = places_model.forward(X)
            h_x = F.softmax(logit, 1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            probs = probs.numpy()
            idx = idx.numpy()
            # output the IO prediction      
            print(labels_IO[:10])
            io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
            
            if io_image < 0.6:
                indoor = True
                print('--TYPE OF ENVIRONMENT: indoor')
            else:
                indoor = False
                print('--TYPE OF ENVIRONMENT: outdoor')

            # output the prediction of scene category
            print('--SCENE CATEGORIES:')
            #for i in range(0, 5):
            #    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
            # output the prediction of scene category
            #print('--SCENE CATEGORIES:')
            for i in range(0, 5):
                probs[i] = round(probs[i],2)
                print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

            model.eval()
            #x, image_name, original_image = dataset[i]
            x = x.unsqueeze(dim=0)
            x = x.to(device)
            y_pred = model(x)
            y_prob = F.softmax(y_pred, dim=-1)
            images = x.cpu()
            myprobs = y_prob.cpu()
        # generate class activation mapping
        #CAMs = places.returnCAM(places.features_blobs[0], weight_softmax, [idx[0]])

        pred_labels = torch.argmax(myprobs, 1)
        myprobs = myprobs.tolist()
        pred_labels = pred_labels.tolist()
        pred_labels = pred_labels[0]
        probability = trunc(myprobs[0][pred_labels],3)
        #image_path = f"{output_images}/{image_name}_{pred_labels}_{probability}.png"
        
        
        #if k < M*N: 
           # ax = fig.add_subplot(M, N, k)
           # ax.imshow(original_image)
           # print(i, k, M*N, len(dataset))
           # ax.set_aspect('equal')
          #  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
         #   textstr = '\n'.join((
        #            r'Batch size     : %s' % (config.CFG.batch_size,),        
       #             r'Image in batch : %s' % (k-1),
      #              r'Indoor         : %s' % (indoor),
     #               f'Class          : {classes[idx[0]]}, {classes[idx[1]]}, {classes[idx[2]]}',
    #                f'Water flooding : {pred_labels}, {probability}',
   #                 #r'Class         : %s' % (config.CFG.class_name[labels[k-1].numpy()]),
  #                  ))
 #           ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
#            ax.grid(False)
        #batch_size = f'BS : {config.CFG.batch_size}'
        #img_in_batch = f'Image in B : {k-1}'
        #imgclass = f'C : {pred_labels}:{probability}'
        #original_image= cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        #cv2.putText(original_image, imgclass, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 
        #           1, (255, 0, 0), 1, cv2.LINE_AA)
        #cv2.putText(original_image, img_in_batch, (10,45), cv2.FONT_HERSHEY_SIMPLEX, 
        #           1, (255, 0, 0), 1, cv2.LINE_AA)
        if not indoor:
            image_path = f"{output_images}/{pred_labels}_outdoor_{image_name}_{probability}.png"           
        else:
            image_path = f"{output_images}/{0}_indoor_{image_name}_{io_image}.png"           
        #image_path = f"{output_images}/{image_name}_indoor.png"
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
    print('done')
    
    

