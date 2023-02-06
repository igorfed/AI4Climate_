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

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",   "--path",       type=str, required=False, help="path to the dataset")
    parser.add_argument("-d",   "--dataset",    type=str, required=False, help="type of the dataset")
    parser.add_argument("-md",  "--model",      type=str, required=False, help="type of model")
    parser.add_argument("-pth",  "--pth",       type=str, required=False, help="path to pretrained model")
    parser.add_argument("-plot", "--plot",      type=str, required=False, help="Do you want to plot")

    return vars(parser.parse_args())

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
    test_path = os.path.join(dataset, 'test')
    check_if_dir_existed(test_path)
    device = 'cpu'
    checkpoint = torch.load(model_path)    
    model, total_params = models.build_model(model_name, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    ###################
    output_paths = os.path.join(os.getcwd(), 'outputs')
    output_images = os.path.join(output_paths, 'images')
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
    for i in range(len(dataset)):
        with torch.no_grad():
            model.eval()
            x, image_name, original_image = dataset[i]
            x = x.unsqueeze(dim=0)
            x = x.to(device)
            y_pred = model(x)
            y_prob = F.softmax(y_pred, dim=-1)
            images = x.cpu()
            probs = y_prob.cpu()

        pred_labels = torch.argmax(probs, 1)
        probs = probs.tolist()
        pred_labels = pred_labels.tolist()
        pred_labels = pred_labels[0]
        probability = trunc(probs[0][pred_labels],2)
        image_path = f"{output_images}/{image_name}_{pred_labels}_{probability}.png"
    

        print(i, image_path)
        cv2.imwrite(image_path, cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                           

    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #textstr = '\n'.join((
    #            f'Image name : {image_name}',
    #            f'Probability : {probability}',
    #            f'Class Name : {pred_labels}',
    #            ))
    
    #img = config.denormalisation(images[0]).permute(1,2,0)
    cv2.imwrite(image_path, cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    #plt.imshow(original_image) 
    #plt.text(0.05, 0.95, textstr, fontsize=10, verticalalignment='top', bbox=props)
    #plt.grid(False)
    #plt.show()
    print('done')
    
    

