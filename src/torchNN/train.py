import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from com.common_packages import check_if_dir_existed
from com.colors import COLOR
from config import CFG
from dataset import get_datasets
from dataset import get_data_loaders
from config import slice_plot
from utils import save_plots
from utils import save_model
from model import build_efficient_net
from model import build_dense_net
from torchsummary import  summary
import time 
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=False, help="path to the dataset")
    parser.add_argument('-e', '--epochs', type=int, default=20,help='Number of epochs to train our network for')
    parser.add_argument('-pt', '--pretrained', action='store_true',help='Whether to use pretrained weights or not')
    parser.add_argument('-lr', '--learning-rate', type=float,dest='learning_rate', default=0.0001, help='Learning rate for training the model')
    
    return vars(parser.parse_args())

# Training function.

def accuracy(y_pred, y_true):
    y_pred = F.softmax(y_pred,dim = 1)
    _,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))

class Trainer():

    def __init__(self, criterion=None, optimizer=None):
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []
        self.class_name = ['No-Flooded Image', 'Flooded Image']

    def accuracy(self,outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return (preds == labels).sum().item()

    def train(self, model, train_loader):
        model.train()
        print(COLOR.Red, 'Training', COLOR.END)
        train_loss = 0.0
        train_acc = 0
        #for i, data in tqdm(enumerate(train_loader), total=len(train_loader)): 
        #for image, labels in tqdm(train_loader): 
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(image)
            # Calculate the loss.
            loss = self.criterion(outputs, labels)

            # Clear the gradients
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            # Calculate the accuracy.
            train_acc += accuracy(outputs,labels)

        # Calculate Loss and Acc for complete Epoch
        return train_loss / len(train_loader), train_acc / len(train_loader)


    # Validation function.
    def validate(self, model, valid_loader):
        model.eval() 
        print(COLOR.Purple, 'Validation', COLOR.END)
        valid_loss = 0.0
        valid_acc = 0
        cnt=0
        with torch.no_grad():
            for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                cnt+=1
                image, labels = data
                image = image.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(image)
                # Calculate the loss.
                loss = self.criterion(outputs, labels)
                valid_loss += loss.item()
                # Calculate the accuracy.
                valid_acc += accuracy(outputs,labels)
                

        # Calculate Loss and Acc for complete Epoch
        epoch_loss = valid_loss / len(valid_loader)
        epoch_acc = valid_acc / len(valid_loader)
        return epoch_loss, epoch_acc



    def fit(self, model, train_loader, valid_loader):
        print("[INFO] training the network...")
        startTime = time.time() 
        # Start the training.
        valid_min_loss = np.Inf 
        for epoch in range(CFG.epochs):
            print(f"[INFO]: Epoch {epoch+1} of {CFG.epochs}")
            ###########################################################
            train_epoch_loss, train_epoch_acc = self.train(model, train_loader)
            ###########################################################
            valid_epoch_loss, valid_epoch_acc = self.validate(model, valid_loader)

            if valid_epoch_loss <= valid_min_loss:
                print("Valid_loss decreased {} --> {}".format(valid_min_loss, valid_epoch_loss))

            self.train_loss.append(train_epoch_loss)
            self.train_acc.append(train_epoch_acc)
            self.valid_loss.append(valid_epoch_loss)
            self.valid_acc.append(valid_epoch_acc)

            print(f"\t Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"\t Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            print('-'*50)
            time.sleep(5)
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        save_model(CFG.epochs, model, self.optimizer, self.criterion, pretrained, 'weather_4')
        # Save the loss and accuracy plots.
        save_plots(self.train_acc, self.valid_acc, self.train_loss, self.valid_loss, pretrained, 'weather_4')
        print('TRAINING COMPLETE')

    
        

        
if __name__ == '__main__':
    ''''
    --epochs        : The number of epochs to train for.
    --pretrained    : Whenever we pass this flag from the command line, 
                    pretrained EfficientNetB0 weights will be loaded.
    --learning-rate : The learning rate for training. As you might remember, 
                    we will train twice, once with pretrained weights, once without. 
                    Both cases require different learning rates, so, 
                    it is better to control it while executing the training script.
    '''

    # construct the argument parser
    args = arg_parser()
    pretrained = args['pretrained']
    dataset = args['dataset']
    train_path = os.path.join(dataset, 'train')
    valid_path = os.path.join(dataset, 'valid')
    test_path = os.path.join(dataset, 'test')
    check_if_dir_existed(train_path)
    check_if_dir_existed(test_path)
    check_if_dir_existed(valid_path)

    # Load the training and validation datasets.

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    Color = (COLOR.Blue if torch.cuda.is_available() else COLOR.Red)
    print(Color)

    dataset_train, dataset_test, dataset_valid = get_datasets(train_path, test_path, valid_path, pretrained)
    train_loader, test_loader, valid_loader = get_data_loaders(dataset_train, dataset_test, dataset_valid)
    # set the device we will be using to train the model
    print(f"[INFO]: Computation device: \t\t{device}")
    print(f"[INFO]: Learning rate: \t\t\t{CFG.lr}")
    print(f"[INFO]: Epochs to train for: \t\t{CFG.epochs}\n")
    slice_plot(data_loader= train_loader, nrow = 4, ncol= 4, fig_title= "train plot")
    slice_plot(data_loader= valid_loader, nrow = 4, ncol= 4, fig_title= "valid plot")
    slice_plot(data_loader= test_loader, nrow = 4, ncol= 4, fig_title= "test plot")
    #plt.show()
    #for image, labels in tqdm(train_loader): 
    #    print(labels)
    print(COLOR.END)
    
    model = build_efficient_net(pretrained).to(device)
    # Total parameters and trainable parameters.
    ##summary(model,input_size=(3,224,224))
    print(Color)
    #total_params = sum(p.numel() for p in model.parameters())
    #print(f"\tTotal parameters: \t\t{total_params:,}")
    #total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"\tTraining parameters: \t\t{total_trainable_params:,}")
    #print(COLOR.END)
    # Optimizer.
    optimizer = optim.Adam(model.parameters(), CFG.lr)
    # Loss function.
    criterion = nn.CrossEntropyLoss()
    # Lists to keep track of losses and accuracies.

    __MODEL = Trainer(criterion, optimizer)
    __MODEL.fit(model, train_loader, valid_loader)



    # Save the trained model weights.
    #return __MODEL
    

    plt.show()
    
    print('done')
