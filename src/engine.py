import torch
from typing import Tuple, Dict, List
import numpy as np
from tqdm import tqdm
from com.colors import COLOR
import os
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.style.use('ggplot')

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch.nn.functional as F 


import time
import datetime

class LossMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccMeter:
    def __init__(self):
        self.true_count = 0
        self.all_count = 0
        self.avg = 0
        
    def update(self, y_true, y_pred):
        y_true = y_true
        y_pred = y_pred.argmax(axis=1)
        self.true_count += (y_true == y_pred).sum()
        self.all_count += y_true.shape[0]
        self.avg = self.true_count / self.all_count

def accuracy(y_pred,y_true):
    y_pred = F.softmax(y_pred,dim = 1)
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))

class Trainer():

    def __init__(self, 
                model           : torch.nn.Module, 
                model_name      : str,
                device          : str, 
                batch_size      : int,
                criterion       : torch.nn.CrossEntropyLoss,
                optimizer       : torch.optim.Adam,
                epochs          : int,
                learning_rate   : float,
                scheduler       =None, #: torch.optim.lr_scheduler.ReduceLROnPlateau,
                scheduler_params=None
                ):

        self.valid_min_loss = np.inf
        ###################
        self.model = model
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.learning_rate = learning_rate
        ###################
        self.train_loss = []
        self.valid_loss = []
        self.train_acc = []
        self.valid_acc = []
        self.best_train_loss = []
        self.best_valid_loss = []
        self.best_train_acc = []
        self.best_valid_acc = []

        
        if scheduler:
            self.scheduler = scheduler(self.optimizer, **scheduler_params)
           
    def fit(
            self,
            train_loader : torch.utils.data.DataLoader, 
            valid_loader : torch.utils.data.DataLoader, 
            last_checkpoint : str, 
            best_checkpoint : str,
            log_path : str,
            ):#-> Dict[str, List[float]]:

        """Trains and tests a PyTorch model.
        Calculates, prints and stores evaluation metrics throughout.
        Args:
        
        
        train_loader: A DataLoader instance for the model to be trained on.
        valid_loader: A DataLoader instance for the model to be tested on.
        
        Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                      train_acc: [...],
                      test_loss: [...],
                      test_acc: [...]} 
        For example if training for epochs=2: 
                     {train_loss: [2.0616, 1.0537],
                      train_acc: [0.3945, 0.3945],
                      test_loss: [1.2641, 1.5706],
                       test_acc: [0.3400, 0.2973]}
        """
        # Create empty results dictionary
        #results = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

        print(COLOR.Green, "[INFO] training the network...", COLOR.END)
        self.start_time = time.time()
        self.log_path = log_path #f"{self.model_name}_log.txt"


        self.best_epochs = []
        
        # Loop through training and testing steps for a number of epochs
        for epoch in range(self.epochs):

            current_lr = self.optimizer.param_groups[0]['lr']

            self.log(f'\n{datetime.datetime.utcnow().isoformat()}\nLR: {current_lr}')
            ##################TRAIN ONE EPOCH##########################            
            t = int(time.time())
            avg_train_loss, avg_train_acc = self.train(data_loader=train_loader)
            self.log(
                    f"[RESULT]: Train. Epoch {epoch+1} of {self.epochs}" + \
                    f"train_loss: {avg_train_loss:.3f}, train_acc: {avg_train_acc:.3f}" + \
                    f"time: {int(time.time()) - t} s")
            ##################VALID ONE EPOCH##########################            
            t = int(time.time())
            avg_valid_loss, avg_valid_acc = self.validate(data_loader=valid_loader)
            self.log(
                    f"[RESULT]: Valid. Epoch {epoch+1} of {self.epochs}" + \
                    f"valid_loss: {avg_valid_loss:.3f}, train_acc: {avg_valid_acc:.3f}" + \
                    f"time: {int(time.time()) - t} s")

            ###########################################################            
                        ###########################################################            
            self.train_loss.append(avg_train_loss)
            self.train_acc.append(avg_train_acc)
            self.valid_loss.append(avg_valid_loss)
            self.valid_acc.append(avg_valid_acc)

            f_best = 0
            if avg_valid_loss < self.valid_min_loss:
                print(COLOR.Green, "Valid_loss decreased {} --> {}".format(self.valid_min_loss, avg_valid_loss))
                self.valid_min_loss = avg_valid_loss
                self.best_train_loss.append(avg_train_loss)
                self.best_train_acc.append(avg_train_acc)
                self.best_valid_loss.append(avg_valid_loss)
                self.best_valid_acc.append(avg_valid_acc)
                f_best = 1

            self.scheduler.step(metrics=avg_valid_loss)   
            self.save_model(last_checkpoint, self.train_loss, self.train_acc, self.valid_loss, self.valid_acc)
            if f_best:
                self.save_model(best_checkpoint, self.best_train_loss, self.best_train_acc, self.best_valid_loss, self.best_valid_acc)
            # Print out what's happening
            print(f"[INFO]: Train. Epoch {epoch+1} of {self.epochs}")
            print(f"\t Train loss: {avg_train_loss:.3f}, Train acc: {avg_train_acc:.3f}")
            print(f"\t Valid loss: {avg_valid_loss:.3f}, Valid acc: {avg_valid_acc:.3f}")
            
            ###########################################################            
            #f_best = 0
            # Update results dictionary
            #results["train_loss"].append(train_loss)
            #results["train_acc"].append(train_acc)
            #results["valid_loss"].append(valid_loss)
            #results["valid_acc"].append(valid_acc)
            
        
         
    def train(  
                self, 
                data_loader : torch.utils.data.DataLoader
            ) -> Tuple[float, float]:

        """ Trains a PyTorch model for a single epoch.
        Turns a target PyTorch model to training mode and then
        runs through all of the required training steps (forward
        pass, loss calculation, optimizer step).

        Args:
        dataloader: A DataLoader instance for the model to be trained on.
        
        Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:
        (0.1112, 0.8743)
        """

        print(COLOR.Green, '\t  # Put model in train mode', COLOR.END)
        self.model.train()
        # Setup train loss and train accuracy values
        train_loss  = 0
        train_acc   = 0

        for data in tqdm(data_loader):

            images, labels = data
            # Send data to target device
            images = images.to(self.device)
            labels = labels.to(self.device)
            # 1. Forward pass
            y_pred = self.model(images) 
        
            # 2. Calculate and accumulate loss
            loss = self.criterion(y_pred, labels)
            train_loss += loss.item()

            # 3. Optimizer zero grad
            self.optimizer.zero_grad()

            # 4. loss backward
            loss.backward()

            # 5. Optimizer step
            self.optimizer.step()
            train_acc += accuracy(y_pred,labels)

            # Calculate and accumulate accuracy metric across all batches
            #y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            #train_acc += (y_pred_class == labels).sum().item()/len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch 
        
        return  train_loss / len(data_loader), train_acc / len(data_loader)



    def validate(
                self,
                data_loader : torch.utils.data.DataLoader
            ) -> Tuple[float, float]:
        """Tests a PyTorch model for a single epoch.
        Turns a target PyTorch model to "eval" mode and then performs
        a forward pass on a testing dataset.

        Args:
        dataloader: A DataLoader instance for the model to be tested on.
        
        Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
        """
        print(COLOR.Blue, '\t  # Put model in eval mode', COLOR.END)
        self.model.eval()         
        # Setup test loss and test accuracy values
        valid_loss =0
        valid_acc = 0
        with torch.no_grad():
            for data in tqdm(data_loader):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                # 1. forward pass
                outputs = self.model(images)

                # 2. Calculate and accumulate loss
                loss = self.criterion(outputs, labels)
                valid_loss += loss.item()

                # Calculate and accumulate accuracy metric across all batches

                valid_acc += accuracy(outputs,labels)

                #test_pred_labels = outputs.argmax(dim=1)
                #test_acc += ((test_pred_labels == labels).sum().item()/len(test_pred_labels))

        # Adjust metrics to get average loss and accuracy per batch 
        return valid_loss / len(data_loader), valid_acc / len(data_loader)

    def log(self, message):
        print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
    
    def save_model(
                    self, 
                    path: str,
                    train_loss, train_acc, valid_loss, valid_acc):
        """Saves a PyTorch model to a target directory.

        """
        print(COLOR.Green, f"[INFO] Saving model to: {path}", COLOR.END)
        self.model.eval()

        torch.save({
            'model_state_dict': self.model.state_dict(),

            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            
            'model_name': self.model_name,
            
            'train_loss': train_loss,
            'train_acc': train_acc,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc
        }, path)
        print(COLOR.Red, f"[INFO] model saved", COLOR.END)
        
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.model_name = checkpoint['model_name']

        self.train_loss = checkpoint['train_loss']
        self.valid_loss = checkpoint['valid_loss']
        self.train_acc  = checkpoint['train_acc']
        self.valid_acc  = checkpoint['valid_acc']
        
        print(COLOR.Yellow, f"[INFO] model loaded", COLOR.END)

    def save_plots(self, train_loss, train_acc, valid_loss, valid_acc):
        """
        Function to save the loss and accuracy plots to disk.
        """
        # accuracy plots
        # loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(train_loss, label='Train loss')
        plt.plot(valid_loss, label='Valid loss')
        plt.plot(train_acc, label='Train acc')
        plt.plot(valid_acc, label='Valid loss')
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        


def test(model, device,  criterion, data_loader : torch.utils.data.DataLoader
            ) -> Tuple[float, float]:
        print(COLOR.Blue, '\t  # Put model in eval mode', COLOR.END)
        model.eval()         
        # Setup test loss and test accuracy values
        test_loss =0
        test_acc = 0
        with torch.no_grad():
            for data in tqdm(data_loader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                # 1. forward pass
                outputs = model(images)

                # 2. Calculate and accumulate loss
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Calculate and accumulate accuracy metric across all batches

                test_acc += accuracy(outputs,labels)

                #test_pred_labels = outputs.argmax(dim=1)
                #test_acc += ((test_pred_labels == labels).sum().item()/len(test_pred_labels))

        # Adjust metrics to get average loss and accuracy per batch 
        return test_loss / len(data_loader), test_acc / len(data_loader)