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

#from dataset import get_datasets
#from dataset import get_data_loaders
#from dataset import show_batch_grid
from models import efficientnet
from models import densenet
import config
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False
#from dataset import load_dataset

from dataset import load_dataset
import numpy as np
import utils
print(config.CFG.epochs)
import engine


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",   "--path",       type=str, required=False, help="path to the dataset")
    parser.add_argument("-d",   "--dataset",    type=str, required=False, help="type of the dataset")
    parser.add_argument("-md",  "--model",      type=str, required=False, help="type of model")

    return vars(parser.parse_args())


if __name__ == '__main__':
    '''
    '''
    print('--------------------------------')
    args = arg_parser()
    print(args)
    print('--------------------------------')
    DATA = args['path']
    dataset = args['dataset']
    model_name = args['model']

    train_path = os.path.join(dataset, 'train')
    valid_path = os.path.join(dataset, 'valid')
    #test_path = os.path.join(dataset, 'test')
    model_name = args['model']
    check_if_dir_existed(train_path)
    #check_if_dir_existed(test_path)
    check_if_dir_existed(valid_path)
    output_paths = os.path.join(os.getcwd(), 'outputs')
    check_if_dir_existed(output_paths, True)
    last_checkpoint = f"{output_paths}/{args['model']}_{args['dataset']}_{'last_checkpoint'}.bin"
    best_checkpoint = f"{output_paths}/{args['model']}_{args['dataset']}_{'best_checkpoint'}.bin"
    figure_name_last = f"{output_paths}/{args['model']}_{args['dataset']}_{'last_checkpoint'}.png"
    figure_name_best = f"{output_paths}/{args['model']}_{args['dataset']}_{'best_checkpoint'}.png"

    log_path = f"{output_paths}/{args['model']}_{args['dataset']}_{'log'}.txt"
    
    print('###########################')
    print(last_checkpoint)
    print(best_checkpoint)
    print(log_path)
    print(figure_name_last)
    print(figure_name_best)
    print('###########################')    


    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    Color = (COLOR.Blue if torch.cuda.is_available() else COLOR.Red)
    
    train_set, train_loader = load_dataset(train_path, mode='train', balance_dataset=True)
    valid_set, valid_loader = load_dataset(train_path, mode='valid', balance_dataset=True)
    
    model, total_params = utils.build_model(device, version=model_name)
    print(f"[INFO]:{model_name} :{total_params} parameters")
    model.to(device)
    criterion = utils.define_loss(device)
    print(COLOR.BGreen, "Define all loss functions successfully.", COLOR.END)
    optimizer = utils.define_optimizer(model)
    print(COLOR.BGreen,"Define all optimizer functions successfully.", COLOR.END)
    utils.show_batch_grid(train_loader, True, 'train', figure_save=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )
    __trainer = engine.Trainer(
                        model, 
                        model_name, 
                        device, 
                        batch_size=config.CFG.batch_size,
                        criterion=criterion, 
                        optimizer=optimizer, 
                        epochs=config.CFG.epochs, 
                        learning_rate=config.CFG.learning_rate, 
                        scheduler=scheduler, 
                        scheduler_params=scheduler_params
                        )

    print('Train done')
    __trainer.fit(train_loader, valid_loader, last_checkpoint, best_checkpoint, log_path)


    model_A, total_params_A = utils.build_model(device, version=model_name)
    checkpoint = torch.load(last_checkpoint)
    model_A.load_state_dict(checkpoint['model_state_dict'])
    train_loss = checkpoint['train_loss']
    train_acc = checkpoint['train_acc']
    valid_loss = checkpoint['valid_loss']
    valid_acc = checkpoint['valid_acc']
    utils.save_plots(train_acc, valid_acc, train_loss, valid_loss, figure_name=last_checkpoint)



    model_B, total_params_B = utils.build_model(device, version=model_name)
    checkpoint_B = torch.load(best_checkpoint)
    model_A.load_state_dict(checkpoint_B['model_state_dict'])
    train_loss_B = checkpoint_B['train_loss']
    train_acc_B = checkpoint_B['train_acc']
    valid_loss_B = checkpoint_B['valid_loss']
    valid_acc_B = checkpoint_B['valid_acc']
    utils.save_plots(train_acc_B, valid_acc_B, train_loss_B, valid_loss_B, figure_name=best_checkpoint)

    plt.show()

