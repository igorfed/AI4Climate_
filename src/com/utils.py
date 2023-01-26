import torch 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.style.use('ggplot')
from models import efficientnet
from models import densenet
import com.config as config
import math
import os
import time
from com.colors import COLOR
from com.common_packages import check_if_file_existed
import sys


#from sklearn import decomposition
#from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
timestr = time.strftime("%Y%m%d-%H%M%S")

def slice_plot(data_loader, nrow, ncol, fig_title):
        mean, std = torch.FloatTensor(config.CFG.mean), torch.FloatTensor(config.CFG.mean)
        fig = plt.figure(fig_title, figsize=(30, 20), dpi=80)
        data_iter = iter(data_loader)
        images, labels = data_iter.next()
        k=0
        for i in range(nrow):
            for j in range(ncol):
                k = k+1
                print(i+1, j+1, k, type(images[i+j]), images[i+j].shape, type(labels[i+j]), labels[i+j].shape )
                ax = fig.add_subplot(nrow, ncol, k)
                img = images[i+j].permute(1,2,0)*std + mean
                ax.imshow(np.asarray(img))
                ax.set_aspect('equal')
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                textstr = '\n'.join((
                    r'Batch size    : %s' % (config.CFG.batch_size,),        
                    r'Image in batch: %s' % (k-1),
                    r'Label         : %s' % (labels[i+j].numpy()),
                    ))
                ax.set_title(CFG.class_name[labels[i+j].numpy()], fontsize=10, color='red' if labels[i+j].numpy() == 1 else 'green')
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        #print(data_loader.classes, data_loader.class_to_idx)

def save_model(epochs, model, output, optimizer, criterion, pretrained):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"{output}/model_pretrained_{pretrained}.pth")



def save_plots(train_acc, valid_acc, train_loss, valid_loss, enable_plot, figure_name=None, figure_save=False):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss')
    plt.plot(train_acc, label='train acc')
    plt.plot(valid_acc, label='valid acc')
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    if figure_save:
        plt.savefig(f"{figure_name}.png")
    if enable_plot:
        plt.show()


def check_if_dir_existed(dir_name, create=False):
	if not os.path.exists(dir_name):
		print(f'Folder \t\t: {dir_name} is not available')
		if create:
			os.mkdir(dir_name)
			print(f'Folder \t\t: {dir_name} created')	
	else:
		print(f'Folder \t\t: {dir_name} is available')

def show_batch_grid(
                dataloader  : torch.utils.data.DataLoader, 
                balance     : bool, 
                mode        : str,
                enable_plot : bool,  
                figure_save : bool):
    """
    plot grid of dataloader
    """                
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    num_image_in_class0_fig, num_image_in_class1_fig = 0, 0
    num_image_in_class0_fig += torch.sum(labels==0)
    num_image_in_class1_fig += torch.sum(labels==1)

    if balance:
        b = 'Balanced'
    else:
        b = 'UnBalanced'
    
    figure_name =  f'{mode} : {b} Images in [Class0, Class1] :\t[{num_image_in_class0_fig},{num_image_in_class1_fig}]'
    
    fig = plt.figure(figure_name, figsize=(30, 20), dpi=80)
    k = 0
    mean, std = torch.FloatTensor(config.CFG.mean), torch.FloatTensor(config.CFG.std)
    I = int(math.sqrt(config.CFG.batch_size))
    J = int(config.CFG.batch_size/ I)
    for _ in range(I):
        for _ in range(J):
            k = k +1
            ax = fig.add_subplot(I, J+1, k)
            ax.imshow(images[k-1].permute(1, 2, 0)*std + mean)
            ax.set_aspect('equal')
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            textstr = '\n'.join((
                    r'Batch size    : %s' % (config.CFG.batch_size,),        
                    r'Image in batch: %s' % (k-1),
                    r'Label         : %s' % (labels[k-1].numpy()),
                    r'Class         : %s' % (config.CFG.class_name[labels[k-1].numpy()]),
                    ))
            ax.set_title(str(k) + ':' + str(config.CFG.batch_size) + ' ' + config.CFG.class_name[labels[k-1].numpy()], fontsize=12, color='red' if labels[k-1].numpy() == 1 else 'green')
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
            ax.grid(False)
    ax = fig.add_subplot(I, J+1, k+1)
    textstr = '\n'.join((
        r'Batch size    : %s' % (config.CFG.batch_size,),        
        r'Img in Class0 : %s' % (num_image_in_class0_fig),
        r'Img in Class1 : %s' % (num_image_in_class1_fig),
        ))
    ax.set_title('Number of Classes in batch', fontsize=12)
    ax.bar(x = ['Class0', 'Class1'], height=[num_image_in_class0_fig.numpy(), num_image_in_class1_fig.numpy()])
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    if figure_save:
        figures = 'figures'
        figures = os.path.join(os.getcwd(), figures)
        check_if_dir_existed(figures, True)
        
        f = f"{figures}/{b}_{mode}_{timestr}.png"
        print(f)
        plt.savefig(f)
        check_if_file_existed(f)

    

def convertImages(source, destination):
    import os,cv2
    from skimage import io
    source = r'/media/igofed/SSD_2T/DATASETS/PetImages/train/Cat' # Source Folder
    destination = r'/media/igofed/SSD_2T/DATASETS/PetImages/train1/Cat' # Destination 



    try:
        os.makedirs(destination)
    except:
        print ("Directory already exist, images will be written in asme folder")

    # Folder won't used
    files = os.listdir(source)
    i = 0
    for image in files:
        print(len(files), i, "\t", image)
        img = cv2.imread(os.path.join(source,image))
#        if img is not None:
#            if(len(img.shape)<3):
#                print ('gray')
#        elif len(img.shape)==3:
#            print ('Color(RGB)')
#    else:
#        print("ERROR")
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        image_name = os.path.splitext(image)[0]+'.png'
        cv2.imwrite(image_name,gray)
#    i =i+1



def build_model(device,version="efficientnet_b0"):

    model = efficientnet.EfficientNet(
        version=version,
        num_classes=2,
    ).to(device)

    #total_params= sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model#, total_params

def define_loss(device) -> torch.nn.CrossEntropyLoss:
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device, memory_format=torch.channels_last)
    return criterion

def define_optimizer(model) -> torch.optim.SGD:
    optimizer = torch.optim.SGD(model.parameters(), lr=config.CFG.learning_rate)
    return optimizer



# Select GPU or CPU device
def select_device(gpu):
    """Select GPU or CPU device for training or prediction.
       Notify user if GPU is requested but not available and exit program.
    
    Args:
        gpu (bool): selects GPU if True, otherwise CPU
        
    Returns:
        device (torch.device): 'cpu' or 'cuda:0'
    """
    
    # Check if GPU is requested it is available
    if gpu:
        assert torch.cuda.is_available(), 'Error: Requested GPU, but GPU is not available.'
        
    # Select device
    device = torch.device('cuda:0') if gpu else torch.device('cpu')  
    
    return device



     


def plot_confusion_matrix(labels, pred_labels, classes, model_name, enable_plot, figure_save):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, display_labels=classes)
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    plt.xticks(rotation=20)     

    if figure_save:
        figures = 'figures'
        figures = os.path.join(os.getcwd(), figures)
        check_if_dir_existed(figures, True)
        
        f = f"{figures}/confusion_{model_name}_{timestr}.png"
        print()
        plt.savefig(f)
        check_if_file_existed(f)
    if enable_plot:
        plt.show()

def plot_most_incorrect(fname, incorrect, classes, n_images, model_name, normalize=True, figure_save=True):
    def normalize_image(image):
        image_min = image.min()
        image_max = image.max()
        image.clamp_(min=image_min, max=image_max)
        image.add_(-image_min).div_(image_max - image_min + 1e-5)
        return image

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    figure_name =  f'{fname}'
    fig = plt.figure(figure_name, figsize=(30, 20), dpi = 80)

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)

        image, true_label, probs = incorrect[i]
        image = image.permute(1, 2, 0)
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim=0)
        true_class = classes[true_label]
        incorrect_class = classes[incorrect_label]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.cpu().numpy())
        ax.set_title(f'true label: {true_class} ({true_prob:.3f})\n'
                     f'pred label: {incorrect_class} ({incorrect_prob:.3f})')

        #ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

        ax.grid(False)
    fig.subplots_adjust(hspace=0.4)

    if figure_save:
        figures = 'figures'
        figures = os.path.join(os.getcwd(), figures)
        check_if_dir_existed(figures, True)
        
        f = f"{figures}/{fname}_{model_name}_{timestr}.png"
        print(f)
        plt.savefig(f)
        check_if_file_existed(f)
    