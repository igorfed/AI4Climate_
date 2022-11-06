import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')


def save_model(epochs, model, optimizer, criterion, pretrained, model_name):
    """
    Function to save the trained model to disk.
    Arguments:
        epochs number of epochs trained for the model
        optimizer - 
        loss function - 
        pretrained - True/ False
    """

    
    
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"{model_name}_{pretrained}.pth")

def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained, figure_name):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss')
    plt.plot(train_acc, label='train acc')
    plt.plot(valid_acc, label='valid loss')
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(f"{figure_name}_{pretrained}.png")


