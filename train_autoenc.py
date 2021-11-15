from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from autoenc import AutoEncoder 
import argparse
import numpy as np 
import matplotlib.pyplot as plt


def training_plots(train_losses,test_losses,title):
    #Loss plot
    plt.plot(train_losses,'-ob',label='Train Loss')
    plt.plot(test_losses,'-xg',label='Test Loss')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Loss")
    plt.savefig('loss.png')


def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)

        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # ======================================================================
        # Compute loss based on criterion
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Remove NotImplementedError and assign correct loss function.
        loss = criterion(output,data.view(batch_size,-1))
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        
    train_loss = float(np.mean(losses))
    print("Epoch: ",epoch)
    print('Train set- Average loss: {:.4f}'.format(float(np.mean(losses))))
    return train_loss
    


def test(model, device, test_loader,criterion,epoch,batch_size):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            
            # Predict for data by doing forward pass
            output = model(data)
            
            #data=data.type(torch.LongTensor).to(device)

            # Compute loss based on same criterion as training 
            loss = criterion(output,data.view(batch_size,-1))
            
            # Append loss to overall test loss
            losses.append(loss.item())
           
    test_loss = float(np.mean(losses))
    
    print("Epoch: ",epoch)
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    
    return test_loss
    

def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Initialize the model and send to device 
    model = AutoEncoder(FLAGS.mode).to(device)
    
    criterion = nn.MSELoss()
    
    
    optimizer = optim.SGD(model.parameters(),lr=FLAGS.learning_rate,momentum=0.9)
        
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                       transform=transform)
    train_loader = DataLoader(dataset1, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4)
        
    train_losses=[]
    test_losses=[]
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        train_loss = train(model, device, train_loader,
                                            optimizer, criterion, epoch, FLAGS.batch_size)
        test_loss = test(model, device, test_loader, criterion, epoch,FLAGS.batch_size)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    torch.save(model.state_dict(), 'model.pt')
    print("Final Epoch Mean Test Loss is {:2.2f}".format(test_loss))
    print("Final Epoch Mean Train Loss is {:2.2f}".format(train_loss))

    
    print("Training and evaluation finished")
    print(train_losses)
    print(test_losses)
    
    #Get train and test loss and accuracy plots for each training iter
    training_plots(train_losses,test_losses,FLAGS.mode+' AutoEncoder')

if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('AutoEncoder Exercise.')
    parser.add_argument('--mode',
                        type=str, default='Lin',
                        help='Select mode between Lin/Conv')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
