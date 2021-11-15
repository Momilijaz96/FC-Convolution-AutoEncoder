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
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from autoenc import AutoEncoder


def test_autoenc(model,device,n,class_id):
  """
  Function to return reconstructed and true images for
  n samples from each class.
  model=Trained model, already on device
  data_loader=tet or train loader
  n=number of samples from given class 
  class_id = id of class for which to return reconstructed samples
  """
  batch_size=20
  test_loader = DataLoader(dataset2, batch_size =batch_size, shuffle=False, num_workers=4)

  #Set model to eval mode to notify all layers.
  model.eval()
  for batch_idx, sample in enumerate(test_loader):
    data, target = sample
    data, target = data.to(device), target.to(device)
    
    #Collect two samples for given class
    idx=np.where(target.numpy()==class_id)[0] #index of class_id in target of current batch
    if len(idx)>0:
      try:
        test_samples
        test_samples=np.append(test_samples,data[idx].numpy(),axis=0)
        if test_samples.shape[0]>=n:
          break
      except NameError:
        test_samples=data[idx].numpy()

  with torch.no_grad():
    out=model(torch.tensor(test_samples[:n+1]))

  return out,test_samples


if __name__ == "__main__":
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  PATH='Conv_AutoEncoder_model.pt'
  mode='Conv' #Update mode
  model = AutoEncoder(mode=mode) 
  #Change mode of the model as Conv for Convoutional Encoder/Decoder
  #and Lin for Linear enconder decoder architecture
  model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu'))) #remove map_location if u have GPU on board
  model.eval()
  for class_id in range(10):
    pred,target=func(model,device,2,class_id)
    plt.figure()
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(target[0].reshape(28,28)) #row=0, col=0
    ax[0,0].title.set_text('Original')
    ax[1, 0].imshow(target[1].reshape(28,28)) #row=0, col=1
    ax[0, 1].imshow(pred[0].reshape(28,28)) #row=1, col=0
    ax[0,1].title.set_text('Reconstructed')
    ax[1, 1].imshow(pred[1].reshape(28,28)) #row=1, col=1

    plt.suptitle("Reconstruct Results for "+str(class_id))
    plt.save(str(class_id)+'_'+mode+'.png');

