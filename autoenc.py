import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self,mode):
        super(Encoder,self).__init__()
        self.layer1=nn.Linear(784,256)
        self.layer2=nn.Linear(256,128)
        self.conv1=nn.Conv2d(in_channels=1, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.conv2=nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1)

        if mode=='Lin':
            self.forward=self.linear
        elif mode=='Conv':
            self.forward=self.conv

    def linear(self,x):
        #Flatten data
        x=x.view(-1,np.prod(x.size()[1:]))
        x=F.relu(self.layer1(x))
        return F.relu(self.layer2(x))

    def conv(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),2,1)
        x=F.max_pool2d(F.relu(self.conv2(x)),2,1)
        return x

class Decoder(nn.Module):
    def __init__(self,mode):
        super(Decoder,self).__init__()
        self.layer1=nn.Linear(128,256)
        self.layer2=nn.Linear(256,784)
        self.conv1=nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.conv2=nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.conv3=nn.Conv2d(in_channels=80, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.upsample=nn.Upsample(scale_factor=2, mode='bilinear')
        self.fc=nn.Linear(24*48*80,784)
        if mode=='Lin':
            self.forward=self.linear
        elif mode=='Conv':
            self.forward=self.conv

    def linear(self,x):
        #Flatten data
        x=x.view(-1,np.prod(x.size()[1:]))
        x=F.relu(self.layer1(x))
        return F.relu(self.layer2(x))

    def conv(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),2,1)
        x=F.max_pool2d(F.relu(self.conv2(x)),2,1)
        x=self.upsample(x)
        x=x.view(-1,np.prod(x.size()[1:]))
        return F.relu(self.fc(x))


class AutoEncoder(nn.Module):
    def __init__(self,mode):
        super(AutoEncoder,self).__init__()
        self.encoder=Encoder(mode)
        self.decoder=Decoder(mode)

    def forward(self,x):
        latent_vector=self.encoder(x)
        return self.decoder(latent_vector)