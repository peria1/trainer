# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:32:52 2019

@author: Bill
"""

# -*- coding: utf-8 -*-billUti
"""
Created on Sun Oct  6 07:32:17 2019

@author: Bill


"""
import torch
#import numpy as np
import torch.utils.data
from torch import nn

class one_linear_layer(nn.Module):
    def __init__(self, npts=None, nbatch=None):
        super(one_linear_layer, self).__init__()

        if npts is None:
            self.npts = 50
        if nbatch is None:
            self.nbatch = 128
        
        self.L1 = nn.Linear(self.npts,1)
       
    def forward(self,x):
        return torch.squeeze(self.L1(x))

     
class bisect_to_one(nn.Module):
    def __init__(self, npts=None, nbatch=None):  # trying to see if machine can tell that y is the sum over x 
        super(bisect_to_one, self).__init__()

        if npts is None:
            npts = 64
        if nbatch is None:
            nbatch = 128
        
        try:
            assert((npts != 0) and (npts & (npts-1) == 0))
        except AssertionError:
            print('Number of points in each example must be a power of two')
        
        self.npts = npts
        self.nbatch = nbatch
        
        n = npts*2
        self.layer_list = nn.ModuleList([nn.Linear(npts,n)])
        while n > 2:
            self.layer_list.append(nn.Linear(n,n//2))
            n//=2
        self.leaky = nn.LeakyReLU()

    def forward(self, xy):
        dataflow = xy
        for L in self.layer_list:
            dataflow = self.leaky(L(dataflow))
        
        return dataflow



class n_double_n(nn.Module): # moved to n_double_n
    def __init__(self, npts=None, nbatch=None): 
        super(n_double_n, self).__init__()

        if npts is None:
            npts = 50
        if nbatch is None:
            nbatch = 128
        
        self.npts = npts
        self.nbatch = nbatch
        
        self.L1 = nn.Linear(npts, 2*npts)
        self.L2 = nn.Linear(2*npts, 2*npts)
        self.L3 = nn.Linear(2*npts, 2*npts)
        self.L4 = nn.Linear(2*npts, 2*npts)
        self.L5 = nn.Linear(2*npts, 2*npts)
        self.L6 = nn.Linear(2*npts, 2*npts)
        self.L7 = nn.Linear(2*npts, 2*npts)
        self.Llast = nn.Linear(2*npts, npts)
        self.leaky = nn.LeakyReLU()


    def forward(self, x):
        dataflow = self.leaky(self.L1(x))
        dataflow = self.leaky(self.L2(dataflow))
        dataflow = self.leaky(self.L3(dataflow))
        dataflow = self.leaky(self.L4(dataflow))
        dataflow = self.leaky(self.L5(dataflow))
        dataflow = self.leaky(self.L6(dataflow))
        dataflow = self.leaky(self.L7(dataflow))
        dataflow = self.leaky(self.Llast(dataflow))
        
        return dataflow

class n_double_one(nn.Module):  # moved to n_double_one
    def __init__(self, npts=None, nbatch=None):  # trying to see if machine can tell that y is the sum over x 
        super().__init__()

        self.custom_loss = nn.L1Loss()
        
        if npts is None:
            npts = 50
        if nbatch is None:
            nbatch = 128
        
        self.npts = npts
        self.nbatch = nbatch
        width_factor = 2
#        self.bn1 = nn.BatchNorm1d(npts)
        self.L1 = nn.Linear(npts, width_factor*npts)
        self.L2 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L3 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L4 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L5 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L6 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L7 = nn.Linear(width_factor*npts, width_factor*npts)
        self.Llast = nn.Linear(width_factor*npts, npts)
        
        self.weight_vector = nn.Linear(npts, 1) # npts is the size of each 1D example
        
        self.leaky = nn.LeakyReLU()

        
    def forward(self, x):
        dataflow = x
        dataflow = self.leaky(self.L1(dataflow))
        dataflow = self.leaky(self.L2(dataflow))
        dataflow = self.leaky(self.L3(dataflow))
        dataflow = self.leaky(self.L4(dataflow))
        dataflow = self.leaky(self.L5(dataflow))
        dataflow = self.leaky(self.L6(dataflow))
        dataflow = self.leaky(self.L7(dataflow))
        dataflow = self.leaky(self.Llast(dataflow))
        result = self.weight_vector(dataflow)
        return result



class n_double_one_tanh(nn.Module):    # moved to n_double_one_tanh
    def __init__(self, npts=None, nbatch=None):  # trying to see if machine can tell that y is the sum over x 
        super(n_double_one_tanh, self).__init__()

        if npts is None:
            npts = 50
        if nbatch is None:
            nbatch = 128
        
        self.npts = npts
        self.nbatch = nbatch
        
        self.L1 = nn.Linear(npts, 2*npts)
        self.L2 = nn.Linear(2*npts, 2*npts)
        self.L3 = nn.Linear(2*npts, 2*npts)
        self.L4 = nn.Linear(2*npts, 2*npts)
        self.L5 = nn.Linear(2*npts, 2*npts)
        self.L6 = nn.Linear(2*npts, 2*npts)
        self.L7 = nn.Linear(2*npts, 2*npts)
        self.Llast = nn.Linear(2*npts, npts)
        
        self.weight_vector = nn.Linear(npts, 1) # npts is the size of each 1D example

    def forward(self, x):
        dataflow = torch.relu(self.L1(x))
        dataflow = torch.relu(self.L2(dataflow))
        dataflow = torch.relu(self.L3(dataflow))
        dataflow = torch.relu(self.L4(dataflow))
        dataflow = torch.relu(self.L5(dataflow))
        dataflow = torch.relu(self.L6(dataflow))
        dataflow = torch.relu(self.L7(dataflow))
        dataflow = torch.relu(self.Llast(dataflow))
        dataflow = torch.squeeze(self.weight_vector(dataflow))
        result = torch.tanh(dataflow)
        return result



class vectorVAE(nn.Module):   # moved to vectorVAE
    def __init__(self,dim=2,npts=50, nbatch=128):  # this sets up 5 linear layers
        super(vectorVAE, self).__init__()

        self.npts = npts
        self.nbatch = nbatch        
        self.z_dimension = dim
#        self.register_backward_hook(grad_hook)
        self.fc1 = nn.Linear(npts, npts) # stacked MNIST to 400
        self.fc21 = nn.Linear(npts, self.z_dimension) # two hidden low D
        self.fc22 = nn.Linear(npts, self.z_dimension) # layers, same size
        self.fc3 = nn.Linear(self.z_dimension, npts)  
        self.fc4 = nn.Linear(npts, npts)

    def encode(self, x): 
        from torch.nn import functional as F
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # <- Stochasticity!!!
        # How can the previous line allow back propagation? The question is, does the 
        #   loss function depend on eps? For starters, it returns a normal 
        #   with the same dimensions as std, rather than one that has the variances
        #   implied by std. That's why we scale by std, below, when returning. 
        #   Because we are only using the dimensions of std to get eps, does that mean
        #   that the loss function is independent of eps? I am missing something. 
        #
        return mu + eps*std

    def decode(self, z):
        from torch.nn import functional as F
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
#        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z) #, mu, logvar



#--------------------
