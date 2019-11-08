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
import numpy as np
import torch.utils.data
from torch import nn

class opp_cos(nn.Module):
    def __init__(self, npts=None, nbatch=None):
        super(opp_cos, self).__init__()
        
        if npts is None:
            self.npts = 50
        if nbatch is None:
            self.nbatch = 128

        print(self.npts)
        
        self.L1 = nn.Linear(self.npts,self.npts*2)
        self.L2 = nn.Linear(self.npts*2,self.npts*2)
        self.L3 = nn.Linear(self.npts*2,self.npts)
        self.leaky = nn.LeakyReLU()
    
    def forward(self,x):
        dataflow = self.leaky(self.L1(x))
        dataflow = self.leaky(self.L2(dataflow))
        dataflow = self.leaky(self.L3(dataflow))
        
        return dataflow
    
    def get_xy_batch(self):
        nbatch = self.nbatch
        npts = self.npts
        
        x = np.random.normal(size=(nbatch,npts))
        x = np.cos(x)
        x = torch.from_numpy(x)
        x = x.to(torch.float32)
    
        y = -x.to(torch.float32)
        
        return x,y
   

class plain_sum(nn.Module):
    def __init__(self, npts=None, nbatch=None):
        super(plain_sum, self).__init__()

        if npts is None:
            self.npts = 50
        if nbatch is None:
            self.nbatch = 128
        
        self.L1 = nn.Linear(self.npts,1)
       
    def forward(self,x):
        return torch.squeeze(self.L1(x))

    def get_xy_batch(self):
        nbatch = self.nbatch
        npts = self.npts
        x = np.random.normal(size=(nbatch,npts))
    
        x = torch.from_numpy(x)
        y = torch.sum(x,1).to(torch.float32)
        
        x = x.to(torch.float32)
        
        return x,y
   

class quadsum(nn.Module):
    def __init__(self, npts=None, nbatch=None):  # trying to see if machine can tell that y is the sum over x 
        super(quadsum, self).__init__()

        if npts is None:
            npts = 50
        if nbatch is None:
            nbatch = 128
        
        self.npts = npts
        self.nbatch = nbatch
        
        self.L1 = nn.Linear(npts, 2*npts)
        self.L2 = nn.Linear(2*npts, 2*npts)
        self.L3 = nn.Linear(2*npts, npts)
        
        self.weight_vector = nn.Linear(npts, 1) # npts is the size of each 1D example
                
    def forward(self, x):
        dataflow = torch.relu(self.L1(x))
        dataflow = torch.relu(self.L2(dataflow))
        dataflow = torch.relu(self.L3(dataflow))
        result = self.weight_vector(dataflow)
        return torch.squeeze(result)
       
    def get_xy_batch(self):
        nbatch = self.nbatch
        npts = self.npts
        xrange = 20.
        noiseamp = 0.3
        noise = noiseamp * torch.randn(nbatch)
        
        x = torch.from_numpy(np.random.uniform(-xrange,xrange,size=(nbatch,npts))).to(torch.float32)
        y = torch.sum(x.pow(2),1).pow(0.5)
#        print('y size is ',y.size())
    
        y = y + noise
        return x,y


class linreg(nn.Module):
    def __init__(self, npts=None, nbatch=None):  # trying to see if machine can tell that y is the sum over x 
        super(linreg, self).__init__()

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
            
#        self.L1 = nn.Linear(npts, 2*npts)
#        self.L2 = nn.Linear(2*npts, 2*npts)
#        self.L3 = nn.Linear(2*npts, 2*npts)
#        self.L4 = nn.Linear(2*npts, 2*npts)
#        print(type(npts//2))
#        self.L5 = nn.Linear(2*npts, npts//2)
#        self.L6 = nn.Linear(npts//2, npts//4)
#        self.L7 = nn.Linear(npts//4, npts//8)
#        self.L8 = nn.Linear(npts//8, npts//4)
#        self.L9 = nn.Linear(npts//4, npts//2)
#        self.L10 = nn.Linear(npts//2, npts)
#        self.Llast = nn.Linear(npts, npts)
#        
#        self.weight_vector = nn.Linear(npts, 1) # npts is the size of each 1D example

    def forward(self, xy):
        dataflow = xy
        for L in self.layer_list:
            dataflow = torch.relu(L(dataflow))
        
        return dataflow
        
#        dataflow = torch.sigmoid(self.L1(x))
#        dataflow = torch.sigmoid(self.L2(dataflow))
#        dataflow = torch.sigmoid(self.L3(dataflow))
#        dataflow = torch.relu(self.L4(dataflow))
#        dataflow5 = torch.relu(self.L5(dataflow))
#        dataflow6 = torch.relu(self.L6(dataflow5))
#        dataflow7 = torch.relu(self.L7(dataflow6))
#        dataflow = torch.relu(self.L8(dataflow7))
#        dataflow = torch.relu(self.L9(dataflow))
#        dataflow = torch.relu(self.L10(dataflow))
#        result = torch.relu(self.Llast(dataflow))
        
#        return result

    def get_xy_batch(self):
        nbatch = self.nbatch
        npts = self.npts
        
        half = int(npts/2)
        xsize = (nbatch,half)
        
        xrange = 20.0
        sloperange = 10.0
        
        noiseamp = np.random.uniform(low=xrange/100.0, high = xrange/5.0, size=(nbatch,1))
        noise = np.random.normal(scale=noiseamp,size=xsize)
        slope = np.random.uniform(-sloperange,sloperange,size=(nbatch,1))
        offset = np.random.uniform(-sloperange,sloperange,size=(nbatch,1))
        
        x = np.random.uniform(low=-xrange, high=xrange, size=xsize)
        y = (x - offset) * slope
        y = y + noise
        
        x = torch.from_numpy(x).to(torch.float)
        y = torch.from_numpy(y).to(torch.float)
        xy = torch.cat((x,y),1)
        
        labels = torch.cat((torch.from_numpy(slope),torch.from_numpy(offset)),1).to(torch.float)
        
        return xy, labels
 

class prop(nn.Module):
    def __init__(self, npts=None, nbatch=None): 
        super(prop, self).__init__()

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
        while n >= 2:
            self.layer_list.append(nn.Linear(n,n//2))
            n//=2

    def forward(self, xy):
        dataflow = xy
        for L in self.layer_list:
            dataflow = torch.relu(L(dataflow))
        
        return dataflow
        
    def get_xy_batch(self):
        nbatch = self.nbatch
        npts = self.npts
        
        half = int(npts/2)
        xsize = (nbatch,half)
        
        xrange = 20.0
        sloperange = 10.0
        
        noiseamp = np.random.uniform(low=xrange/100.0, high = xrange/20.0, size=(nbatch,1))
        noise = np.random.normal(scale=noiseamp,size=xsize)
        slope = np.random.uniform(-sloperange,sloperange,size=(nbatch,1))
        
        x = np.random.uniform(low=-xrange, high=xrange, size=xsize)
        y = x * slope
        y = y + noise
        
        x = torch.from_numpy(x).to(torch.float)
        y = torch.from_numpy(y).to(torch.float)
        xy = torch.cat((x,y),1)
        
        slopes = torch.from_numpy(slope).to(torch.float)
        
        return xy, slopes


class cumulative_sum(nn.Module):
    def __init__(self, npts=None, nbatch=None): 
        super(cumulative_sum, self).__init__()

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
        
    def get_xy_batch(self):
        nbatch = self.nbatch
        npts = self.npts
        xsize = (nbatch,npts)

        xrange = 20.0
        x = np.random.uniform(low=-xrange, high=xrange, size=xsize)
        y = np.cumsum(x,axis=1)
        x = torch.from_numpy(x).to(torch.float)
        y = torch.from_numpy(y).to(torch.float)
        
        return x, y



class xycorr(nn.Module):
    def __init__(self, npts=None, nbatch=None):  # trying to see if machine can tell that y is the sum over x 
        super(xycorr, self).__init__()

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
       
    def get_xy_batch(self):
        nbatch = self.nbatch
        npts = self.npts
        
        assert((npts % 2) == 0)
        xsize = (nbatch,npts)
        half = int(npts/2)
        
        xrange = 20.0
        sloperange = 10.0
        
        noiseamp = np.random.uniform(low=1.0, high = 10*xrange, size=(nbatch,1))
        noise = np.random.normal(scale=noiseamp,size=xsize)
       
        slope = np.random.uniform(-sloperange,sloperange,size=(nbatch,1))
        x = np.random.uniform(low=-xrange, high=xrange, size=xsize)
        x[:,half:] = (x[:,0:half].reshape((nbatch,half))* slope).reshape((nbatch,half))
        x = x + noise
        
        r = np.zeros((nbatch))
        for i in range(nbatch):
            r[i] = np.corrcoef(x[i,0:half], y=x[i,half:])[0,1]

        x = torch.from_numpy(x).to(torch.float)
        r = torch.from_numpy(r).to(torch.float)
        return x,r

 

class inner_prod(nn.Module):
    def __init__(self, npts=None, nbatch=None):  # trying to see if machine can tell that y is the sum over x 
        super(inner_prod, self).__init__()

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
                
    def forward(self, x):
        dataflow = x
#        dataflow = self.bn1(x)
        dataflow = torch.relu(self.L1(dataflow))
        dataflow = torch.relu(self.L2(dataflow))
        dataflow = torch.relu(self.L3(dataflow))
        dataflow = torch.relu(self.L4(dataflow))
        dataflow = torch.relu(self.L5(dataflow))
        dataflow = torch.relu(self.L6(dataflow))
        dataflow = torch.relu(self.L7(dataflow))
        dataflow = torch.relu(self.Llast(dataflow))
        result = torch.squeeze(self.weight_vector(dataflow))
        return result
       
    def get_xy_batch(self):
        nbatch = self.nbatch
        npts = self.npts
        
        assert((npts % 2) == 0)
        xsize = (nbatch,npts)
        half = int(npts/2)
        
        xrange = 20.0
        sloperange = 10.0
        
        noiseamp = np.random.uniform(low=1.0, high = 10*xrange, size=(nbatch,1))
        noise = np.random.normal(scale=noiseamp,size=xsize)
       
        slope = np.random.uniform(-sloperange,sloperange,size=(nbatch,1))
        x = np.random.uniform(low=-xrange, high=xrange, size=xsize)
        x[:,half:] = (x[:,0:half].reshape((nbatch,half))* slope).reshape((nbatch,half))
        x = x + noise

        xdotx = np.sum(x[:,0:half] * x[:,half:], axis=1)
        x = torch.from_numpy(x).to(torch.float)
        xdotx = torch.from_numpy(xdotx).to(torch.float)
        return x,xdotx
 


class linregVAE(nn.Module):
    def __init__(self,dim=2,npts=50, nbatch=128):  # this sets up 5 linear layers
        super(linregVAE, self).__init__()

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

    def get_xy_batch(self):
        nbatch = self.nbatch
        npts = self.npts
        
        xsize = (nbatch,npts)
        
        xrange = 20.0
        sloperange = 10.0
        
        noiseamp = np.random.uniform(low=xrange/100.0, high = xrange/5.0, size=(nbatch,1))
        noise = np.random.normal(scale=noiseamp,size=xsize)
        slope = np.random.uniform(-sloperange,sloperange,size=(nbatch,1))
        offset = np.random.uniform(-sloperange,sloperange,size=(nbatch,1))
        
        x = np.random.uniform(low=-xrange, high=xrange, size=xsize)
        y = (x - offset) * slope
        x = x + noise
        
        x = torch.from_numpy(x).to(torch.float)
        y = torch.from_numpy(y).to(torch.float)
        return x,y
