# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 07:31:52 2019

@author: Bill
"""
import torch
import numpy as np
import torch.utils.data
from torch import nn
"""
The superclass Problem takes care of things common to all of our "math 
    problem" data sets, like default dimensions and the requirement to 
    define a get_input_and_target method. 

The subclasses here, in their init methods, take care of things like special
    requirements on the dimensions (e.g. npts must be a power of two, etc.)
    These subclasses are named so as to answer the question: "How does the
    target depend on the input?" When I say x, I mean the input. x0 is the 
    left half of the input, and x1 is the right half (they are row vectors).


"""
class Problem():
    def __init__(self, npts=None, nbatch=None):
        if npts:
            self.npts = npts
        else:
            self.npts = 50
            
        if nbatch:
            self.nbatch = nbatch
        else:
            self.nbatch = 128

    def get_input_and_target(self):
        print('You must define your data generator.')
        return None

class plain_sum_of_x(Problem):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
 
    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        x = np.random.normal(size=(nbatch,npts))
    
        x = torch.from_numpy(x)
        y = torch.sum(x,1).to(torch.float32)
        
        x = x.to(torch.float32)
        
        return x,y

class quad_sum_of_x(Problem):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
      
    def get_input_and_target(self):
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


class slope_and_offset_x1_vs_x0(Problem): 
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        try:
            assert((self.npts != 0) and (self.npts & (self.npts-1) == 0))
        except AssertionError:
            print('Number of points in each example must be a power of two.')
            print('You requested npts =', self.npts)

    def get_input_and_target(self):
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
 

class x1_over_x0(Problem):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        npts = self.npts
        try:
            assert((npts != 0) and (npts & (npts-1) == 0))
        except AssertionError:
            print('Number of points in each example must be a power of two')
            print('You requested npts =', self.npts)
        
    def get_input_and_target(self):
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


class cumulative_sum_of_x(Problem): # moved to n_double_n
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        xsize = (nbatch,npts)

        xrange = 20.0
        x = np.random.uniform(low=-xrange, high=xrange, size=xsize)
        y = np.cumsum(x,axis=1)
        x = torch.from_numpy(x).to(torch.float)
        y = torch.from_numpy(y).to(torch.float)
        
        return x, y



class x0_corr_x1(Problem):    # moved to n_double_one_tanh
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

       
    def get_input_and_target(self):
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

 

class x0_dot_x1(Problem):  # moved to n_double_one
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        assert((self.npts % 2) == 0)
        
    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        
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
 

