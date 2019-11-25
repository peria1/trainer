# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 07:31:52 2019

@author: Bill
"""
import torch
import numpy as np
import torch.utils.data
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
    def __init__(self, npts=None, nbatch=None, nout=None, **kwargs):
        if npts:
            self.npts = npts
        else:
            self.npts = 50
       
        if nbatch:
            self.nbatch = nbatch
        else:
            self.nbatch = 128

    def move_to_torch(self, input, target):
        input = torch.from_numpy(input).to(torch.float32)
        target = torch.from_numpy(target).to(torch.float32)
        return input, target

    def get_input_and_target(self):
        print('You must define your data generator.')
        return None
    
class tri_to_perimeter(Problem): # takes input of triange vertices and finds area
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        #
        # The following really just limits the range of x, and then flips the 
        #    sign, to generate y. 
        #
        x = np.random.normal(size=(nbatch,npts))
        
        perimeter_np = np.zeros((128)) #create a empty numpy for new batches
        
        
        def distance(x1,y1,x2,y2): #this function finds the distance between each pair
            length = np.sqrt(((x1-x2)**2) + ((y1 -y2)**2))
            return length

        
        def get_perimeter(x): #this functions get the areas of all the examples
            for i in range (nbatch):
                
                x1,y1 = x[i][0],x[i][1]
                x2,y2 = x[i][2],x[i][3]
                x3,y3 = x[i][4],x[i][5]
                
                side1= distance(x1,y1,x2,y2)
                side2= distance(x1,y1,x3,y3)
                side3= distance(x3,y3,x2,y2)
                
                perimeter = side1 + side2 + side3
                
                perimeter_np[i] = perimeter
            return perimeter_np

        return self.move_to_torch(x, get_perimeter(x).reshape((nbatch,1)))

    
class x0_dist_x1(Problem): # target the input squared
                        #   But see below!
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        #
        # The following really just limits the range of x, and then flips the 
        #    sign, to generate y. 
        #
        x = np.random.normal(size=(nbatch,npts))
        half = npts//2
        dist = np.sqrt(np.sum(np.power((x[:,0:half] - x[:,half:]),2),\
                              axis=1,keepdims=True))
        
#        x = torch.from_numpy(x)
#        x = x.to(torch.float32)
#        y = dist
#        y = torch.from_numpy(y)
#        y = y.to(torch.float32).reshape((nbatch,1))
#        
        
        return self.move_to_torch(x, dist)


class x_triple_x(Problem): # target the input squared
                        #   But see below!
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        #
        # The following really just limits the range of x, and then flips the 
        #    sign, to generate y. 
        #
        x = np.random.normal(size=(nbatch,npts))
        
    
        y = np.power(x,3)
#        alpha = np.random.normal(size=(nbatch,npts))
#        y = #alpha * y
        
        x = torch.from_numpy(x)
        x = x.to(torch.float32)
        y = torch.from_numpy(y)
        y = y.to(torch.float32)
        
        return x,y




class roots_of_poly(Problem):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def get_input_and_target(self):
    
        x = np.random.normal(size=(self.nbatch,self.npts))
        y = np.zeros_like(x); y = np.concatenate((y,y),axis=1); y = y[:,:-2]
        for i in range(self.nbatch):
            r = np.roots(x[i,:])
            y[i,0:self.npts-1] = np.real(r)
            y[i,self.npts-1:] = np.imag(r)

        return self.move_to_torch(x,y)


class x_double_x(Problem): # target the input squared
                        #   But see below!
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        #
        # The following really just limits the range of x, and then flips the 
        #    sign, to generate y. 
        #
        x = np.random.normal(size=(nbatch,npts))    
        alpha = np.random.uniform(size=(nbatch,1))
        y = alpha*x*x
        
        y = torch.from_numpy(y)
        y = y.to(torch.float32)
        x = torch.from_numpy(x)
        x = x.to(torch.float32)
       
        return x,y
   

class opp_cos(Problem): # target is opposite of cosine of input. Why not? 
                        #   But see below!
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        #
        # The following really just limits the range of x, and then flips the 
        #    sign, to generate y. 
        #
        x = np.random.normal(size=(nbatch,npts))
        x = np.cos(x)
        x = torch.from_numpy(x)
        x = x.to(torch.float32)
    
        y = -x.to(torch.float32)
        
        return x,y

class tri_to_area(Problem): # takes input of triange vertices and finds area
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_and_target(self):
        
        nbatch = self.nbatch
        npts = self.npts
   
        #
        # The following really just limits the range of x, and then flips the 
        #    sign, to generate y. 
        #
        x = np.random.normal(size=(nbatch,npts))
        
        area_np = np.zeros((nbatch)) #create a empty numpy for new batches
        
        
        def distance(x1,y1,x2,y2): #this function finds the distance between each pair
            length = np.sqrt(((x1-x2)**2) + ((y1 -y2)**2))
            return length

        
        def get_areas(x): #this functions get the areas of all the examples
            for i in range (nbatch):
                
                x1,y1 = x[i][0],x[i][1]
                x2,y2 = x[i][2],x[i][3]
                x3,y3 = x[i][4],x[i][5]
                
                side1= distance(x1,y1,x2,y2)
                side2= distance(x1,y1,x3,y3)
                side3= distance(x3,y3,x2,y2)
                
                sp = (side1 +side2 +side3)/2
                
                area = np.sqrt((sp*(sp-side1)*(sp-side2)*(sp-side3)))
                area_np[i] = area
            return area_np

        return self.move_to_torch(x, get_areas(x).reshape((nbatch,1)))


class plain_sum_of_x(Problem):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
 
    def get_input_and_target(self):
        x = np.random.normal(size=(self.nbatch,self.npts))
        y = np.sum(x,axis=1,keepdims=True)
        return self.move_to_torch(x,y)

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
        y = torch.reshape(y,(nbatch,1))
    
        y = y + noise.reshape((nbatch,1))
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
        bias = np.random.uniform(low=-xrange/10, high=xrange/10,size=(nbatch,1))
        y = np.cumsum(x+bias,axis=1)
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
        r = torch.from_numpy(r).to(torch.float).reshape((nbatch,1))
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
        xdotx = torch.reshape(xdotx,(nbatch,1))
        return x,xdotx
 
class y_linear_with_x(Problem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        
        xsize = (nbatch,npts)
#        half = int(npts/2)
        
        xrange = 20.0
        sloperange = 10.0
        offsetrange = 10.0
        
        noiseamp = np.random.uniform(low=1.0, high = 10*xrange, size=(nbatch,1))
        noise = np.random.normal(scale=noiseamp,size=xsize)
       
        slope = np.random.uniform(-sloperange,sloperange,size=(nbatch,1))
        offset = np.random.uniform(-offsetrange,offsetrange,size=(nbatch,1))
        
        x = np.random.uniform(low=-xrange, high=xrange, size=xsize)
        y = (x-offset)* slope
        x = x + noise

        x = torch.from_numpy(x).to(torch.float)
        y = torch.from_numpy(y).to(torch.float)
        return x, y
        
class y_equals_x(Problem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        
        xsize = (nbatch,npts)
#        half = int(npts/2)
        
        xrange = 20.0        
        noiseamp = np.random.uniform(low=1.0, high = 10*xrange, size=(nbatch,1))
        noise = np.random.normal(scale=noiseamp,size=xsize)
               
        x = np.random.uniform(low=-xrange, high=xrange, size=xsize)
        y = x
        x = x + noise

        x = torch.from_numpy(x).to(torch.float)
        y = torch.from_numpy(y).to(torch.float)
        return x, y
    
class cos_of_x(Problem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        
        xrange = np.pi
        
        x = np.random.uniform(-xrange,xrange,size=(nbatch,npts))
        y = np.cos(x)
        
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)
        return x,y
    
class abs_fft(Problem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts

        xrange = 100
        x = np.random.uniform(-xrange,xrange,size=(nbatch,npts))
        x = torch.from_numpy(x)
        x = x.to(torch.float32)

        y = np.abs(np.fft.fft(x))
        y = torch.from_numpy(y)
        y = y.to(torch.float32)
        
        return x,y
        
