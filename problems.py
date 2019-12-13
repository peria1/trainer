# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 07:31:52 2019

@author: Bill
"""
import torch
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms


from PIL import Image

import skimage

from skimage import transform
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
    
class MNST_solver(Problem): # implenenting the MNST problem
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
        

    def get_input_and_target(self):
        nbatch = self.nbatch
#        npts = self.npts
#
        half = nbatch

        
        kwargs = {'num_workers': 1, 'pin_memory': True}
        
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.ToTensor()),
                               batch_size= nbatch*2, shuffle=True, **kwargs)

        for batch_idx, (data, which_digit) in enumerate(train_loader):
            break
        
        def Image_Tensor(image):
            rgb_image = pil2tensor(image)
            np_image = skimage.transform.resize(rgb_image.numpy(), (3,255,255))
            new_image = np_image.transpose(1,2,0)
            final = new_image[:,:,0]
            return final

        data_half = data[:half]
        data_full = data[half:]
        
        num_half = which_digit[:half]
        num_full = which_digit[half:]
        
        add_pic = 1 -Image_Tensor(Image.open('plus.jpg'))
        sub_pic = 1- Image_Tensor(Image.open('minus.jpg'))
        mul_pic = 1- Image_Tensor(Image.open('multiply.jpg'))
    
        add_list=[np.add,add_pic]
        sub_list=[np.subtract,sub_pic]
        min_list=[np.multiply,mul_pic] 
        
        operations_list=[add_list,sub_list,min_list]
            
        def index_opperation():
            key = np.random.randint(3)
            return key
        
        num1 = num_half.numpy()
        num2 = num_full.numpy()
        
        data_half = skimage.transform.resize(data_half.numpy(), (64,255,255))
        data_full = skimage.transform.resize(data_full.numpy(), (64,255,255))
        
        total_opp = np.zeros(64)
        total_pic = np.zeros((64,255*3,255))
        
        for i in range (64):     
            value = index_opperation()
            total_opp[i] = operations_list[value][0](num1[i],num2[i])
            total_pic[i] = np.concatenate((data_half[i],operations_list[value][1],data_full[i]))
        
        pts = total_pic.shape
        self.npts = pts[1] * pts[2]
        
        inp = self.move_to_torch(total_pic)
        target = self.move_to_torch(total_opp)
    
            
        return inp.view(-1,self.npts), target.reshape(-1,1)
    
    
class MNST_opp(Problem): #using handwritten digits and artimatic symbol to perform
    #opperation 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
        

    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        
        MNST_batch = 128
        half = MNST_batch//2
#        print(half)
        img_sz = 28
        
        kwargs = {'num_workers': 1, 'pin_memory': True}
        
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.ToTensor()),
                               batch_size=MNST_batch, shuffle=True, **kwargs)

        pil2tensor = transforms.ToTensor() #code to change image to a tensor
        tensor2pil = transforms.ToPILImage()
        
        for batch_idx, (data, which_digit) in enumerate(train_loader):
            break
        
        data_half = data[:half,:,:,:].squeeze() #splits the data in half
        data_full = data[half:,:,:,:].squeeze()
        
        num_half = which_digit[:half] #split to target numbers in half
        num_full = which_digit[half:]
        
        def Image_Tensor(image): #changes images to tensors
            rgb_image = pil2tensor(image)
            np_image = skimage.transform.resize(rgb_image.numpy(), (3,img_sz,img_sz))
            new_image = np_image.transpose(1,2,0)
            final = new_image[:,:,0]
            return final
        
        def perform_operation(num_half,num_full): #performs mathimatical operations to 
            #the numbers with the first half of data with the second half and combine 
            #the results into on tensor
            add_opp = torch.add(num_half,num_full)
            sub_opp = torch.sub(num_half,num_full)
            mul_opp = torch.mul(num_half,num_full)
        #    div_opp = torch.div(num_half,num_full)
            all_opp = torch.cat((add_opp,sub_opp,mul_opp))
            return all_opp
        
        def concate_number(data_half,data_full,add_pic,sub_pic,mul_pic): #concatenate hand
            #written digits with mathimatical symbols that is being operated on
            half_num = skimage.transform.resize(data_half.numpy(), (half,img_sz,img_sz))
            full_num = skimage.transform.resize(data_full.numpy(), (half,img_sz,img_sz))
    
            add_eq = np.zeros((half,img_sz*3,img_sz))
            sub_eq = np.zeros((half,img_sz*3,img_sz))
            mul_eq = np.zeros((half,img_sz*3,img_sz))
            for i in range(half):
                add_eq[i] = np.concatenate((half_num[i],add_pic,full_num[i])) 
                sub_eq[i] = np.concatenate((half_num[i],sub_pic,full_num[i])) 
                mul_eq[i] = np.concatenate((half_num[i],mul_pic,full_num[i])) 
            full_image = np.concatenate((add_eq,sub_eq,mul_eq))
            return full_image
    
    
        add_pic = 1 - Image_Tensor(Image.open('plus.jpg')) #creates additions symbol
        sub_pic = 1 - Image_Tensor(Image.open('minus.jpg')) #creates subtractions symbol
        mul_pic = 1 - Image_Tensor(Image.open('multiply.jpg')) #creates multiplication symbol
        
    
        input_results = concate_number(data_half,data_full,add_pic,sub_pic,mul_pic)
        target_results = perform_operation(num_half,num_full)
        
        self.npts = input_results.shape[1] * input_results.shape[2]
        self.nbatch = input_results.shape[0]
        
        input_pic = torch.from_numpy(input_results)
        input_pic = input_pic.view(-1,self.npts)
            
        return input_pic.to(torch.float32),target_results.reshape(-1,1).to(torch.float32)   
    
class MNST(Problem): # implenenting the MNST problem
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
        

    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
#        

        
        kwargs = {'num_workers': 1, 'pin_memory': True}
        
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.ToTensor()),
                               batch_size= nbatch, shuffle=True, **kwargs)

        for batch_idx, (data, which_digit) in enumerate(train_loader):
            xs = data.size()
            self.npts = xs[2] * xs [3]
            data = data.view(-1,self.npts)
            which_digit = which_digit.reshape(nbatch,1).to(torch.float32)
            break
            
            
        return data,which_digit
    
class MNST_sum(Problem): # target the input squared
                        #   But see below!
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
        

    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        
        half = nbatch
        MNST_batchsize = nbatch *2 #using half pairs
        
        sum_digits = torch.zeros(half)
        
        kwargs = {'num_workers': 1, 'pin_memory': True}
        
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.ToTensor()),
                               batch_size=MNST_batchsize, shuffle=True, **kwargs)
        def sum_of_digits(x0,x1):
            for i in range (half):
                sum_digits[i] = x0[i] + x1[i]
            return sum_digits
        
        for batch_idx, (data, which_digit) in enumerate(train_loader):
            datax0 = data[:half,:,:,:]
            datax1 = data[half:,:,:,:]
            
            which_digitx0=which_digit[:half]
            which_digitx1=which_digit[half:]
#            print(data.size())
#            print(datax0.size())
#            print(datax1.size())
            digit_total = sum_of_digits(which_digitx0,which_digitx1)
            x0_to_x1 = torch.cat((datax0.squeeze(),datax1.squeeze()),1)
            
            xs = x0_to_x1.size()
            self.npts = xs[1] * xs[2]
            
            x0_to_x1 = x0_to_x1.view(-1,self.npts)
            break
            
            
        return x0_to_x1,digit_total.reshape((nbatch,1))

class circumference(Problem): # takes input radius and finds circumference of circle
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        #
        r = np.random.normal(size=(nbatch,npts))
        c = 2 * np.pi * r
        
        return self.move_to_torch(r,c)
    
class tri_to_perimeter(Problem): # takes input of triangle vertices and finds perimeter
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
                
        return self.move_to_torch(x, dist)


class x_triple_x(Problem): # target is the cube of the input
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts
        x = np.random.normal(size=(nbatch,npts))
        y = np.power(x,3)
                
        return self.move_to_torch(x,y)

class roots_of_poly(Problem):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def get_input_and_target(self):
    
        x = np.random.normal(size=(self.nbatch,self.npts))
        y = np.zeros_like(x); y = np.concatenate((y,y),axis=1);
        
        xaug = np.concatenate((np.ones((self.nbatch,1)),x),axis=1)
        for i in range(self.nbatch):
            r = np.roots(xaug[i,:])
            y[i,0:self.npts] = np.real(r)
            y[i,self.npts:] = np.imag(r)

        return self.move_to_torch(x,y)

class x_double_x(Problem): # target the input squared
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_and_target(self):
        nbatch = self.nbatch
        npts = self.npts

        x = np.random.normal(size=(nbatch,npts))    
        alpha = np.random.uniform(size=(nbatch,1))
        y = alpha*x*x

        return self.move_to_torch(x,y)
   

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
        y = -x
    
        return self.move_to_torch(x,y)

class tri_to_area(Problem): # takes input of triange vertices and finds area
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_and_target(self):
        
        nbatch = self.nbatch
        npts = self.npts
   
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
        
        return self.move_to_torch(x,y)



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
        
