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
import torch.utils.data
from torch import nn
from torchvision.models.vgg import VGG
import numpy as np

#from criteria import *
#from ..yolact.data import *
#from .. import *
#from ..yolact.yolact import *
import sys
if not 'yolact' in sys.path[1]:
    sys.path.insert(1, '../yolact/')
    
import yolact
from utils.augmentations import SSDAugmentation #, FastBaseTransform, BaseTransform
from yolact import Yolact
from train import  NetWithLoss, CustomDataParallel, MultiBoxLoss, prepare_data
import data as D  



class one_linear_layer(nn.Module):
    def __init__(self, problem):
        super().__init__()

        inp, target = problem.get_input_and_target()

        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        
        self.npts = npts
        self.nbatch = nbatch    
       
        self.L1 = nn.Linear(self.npts,target.size()[1])
       
    def forward(self,x):
        return self.L1(x)

class one_linear_layer_to_n(nn.Module):
    def __init__(self, problem):
        super().__init__()

        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        
        self.L1 = nn.Linear(self.npts,self.npts)
       
    def forward(self,x):
        return self.L1(x)

     
class bisect_to_power_of_two(nn.Module):
    def __init__(self, problem):   
        super().__init__()

        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        nout = target.size()[1]
        nchk = npts*nout
        
        try:
            assert((nchk != 0) and (nchk & (nchk-1) == 0))
        except AssertionError:
            print('Length of both input and output examples must be a power of two.')
        
        self.npts = npts
        self.nbatch = nbatch
        self.nout = nout
        
        n = npts*2
        self.layer_list = nn.ModuleList([nn.Linear(npts,n)])
        while n > nout:
            self.layer_list.append(nn.Linear(n,n//2))
            n//=2
        self.leaky = nn.LeakyReLU()

    def forward(self, xy):
        dataflow = xy
        for L in self.layer_list:
            dataflow = self.leaky(L(dataflow))
        
        return dataflow



class n_double_n_act(nn.Module): # moved to n_double_n
    def __init__(self, problem): 
        super().__init__()

        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
                
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
        self.activation1 = nn.LeakyReLU()
        self.activation2 = nn.Tanh()


    def forward(self, x):
        dataflow = self.activation1(self.L1(x))
        dataflow = self.activation1(self.L2(dataflow))
        dataflow = self.activation1(self.L3(dataflow))
        dataflow = self.activation1(self.L4(dataflow))
        dataflow = self.activation2(self.L5(dataflow))
        dataflow = self.activation2(self.L6(dataflow))
        dataflow = self.activation2(self.L7(dataflow))
        dataflow = self.Llast(dataflow)
        
        return dataflow



class n_double_n(nn.Module): # moved to n_double_n
    def __init__(self, problem): 
        super().__init__()

        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        
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
    def __init__(self, problem):  # trying to see if machine can tell that y is the sum over x 
        super().__init__()

        self.custom_loss = nn.L1Loss()
        
        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        
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



class n_double_nout(nn.Module):  # moved to n_double_one
    def __init__(self, problem):  # trying to see if machine can tell that y is the sum over x 
        super().__init__()


        self.custom_loss = nn.L1Loss()
        
        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        nout = target.size()[1]
        
        
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
        
        self.weight_vector = nn.Linear(npts, nout) # npts is the size of each 1D example
        
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
    def __init__(self, problem):  # trying to see if machine can tell that y is the sum over x 
        super().__init__()

        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        
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
        dataflow = self.weight_vector(dataflow)
        result = torch.tanh(dataflow)
        return result



class vectorVAE(nn.Module):   # moved to vectorVAE
    def __init__(self, problem, dim=2):  # trying to see if machine can tell that y is the sum over x 
        super().__init__()
        
        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        nout = target.size()[1]

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
class TrainerRNN(nn.Module):
    def __init__(self, problem, npts=None, nbatch=None, nout=None):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        nout = target.size()[1]
        
        self.input_size = npts
        self.seq_len = nbatch
        
        self.hidden_dim = 2*self.input_size  
        self.n_layers = 10
        self.output_size = nout
        
        self.rnn = nn.GRU(self.input_size, self.hidden_dim, self.n_layers)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

        self.forward_count = 0
        
    def forward(self, x):
        
        batch_size = 1

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x.unsqueeze(1), hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
#        out = out.view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        hidden = hidden.to(self.device)
        return hidden
         
#--------------------
#            vgg_model = VGGNet(pretrained = pretrained, requires_grad=True,GPU = GPU)

 
class VGGNet(VGG):
    def __init__(self, problem, in_channels=None, num_classes = None, nbatch = None, \
                 model='vgg16', requires_grad=True, \
                 show_params=False, GPU = False):
        from VGGdefs import ranges, cfg, make_layers


        if not num_classes:
            num_classes = 1000
        if not nbatch:
            nbatch = problem.nbatch

        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==4)
        nbatch, in_channels, nx, ny = inp.size()
        assert(np.prod(target.size())==nbatch)

        if not in_channels:
            in_channels=3
        
    
        super().__init__(make_layers(cfg[model],in_channels),\
              num_classes=num_classes)
        self.ranges = ranges[model]

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.device)
        if GPU:
            for name, param in self.named_parameters():
                param.cuda()
                
        self.custom_loss = self.vggloss
        
        self.criterion = torch.nn.CrossEntropyLoss()

        
    def vggloss(self,pred,target):
        
        return self.criterion(pred, target.to(torch.long))
    


    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x
        
        score = self.classifier(torch.flatten(self.avgpool(output['x5']),1))

        return torch.softmax(score,1)


class YOLAB(nn.Module):
    
    def __init__(self, problem):
        super().__init__()
        
        net = Yolact()
        net.train()
        net.init_weights(backbone_path='../yolact/weights/' + D.cfg.backbone.path)

        criterion = MultiBoxLoss(num_classes=D.cfg.num_classes,
                                 pos_threshold=D.cfg.positive_iou_threshold,
                                 neg_threshold=D.cfg.negative_iou_threshold,
                                 negpos_ratio=D.cfg.ohem_negpos_ratio)

        self.net = net
        self.criterion = criterion

    def forward(self, images): 
        self.predsT = self.net(images)
        return self.predsT
    
    def custom_loss(self, input, target):        
        targets, masks, num_crowds = target
        losses = self.criterion(self.net, self.predsT, targets[0], masks[0], num_crowds[0])
        self.loss = sum([losses[k] for k in losses])
        
        return self.loss




class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super().__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(1)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
    
