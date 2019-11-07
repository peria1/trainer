# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:28:42 2019

@author: Bill
"""



import matplotlib.pyplot as plt
#import argparse
#import itertools
import torch
import numpy as np
#import torch.utils.data
from torch import nn, optim
#from torch.nn import functional as F
#from torchvision import datasets, transforms
#from torchvision.utils import save_image
#import time
from models import *
#import trainer_view
#import sys
#sys.path.append('C:\\Users\\peria\\Desktop\\work\\Brent Lab\\Boucheron CNNs\\DLDBproject\\')
from trainer_utils import kscirc, uichoosefile, date_for_filename, get_slash

class trainer():
    def __init__(self, trainee_class,max_loss=None,reload=False, \
                 viewer=None, **kwargs):
        #
        # Given the name of a trainee class, trainer.__init__ will instantiate that 
        #  class and get set up to train that instance. 
        #
        
        if max_loss is None:
            max_loss = 1e30 
        self.max_loss = max_loss
        
        self.set_device()

        trainee = trainee_class(**kwargs).to(self.device)
        self.model = trainee # a trainee needs an optimzer and a criterion, 
                                           #   as well as a way to generate data.
                                           
        self.model.to(self.device)
        if viewer:
            self.viewer = viewer
        else:
            self.viewer = None
            
        self.train_loss_history = []
        self.test_loss_history = []
                       
#        print('trainer device is ',self.device)
        if reload:  # does not work on Windows, can't get Tk to work
            self.model.load_state_dict(torch.load(uichoosefile()))
                                           
        self.xtest, self.ytest = self.model.get_xy_batch()
        self.xtest = self.xtest.to(self.device)
        self.ytest = self.ytest.to(self.device)
        

        try:
            assert(self.model(self.xtest).size()==self.ytest.size())
        except AssertionError:
            print('Model predictons need to have the same dimensions as the targets.')
            return
        
        if 'custom_loss' in dir(self.model):
            self.criterion = self.model.custom_loss
        else:
            self.criterion = nn.MSELoss()

        self.optimizer_type = optim.Adam
        self.optimizer = self.optimizer_type(self.model.parameters(), lr=1e-5)
        self.pause = False
        
        self.iter_per_batch = 10  # not sure what to do with this. Training loop will
                                  #  do this many iterations on each batch, before 
                                  #  running and reporting a test, and grabbing new data. 
        
        self.xp = self.xtest.cpu().detach().numpy()  # these are useful for doing 
        self.yp = self.ytest.cpu().detach().numpy()  #  testing in numpy rather than torch. 

        self.zap_history()        
        
    def train_step(self, input, target):
        self.model.train()  # make sure model is in training mode
        
        self.optimizer.zero_grad()  # Make the gradients zero to start the step.
        pred = self.model(input)      #  Find the current predictions, yhat. 
#        print(type(pred), pred.size(), type(target),target.size())
        loss = self.criterion(pred, target)  # Check the MSE between y and yhat. 
        loss.backward()      # Do back propagation! Thank you Pytorch!
        self.optimizer.step()     # take one step down the gradient. 
        
        return loss.item()
    
    def test(self,input,target):
        self.model.eval()  # make sure model is in testing mode
        pred = self.model(input)      #  Find the current predictions, yhat. 
        loss = self.criterion(pred, target)  # Check the MSE between y and yhat. 
        return loss.item()
    
    def train(self):        
        ppb = self.iter_per_batch
        model = self.model
        xtest, ytest = self.xtest, self.ytest
        _, yp = self.xp, self.yp
        done = False
        while not (done or self.pause):
            x,y = self.get_more_data()
            for i in range(ppb):
                steploss = self.train_step(x,y)
                self.train_loss_history.append(steploss)
            
            res = (model(xtest)-ytest).cpu().detach().numpy()
            pos = res > 0
            neg = res < 0
        
            if np.sum(pos) > 0 and np.sum(neg) > 0:
                h, p, ks = kscirc(yp[pos], yp[neg])
            else:
                p = 0.
            
            self.p_history.append(p)
            
            self.test_loss_history.append(self.test(xtest,ytest))

            up_since_last = self.test_loss_history[-1] > self.train_loss_history[-1]
            larger_than_recent_average = \
            self.test_loss_history[-1] > np.mean(self.test_loss_history[-ppb:-1])
            kscirc_plausible = p > 0.5
            loss_small_enough = self.test_loss_history[-1] < self.max_loss 

            done =  up_since_last and \
             larger_than_recent_average and \
             kscirc_plausible and \
             loss_small_enough  # ignored if user did not set max_loss keyword
            
            loss_str = f'Test loss: {self.test_loss_history[-1]:6.3e}    p:  {p:5.2e}'
            if done:
                loss_str = loss_str + '   DONE!!'
            elif self.pause:
                loss_str = loss_str + '   paused.'
            
            if not self.viewer:
                print(loss_str)
            else:
                self.viewer.ax.set_title(str(loss_str))
                self.viewer.fig.canvas.draw()
                self.viewer.fig.canvas.flush_events()

                self.viewer.update_displays()
 
                           
        if done:
            print('should update!')
            self.viewer.set_update_flag()
            self.viewer.update_displays()
            torch.save(self.model.state_dict(), \
                       'saved_trained_states' + get_slash() + \
                       self.model.__class__.__name__ + \
                       date_for_filename())

    def set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_more_data(self):
        x,y = self.model.get_xy_batch()
        return x.to(self.device), y.to(self.device)
    
    def zap_history(self):
        self.train_loss_history = []
        self.test_loss_history = []
        self.p_history = []

        
        

        
    
#    def set_max_loss(self, max_loss)
        
     
