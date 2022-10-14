# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:28:42 2019

@author: Bill
"""
#import matplotlib.pyplot as plt
#import argparse
#import itertools
import torch
import numpy as np
#import torch.utils.data
from torch import nn, optim
from models import *
from problems import *
from trainer_utils import kscirc, uichoosefile, date_for_filename, get_slash
import os
#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class trainer():
    def __init__(self, trainee_class, problem_class, \
                 max_loss=None, min_pval = None, reload=False, \
                 viewer=None, **kwargs):
        #
        # Given the name of a trainee class, trainer.__init__ will instantiate that 
        #  class and get set up to train that instance. 
        #
        
        self.use_GAlr = True
        
        if not max_loss:
            max_loss = 0.0
        self.max_loss = max_loss

        if not min_pval:
            min_pval = 0.5 
        self.min_pval = min_pval
        
        self.set_device()

        self.problem = problem_class(**kwargs)
                   
        self.data_generator = self.problem.get_input_and_target
                            
        self.xtest, self.ytest = self.data_generator()
#        print('Top of trainer, type of xtest is',type(self.xtest))
#        print('type of ytest is',type(self.ytest))
#        if type(self.ytest) is dict:
#            print(self.ytest.keys())
        
        self.xtest = self.xtest.to(self.device)
        try:
            self.ytest = self.ytest.to(self.device)
        except AttributeError:
            pass               
                
#        if len(self.xtest.size()) == 2:
#            self.npts = None
#            if 'npts' in kwargs:
#                self.npts = kwargs['npts']
#                kwargs.pop('npts')
#            else:
#                self.npts = self.xtest.size()[1]
#        else:
#            self.npts = np.prod(self.xtest.size()[1:])
#            
#        if 'nout' in kwargs:
#            self.nout = kwargs['nout']
#            kwargs.pop('nout')
#        else:
#            if len(self.ytest.size())==2:
#                self.nout = self.ytest.size()[1]
#            else:
#                self.nout = None
        
        trainee = trainee_class(self.problem).to(self.device)
        self.model = trainee # a trainee needs an optimzer and a criterion, 
                                           #   as well as a way to generate data.

        self.model.to(self.device)
        if viewer:
            self.viewer = viewer
        else:
            self.viewer = None
            
        self.train_loss_history = []
        self.test_loss_history = []
                       
        if reload:  # does not work on Windows, can't get Tk to work
            self.model.load_state_dict(torch.load(uichoosefile()))
                     

        self.res_bad = False
        try:
            assert(self.model(self.xtest).size()==self.ytest.size())
        except AssertionError:
            print('Model prediction and target dimensions differ.')
            print('This may indicate a serious problem, and for sure you cannot look at residuals.')
            print('Prediction: ', self.model(self.xtest).size())
            print('Target: ', self.ytest.size())
            self.res_bad = True
        except AttributeError:
            print('yo, it''s a dictionary!')
            
        if 'custom_loss' in dir(self.model):
            print('Using',self.get_model_name(),'custom loss ...')
            self.criterion = self.model.custom_loss
        else:
            self.criterion = nn.MSELoss()

        self.model.set_criterion(self.criterion)
        
        self.optimizer_type = optim.SGD
        self.optimizer = self.optimizer_type(self.model.parameters(), lr=1e-5)
        self.eps = 0.001 # percent reduction in loss function at each iteration. 
        self.pause = False
        
        self.iter_per_batch = 10  # not sure what to do with this. Training loop will
                                  #  do this many iterations on each batch, before 
                                  #  running and reporting a test, and grabbing new data. 
        
        self.xp = self.xtest.cpu().detach().numpy()  # these are useful for doing 
        try:
            self.yp = self.ytest.cpu().detach().numpy()  #  testing in numpy rather than torch. 
        except AttributeError:
            self.yp = self.ytest
            print('ytest type is',type(self.ytest),'...cannot move to numpy')

        self.zap_history()        

    def train_step(self, input, target, linearize=False):
        self.model.train()  # make sure model is in training mode
        
        self.optimizer.zero_grad()  # Make the gradients zero to start the step.
        pred = self.model(input)      #  Find the current predictions, yhat. 
        loss = self.criterion(pred, target)  
        loss.backward()      # Do back propagation! Thank you Pytorch!
        
        if self.use_GAlr:
        # now set the new gradient-adaptive learning rate
            GAlr = self.eps*loss.item()/normsq_grad(self.model)
            for g in self.optimizer.param_groups:
                g['lr'] = GAlr
                
        # if GAlr < 0.02:
        #     print('GAlr:', GAlr)
        # GAlr = min(GAlr, 0.02)
        
        # print()
        
        self.optimizer.step()     # take one step down the gradient. 
        
        return loss.item()
    
    def test(self,input,target):
#        if self.get_model_name() != 'YOLAB':
#            self.model.eval()  # make sure model is in testing mode
        pred = self.model(input)      #  Find the current predictions, yhat. 
        loss = self.criterion(pred, target)  # Check the MSE between y and yhat. 
        return loss.item()
    
    def train(self):        
        ppb = self.iter_per_batch
        model = self.model
        xtest, ytest = self.xtest, self.ytest
        _, yp = self.xp, self.yp
        done = False
        tests_since_update = 0
        display_update_delay = 1 #  only update after this many tests
        while not (done or self.pause):
            x,y = self.get_more_data()
            for i in range(ppb):
                steploss = self.train_step(x,y)
                self.train_loss_history.append(steploss)
            
            if not self.res_bad:
                try:
                    res = (model(xtest)-ytest).cpu().detach().numpy()
                    pos = res > 0
                    neg = res < 0
                
                    if np.sum(pos) > 0 and np.sum(neg) > 0:
                        h, p, ks = kscirc(yp[pos], yp[neg])
                    else:
                        p = 0.
                    
                    self.p_history.append(p)
                except (RuntimeError, TypeError):
                    print('Problem with residuals. Skipping on future iterations')
                    self.res_bad = True

            self.test_loss_history.append(self.test(xtest,ytest))

            up_since_last = self.test_loss_history[-1] > self.train_loss_history[-1]
            larger_than_recent_average = \
            self.test_loss_history[-1] > np.mean(self.test_loss_history[-ppb:-1])
            if not self.res_bad:
                kscirc_plausible = p > self.min_pval
            else:
                kscirc_plausible = True
                
                
            loss_small_enough = self.test_loss_history[-1] < self.max_loss 

            done =  up_since_last and \
             larger_than_recent_average and \
             kscirc_plausible and \
             loss_small_enough  # ignored if user did not set max_loss keyword
            
            if not self.res_bad:
                loss_str = f'Test loss: {self.test_loss_history[-1]:6.3e}    p:  {p:5.2e}'
            else:
                loss_str = f'Test loss: {self.test_loss_history[-1]:6.3e} '
            
            loss_str = self.get_model_name() + ' ' + loss_str
            
            if done:
                loss_str = loss_str + '   DONE!!'
            elif self.pause:
                loss_str = loss_str + '   paused.'
            
            if self.viewer:
                self.viewer.ax.set_title(str(loss_str))
                try:
                    self.viewer.fig.canvas.draw()
                except AttributeError:
                    print('viewer.fig type is',type(self.viewer.fig))
                    print('Why is this a problem?')
                self.viewer.fig.canvas.flush_events()
                tests_since_update = (tests_since_update + 1) % display_update_delay 
                if tests_since_update == 0:
                    self.viewer.update_displays()
            else:
                print(loss_str)
                           
        if done:
            self.viewer.set_update_flag(flag=True)
            self.viewer.update_displays()
            self.viewer.arm_start_button()
            
            self.save_model()
            
#            torch.save(self.model.state_dict(), \
#                       'saved_trained_states' + get_slash() + \
#                       self.model.__class__.__name__ + \
#                       date_for_filename())

    def auto_set_lr(self):
        """
        I want to explore learning rate space to find the current best. 
        """
        paused = self.pause
        self.pause = True
        name = self.model.__class__.__name__ + 'temp_please_delete'
        self.save_model(savename=name)
        opt_type = self.optimizer_type
        opt = self.optimizer

        lr_try = list(10.0**np.linspace(-9,-1,50))
        losses = np.zeros_like(lr_try)
        for i, lr in enumerate(lr_try):
            self.optimizer = torch.optim.SGD(self.model.parameters(),lr=lr)
            self.optimizer.step()
            loss = self.criterion(self.model(self.xtest), self.ytest)
            if i > 1 and loss.item() > losses[0]*1.2:
                losses = losses[0:i]
                lr_try = np.cumsum(lr_try[0:i])
                break
            else:
                losses[i] = loss.item()

        self.optimizer_type = opt_type
        self.optimizer = opt
        
        self.model.load_state_dict(torch.load(name))
        os.remove(name)
        self.pause = paused

        return lr_try, losses
    
    def save_model(self, savename=None):
        state = self.model.state_dict()
        if not savename:
            savename = 'saved_trained_states' + get_slash() + \
               self.model.__class__.__name__ + date_for_filename()
        torch.save(state, savename)
   
    def set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_more_data(self):
        x,y = self.data_generator()
        try:
            x, y = x.to(self.device), y.to(self.device)
        except AttributeError:
            x, y = x.to(self.device), y
        # Can I put x and y into the model for use in Hessian stuff here? 
        self.model.x_now = x
        self.model.y_now = y
        return x, y
    
    def zap_history(self):
        state = self.pause
        self.pause = True
        self.train_loss_history = []
        self.test_loss_history = []
        self.p_history = []
        self.pause = state

    def get_model_name(self):
        func_rep = str(self.model)
        return func_rep[0:func_rep.find('(')]

    def get_problem_name(self):
        func_rep = str(self.problem)
        return func_rep[func_rep.find('.')+1:func_rep.find('object')-1]
     
    def get_named_weight_list(self):
        return [(n,p) for (n,p) in self.model.named_parameters() if '.weight' in n]
    
    def reset_model(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight.data)
            if isinstance(m,nn.Linear):
                m.reset_parameters()
        state = self.pause
        self.pause = True
        self.model.apply(weights_init)
        self.pause = state

def normsq_grad(model):
    absL2 = 0
    for p in model.parameters():
        absL2 += torch.sum(p.grad**2)
        
    return absL2

#    def consume_keyword(keywords, keyword, member, value=None):
#        if keyword in keywords:
#            member = keywords[keyword]
#            keywords.pop(keyword)
#        else:
#            member = value
#        
