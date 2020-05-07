# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:26:31 2019
stuck in merge hell
@author: Bill
"""
import torch
from torch import nn

def KLD_pos_neg():
    pass

def mutual_info_target_residuals():
    pass


def decorrelate_target_residuals(yhat,y):
    pass
    #
# from old matlab code....
#cxy = cov(x(:),y(:)); cxy = cxy(2);
#%tht = 0.5 * atan2(2*cxy, var(x(:)) - var(y(:)));
#z = 2*cxy/(var(x(:)) - var(y(:)));
#
#tht = 0.5 * (1i/2)*(log(1-1i*z) - log(1+1i*z));
    
def target_residual_correlation(yhat,y):
#    import numpy as np
    
    res = yhat - y;
    x0 = torch.mean(res,0,keepdim=True)
    y0 = torch.mean(y,  0,keepdim=True)
    dx = res - x0
    dy = y - y0
    
#    print('np corr',np.corrcoef(dx.cpu().detach().numpy(),\
#                      dy.cpu().detach().numpy()),0)
#    
#    print(dx.size(), dy.size())
    sx = torch.sqrt(torch.mean(torch.pow(dx,2),0,keepdim=True))
    sy = torch.sqrt(torch.mean(torch.pow(dy,2),0,keepdim=True))
    
    corrs = torch.mean(dx*dy,0,keepdim=True)/(sx*sy)
#    print('individual corrs',corrs)

    return torch.abs(corrs)

def target_residual_slope(yhat,y):
    rho = target_residual_correlation(yhat,y)
    dres = torch.std(yhat-y,0,keepdim=True)
    dy = torch.std(yhat-y,0,keepdim=True)
    
    return rho*dres/dy


def target_residual_correlation2(yhat,y):
#    import numpy as np
    
    res = yhat - y;
    x0 = torch.mean(res,1,keepdim=True)
    y0 = torch.mean(y,  1,keepdim=True)
    dx = res - x0
    dy = y - y0
    
#    print('np corr',np.corrcoef(dx.cpu().detach().numpy(),\
#                      dy.cpu().detach().numpy()),0)
#    
#    print(dx.size(), dy.size())
    sx = torch.sqrt(torch.mean(torch.pow(dx,2),1,keepdim=True))
    sy = torch.sqrt(torch.mean(torch.pow(dy,2),1,keepdim=True))
    
    corrs = torch.mean(dx*dy,1,keepdim=True)/(sx*sy)
#    print('individual corrs',corrs)

    return torch.mean(torch.abs(corrs))

def target_residual_correlation3(yhat,y):
#    import numpy as np
    
    res = yhat - y;
    x0 = torch.mean(res,1,keepdim=True)
    y0 = torch.mean(y,  1,keepdim=True)
    dx = res - x0
    dy = y - y0
    
#    print('np corr',np.corrcoef(dx.cpu().detach().numpy(),\
#                      dy.cpu().detach().numpy()),0)
#    
#    print(dx.size(), dy.size())
    sx = torch.sqrt(torch.mean(torch.pow(dx,2),1,keepdim=True))
    sy = torch.sqrt(torch.mean(torch.pow(dy,2),1,keepdim=True))
    
    corrs = torch.mean(dx*dy,1,keepdim=True)/(sx*sy)
#    print('individual corrs',corrs)

    return torch.max(torch.abs(corrs))

def mse_plus_corr(yhat,y):
    mse = nn.MSELoss()
    return mse(yhat,y) + torch.mean(target_residual_correlation(yhat,y))

def mse_plus_slope(yhat,y):
    mse = nn.MSELoss()
    return mse(yhat,y) + torch.mean(target_residual_slope(yhat,y))
    

def L1_plus_corr(yhat,y,alpha=None):
    if not alpha:
        alpha = 1.0

    L1 = nn.L1Loss()
    return L1(yhat,y) + alpha*torch.mean(target_residual_correlation(yhat,y))

def diff_and_product(yhat,y):
    L1 = nn.L1Loss()
    return (1+torch.mean(target_residual_correlation(yhat,y)))*L1(yhat,y)
<<<<<<< HEAD


def diff_and_slope(yhat,y):
    L1 = nn.L1Loss()
    return (1+torch.sum(target_residual_slope(yhat,y)))*L1(yhat,y)


def max_abs(yhat,y):
    return torch.max(torch.abs(yhat-y))

def diff_and_max(yhat,y):
    L1 = nn.L1Loss()
    return L1(yhat,y) + max_abs(yhat, y)
=======
>>>>>>> Merged conflict involving diff_and_product*.

def diff_and_product2(yhat,y):
    L1 = nn.L1Loss()
    return (1+target_residual_correlation2(yhat,y))*L1(yhat,y)

def diff_and_product3(yhat,y):
    L1 = nn.L1Loss()
    return (1+target_residual_correlation3(yhat,y))*L1(yhat,y)

def diff_and_slope(yhat,y):
    L1 = nn.L1Loss()
    return (1+torch.sum(target_residual_slope(yhat,y)))*L1(yhat,y)

def max_abs(yhat,y):
    return torch.max(torch.abs(yhat-y))

def diff_and_max(yhat,y):
    L1 = nn.L1Loss()
    return L1(yhat,y) + max_abs(yhat, y)

def diff_and_slope(yhat,y):
    L1 = nn.L1Loss()
    return (1+torch.sum(target_residual_slope(yhat,y)))*L1(yhat,y)


def max_abs(yhat,y):
    return torch.max(torch.abs(yhat-y))

def diff_and_max(yhat,y):
    L1 = nn.L1Loss()
    return L1(yhat,y) + max_abs(yhat, y)

def diff_max_slope(yhat,y):
    return diff_and_max(yhat,y) + torch.sum(target_residual_slope(yhat,y))

