# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:26:31 2019

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
    res = yhat - y;
    dx = res - torch.mean(res,1,keepdim=True)
    dy = y - torch.mean(y,1,keepdim=True)
    sx = torch.sqrt(torch.mean(torch.pow(dx,2),1,keepdim=True))
    sy = torch.sqrt(torch.mean(torch.pow(dy,2),1,keepdim=True))
    return torch.mean(torch.abs(torch.mean(dx*dy,1,keepdim=True)/(sx*sy)))

def mse_plus_corr(yhat,y):
    mse = nn.MSELoss()
    return mse(yhat,y) + target_residual_correlation(yhat,y)

def L1_plus_corr(yhat,y):
    L1 = nn.L1Loss()
    return L1(yhat,y) + target_residual_correlation(yhat,y)
