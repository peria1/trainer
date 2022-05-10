# -*- coding: utf-8 -*-
"""
Created on Mon May  9 06:39:55 2022

This script demonstrates some sanity checks I needed to go through in learning to 
do vector calculus with Pytorch. 

imports etc. 

instantiate trainer via trainer_view

get references to:
        tvt (trainer itself)
        model
        optimizer
        example
        target
        pred
        
        
compute starting loss, copy to lsave

save current model state in psave and sdsave

backprop
step
get new pred

show change in loss.item

reset model

show return to initial value


@author: Bill
"""
# import trainer_view as TV
import trainer as trainer
import models, problems
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim

def set_learning_rate(optim, lr):
    for g in optim.param_groups:
        g['lr'] = lr

def reset_model(model, sd, g):
    model.load_state_dict(sd)
    for k, v in model.named_parameters():
        v.grad = copy.deepcopy(g[k])
        
def get_model_state(model):
    return copy.deepcopy(model.state_dict())

# def save_params(model):
#     psave = []
#     for p in model.parameters():
#         psave.append(copy.deepcopy(p))
#     return psave

def build_gradient_vector(model):
    g = None
    for p in model.parameters():
        pcpu = p.grad.flatten().detach().cpu()
        if g is None:
            g = pcpu
        else:
            g = torch.cat((g, pcpu))
    return g
 
def capture_gradients(model):
    return {k:copy.deepcopy(v.grad) for k,v in model.named_parameters()}
          

# https://discuss.pytorch.org/t/how-to-compute-magnitude-of-gradient-of-each-loss-function/138361
# grads1 = torch.autograd.grad(loss1(output), model.parameters(), retain_graph=True)
# grads2 = torch.autograd.grad(loss2(output), model.parameters())
# torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in list(model.parameters())]), 2.0)
# def grad_magnitude():

# In [28]: total_norm = 0.0
#     ...: for p in model.parameters():^M
#     ...:     param_norm = p.grad.detach().data.norm(2)^M
#     ...:     total_norm += param_norm.item() ** 2^M
#     ...:

# BUILD MINIMAL EXAMPLE AROUND THIS
# In [29]: ((lincheck[0]-lincheck[-1])/(max(steps)-min(steps)))/total_norm
# Out[29]: 0.9887093111184974

# In [30]: np.corrcoef(lincheck, steps)
# Out[30]:
# array([[ 1.        , -0.99981265],
#        [-0.99981265,  1.        ]])

def grad_angle(gdict0, gdict1):
    dot = 0.0
    norm0 = 0.0
    norm1 = 0.0
    for k, v in gdict0.items():
        g0 = gdict0[k]
        g1 = gdict1[k]
        dot += torch.sum(g0*g1)
        norm0 += torch.sum(g0**2)
        norm1 += torch.sum(g1**2)
    
    # print(dot.item(), torch.sqrt(norm0).item(), torch.sqrt(norm1).item())
    dot /= torch.sqrt(norm0*norm1)
    return torch.acos(torch.clip(dot, -1.0, 1.0))*180.0/np.pi

if __name__=="__main__":
    tvt = trainer.trainer(models.n_double_nout, problems.roots_of_poly)
    
    model = tvt.model
    # optimizer = tvt.optimizer
    optimizer_type = optim.SGD
    optimizer = optimizer_type(model.parameters(), lr=1e-5)
    
    example, target = tvt.xtest, tvt.ytest
    pred = model(example)
    loss = tvt.criterion(pred, target)
    
    optimizer.zero_grad()
    loss.backward()
    
    lsave = copy.copy(loss.item())
    sdsave = get_model_state(model)
    # psave = save_params(model)
    gsave = capture_gradients(model)
    optimizer.step()
    newpred = model(example)
    
    
    outs = {\
            'saved loss': lsave,
            'loss_item': loss.item(),
            'loss_w_newpred': tvt.criterion(newpred, target).item()
            }
    
    for k, v in outs.items():
        print(k, ':\t', v)
        
    # print(lsave, loss.item, tvt.criterion(pred, target))
    
    reset_model(model, sdsave, gsave)
    orig_pred = model(example)
    loss = tvt.criterion(orig_pred, target)
    print('loss after reset:', loss.item())
    
    print(tvt.criterion)
    
    lr0 = copy.copy(optimizer.param_groups[0]['lr'])
    npts = 100
    lincheck = np.zeros(npts)
    angle = np.zeros(npts)

    span = 0.005
    lr0, lr1 = -span, span
    lrs = np.linspace(lr0, lr1, npts)    
    # print('steps are', steps)
    for i in range(npts):
        reset_model(model, sdsave, gsave)

        lr = lrs[i]
        set_learning_rate(optimizer, lr)
        optimizer.step()

        loss = tvt.criterion(model(example), target)
        lincheck[i] = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        
        
        angle[i] = grad_angle(gsave, capture_gradients(model)) # between current grad and grad in gsave
    
    
    # I should be able to recover the learning rate from the magnitude of the 
    #   parameter displacement vector. Can I? 
    
    #
    # Find the gradient at the current location, i.e. the one used in parameter adjusting. 
    gnorm0sq = 0.0
    for k, v in gsave.items():
        gnorm0sq += torch.sum(v**2)
    gnorm0 = torch.sqrt(gnorm0sq)
    #
    # now find the magnitude of the changes in parameters, given different learning
    #   rates i.e. steps. 
    magsteps = []
    for lr in lrs:
        reset_model(model, sdsave, gsave)  # back to starting model. 
        a0 = get_model_state(model)
        set_learning_rate(optimizer, lr)
        optimizer.step() # this takes a step of size s * gnorm0
        a1 = get_model_state(model)
        
        normsq = 0.0
        for k, v in model.named_parameters():
            normsq += torch.sum((a1[k] - a0[k])**2)
        norm = torch.sqrt(normsq)
        magsteps.append(norm*np.sign(lr))      # these are the steps we *observe* in parameters. 
    
    s = np.zeros(npts)
    m = np.zeros(npts)
    for i in range(npts):
        s[i] = np.float(lrs[i]*gnorm0.item()) # these are the putatuve SGD steps
        m[i] = np.float(magsteps[i])
    
    ratio = m/s  # should be 1 if I know what's happening. 
    
        
    print('mean ratio', np.mean(np.abs(ratio)))
    print('std ratio', np.std(np.abs(ratio)))
    print('correlation', np.corrcoef(np.abs(s), np.abs(m))[0,1])

    numgrad = ((max(lincheck)-min(lincheck))/(max(magsteps)-min(magsteps))).item()
    print('grad vs step corr:', np.corrcoef(s, lincheck)[0,1])
    print('numerical gradient:', numgrad )
    print('norm of grad:', gnorm0)
    print('grad times', (numgrad/gnorm0).item(), '= numerical grad')
    
    plt.figure()
    plt.plot(s, lincheck,'-o')
    
    plt.figure()
    plt.plot(s, angle, '-o')
    
    plt.show()
   
    
    