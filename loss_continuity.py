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

def set_learning_rate(optim, lr):
    for g in optim.param_groups:
        g['lr'] = lr

def reset_model(model, sd, g):
    model.load_state_dict(sd)
    for k, v in model.named_parameters():
        v.grad = copy.deepcopy(g[k])
        
def save_model_state(model):
    return copy.deepcopy(model.state_dict())

def save_params(model):
    psave = []
    for p in model.parameters():
        psave.append(copy.deepcopy(p))
    return psave

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
          
# def grad_magnitude():

def grad_angle(gsave, model):
    dot = 0.0
    norm0 = 0.0
    norm1 = 0.0
    for k, v in model.named_parameters():
        g0 = gsave[k]
        g1 = v.grad
        dot += torch.sum(g0*g1)
        norm0 += torch.sum(g0**2)
        norm1 += torch.sum(g1**2)
    
    print(dot.item(), torch.sqrt(norm0).item(), torch.sqrt(norm1).item())
    dot /= torch.sqrt(norm0*norm1)
    return torch.acos(torch.clip(dot, -1.0, 1.0))*180.0/np.pi

if __name__=="__main__":
    # tv = TV.trainer_view()
    
    tvt = trainer.trainer(models.n_double_nout, problems.abs_fft)
    
    model = tvt.model
    optimizer = tvt.optimizer
    
    example, target = tvt.xtest, tvt.ytest
    pred = model(example)
    loss = tvt.criterion(pred, target)
    
    optimizer.zero_grad()
    loss.backward()
    
    lsave = copy.copy(loss.item())
    sdsave = save_model_state(model)
    psave = save_params(model)
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
    npts = 101
    lincheck = np.zeros(npts)
    angle = np.zeros(npts)

    span = 0.005
    step0, step1 = -span, span
    steps = np.linspace(step0, step1, npts)    
    print('steps are', steps)
    for i in range(npts):
        reset_model(model, sdsave, gsave)
        
        step = steps[i]
        set_learning_rate(optimizer, step)
        optimizer.step()
        
        loss = tvt.criterion(model(example), target)
        lincheck[i] = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        angle[i] = grad_angle(gsave, model) # between current grad and grad in gsave
    
    plt.figure()
    plt.plot(steps, lincheck,'-o')
    
    plt.figure()
    plt.plot(steps, angle, '-o')
    
    plt.show()
    
    
