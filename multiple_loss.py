# -*- coding: utf-8 -*-
"""
Created on Thu May  5 08:47:27 2022
Trying to see if my idea for evaluating multple loss functions independently, 
and taking smaller steps when the gradient is large rather than larger ones, i.e. 
step size inversely proportional, not proportional, makes sense. 

What do gradients look like? What are the magnitudes? Do they really tend to be 
perpendicular? Is that concept defeated by the large number of dimensions and 
numbers tiny or huge? In other words, do grads merely look perpendicular, or are
they close enough to make independent stepping an improvement. Also, how often 
do we have the case that one loss function is rapidly varying while another is
flat, at a given point in parameter space? 


@author: Bill
"""
# import copy
import torch
# from torch import nn

def multi_grad(losses, optimizer, model):
    gradients = []
    
    for l in losses:
        optimizer.zero_grad()
        l.backward(retain_graph=True)
        gradients.append(build_gradient(model))
        
    return gradients
        
def build_gradient(model):
    g = None
    for p in model.parameters():
        pcpu = p.grad.flatten().detach().cpu()
        if g is None:
            g = pcpu
        else:
            g = torch.cat((g, pcpu))
        
    
    return g

def all_same(model):
    sx, sxx, N = 0.0, 0.0, 0
    for p in model.parameters():
        
        Np = 1
        for n in p.shape:
            Np *= n
        N += Np
        sx += torch.sum(p)
        sxx += torch.sum(p**2)
        
    print(sxx.item(), sx.item(), sx.item()**2)
    dev = torch.sqrt(sxx-sx**2)
        
    return dev

if __name__=='__main__':
    import trainer_view as TV
    import models
    import problems

    tv = TV.trainer_view(models.n_double_nout, problems.roots_of_poly)
    tvt = tv.trainer
    
    x, y = tvt.get_more_data()
    ypred = tvt.model(x)
    
    losses = [tvt.criterion(ypred, y), all_same(tvt.model)]
    
    optimizer, model = tvt.optimizer, tvt.model
    
    
    mg = multi_grad(losses, optimizer, model)
    
    angles = []
    for i, g1 in enumerate(mg):
        for g2 in mg[(i+1):]:
            cos_tht = torch.clip(torch.sum(g1*g2)/(torch.norm(g1)*torch.norm(g2)),\
                                 min=-1.0, max=1.0)
            angles.append(torch.acos(cos_tht))
    
    for a in angles:
        print(a.item()*180/3.14159,'degrees')
        
        
        
        
        