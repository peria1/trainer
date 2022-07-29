#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:21:49 2022

@author: bill
"""

import torch
from torch import nn
from torch.autograd.functional import vhp
import copy
import numpy as np

torch.manual_seed(0)

# Following are utilities to make nn.Module "functional", in the sense of 
#    being from or compatible with the torch.autograd.functional library. 
#
# borrowed from the link I posted in comment
# 
def del_attr(obj, names_split): # why, why, why? But it definitely breaks without this. 
    if len(names_split) == 1:
#         print('BEFORE:')
#         print([n for n,p in obj.named_parameters()])
        delattr(obj, names_split[0])
#         print('AFTER:')
#         print([n for n,p in obj.named_parameters()])
    else:
#         print('recursing',len(names_split))
        del_attr(getattr(obj, names_split[0]), names_split[1:])

def set_attr(obj, names_split, val):
    if len(names_split) == 1:
#         print('BEFORE:')
#         print([n for n,p in obj.named_parameters()])
        setattr(obj, names_split[0], val)
#         print('AFTER:')
#         print([n for n,p in obj.named_parameters()])
    else:
#         print('recursing',len(names_split))
        set_attr(getattr(obj, names_split[0]), names_split[1:], val)

def make_functional(model):
    orig_params = tuple(model.parameters())
    orig_grad = capture_gradients(model)
    # Remove all the parameters in the model, because reasons. 
    names = []
    for name, p in list(model.named_parameters()):
        del_attr(model, name.split("."))
        names.append(name)
    return orig_params, orig_grad, names

def load_weights(model, names, params, as_params=False):
    for name, p in zip(names, params):
        if not as_params:
            set_attr(model, name.split("."), p)
        else:
            set_attr(model, name.split("."), torch.nn.Parameter(p))

def restore_model(model, names, params, grad): # (this one is mine)
    load_weights(model, names, params, as_params=True)
    for k, v in model.named_parameters():
        if grad[k+'.grad']:
            v.grad = grad[k].clone().detach()
        else:
            v.grad = None


def capture_gradients(model): # returns gradients in dict vector form (this one is also mine)
    g = {}
    for k, v in model.named_parameters():
        gnext = v.grad
        knext = k + '.grad'
        if gnext is not None:
            next_entry = {knext: gnext.clone().detach()}
        else:
            next_entry = {knext: None}
        g.update(next_entry)
    return g
  
def sigmoid(x):
    return 1 /(1+torch.exp(-x))

def sigprime(x):
    return torch.exp(-x)*sigmoid(x)**2

def invsigmoid(x):
    return -torch.log(1/x-1)
    
def max_vect_comp(x, maxabs=False):
    fmax = lambda x : torch.max(torch.abs(x)) if maxabs else torch.max(x)
    maxv = None
    try:
        for xi in x:
            imax = fmax(xi)
            if not maxv or maxv < imax:
                maxv = imax
    except TypeError:
        maxv = fmax(x)
        
    return maxv

def scale_vect(x,a):
    try:
        for xi in x:
            xi /= a
    except TypeError:
        x /= a
            
def norm_vect(x):
    scale_vect(x, torch.sqrt(dot_vect(x,x)))

def dot_vect(a,b):
    adotb = 0.0
    try:
        for ai,bi in zip(a,b):
            assert(ai.shape == bi.shape)
            adotb += torch.sum(ai*bi)            
    except TypeError:
        assert(a.shape==b.shape)
        adotb += torch.sum(a*b)
    except AssertionError:
        print('OOPS!', a.shape, b.shape)
        adotb = None
    return adotb

def angle_vect(a,b):
    cos_ab = dot_vect(a,b)/torch.sqrt(dot_vect(a,a)*dot_vect(b,b))
    
    return torch.acos(cos_ab)
    

# your simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        '''Forward pass'''
#         return self.layers(x)
        return torch.sigmoid(self.layers(x))


#  "loss" function
def objective(X):
    return torch.sum(X**2) # just square, no ground truth. 



# This is how we trick vhp into doing the Hessian with respect to params and not other inputs.
#
# Sadly it contains some global variables. 
def loss_wrt_params(*new_params):
    print('Entering loss_wrt_params...', flush=True)
    load_weights(mlp, names, new_params) # Weird! We removed the params before. 
    if len(tuple(mlp.named_parameters())) == 0:
        print('Model has no parameters!')
    for n,p in mlp.named_parameters():
        if p is None:
            print('whoops p is None!')
        else:
            print(n,p)
            
    out = mlp(xglobal)  # model output
    loss = objective(out)  # comparing model to ground truth, in practice. 
    
    loss.backward(retain_graph=True)
    return loss

if __name__ == "__main__":
    print('Here we go...just defs to start')
    # your model instantiation
    in_dim, out_dim = 30, 20
    mlp = SimpleMLP(in_dim, out_dim)
    
    mlpsave = copy.deepcopy(mlp)
    # print('back from mlp def...')
    # v_to_dot = tuple([torch.rand_like(p.clone().detach()) for p in mlp.parameters()])
    v_to_dot = tuple([torch.ones_like(p.clone().detach()) for p in mlp.parameters()])
    
    xglobal = torch.rand((in_dim,)) # need to eliminate this and other global refs. 
    
    orig_params, orig_grad, names = make_functional(mlp)
    
    mlpbroken = copy.deepcopy(mlp)
    
    
    params2pass = tuple(p.detach().requires_grad_() for p in orig_params)
    
    print('computing loss and vH...')
    # 
    loss_value, v_dot_hessian = \
        torch.autograd.functional.vhp(loss_wrt_params,
                                      params2pass,
                                      v_to_dot, strict=True)
    # print('back from loss and vH...')
    # restore_model(mlp, names, orig_params, orig_grad)
    
    # print('Computing loss to compare with previous...')
    # lossp = loss_wrt_params(*orig_params) # this calls backward on loss = objective(out)
    # grad = capture_gradients(mlp)
    # print('grad is:', grad)
    
    # print('loss_value:', loss_value)
    # print('lossp:', lossp)
    # print('vH(params):', v_dot_hessian)

    vnext = copy.deepcopy(v_dot_hessian)
    while True:
        scale_vect(vnext, max_vect_comp(vnext, maxabs=True))
        vprev = vnext
        _, vnext = \
            torch.autograd.functional.vhp(loss_wrt_params,
                                          params2pass,
                                          vnext, strict=True)
        if (dtht := angle_vect(vnext, vprev)) < 0.0001:
            break
        else:
            print(dtht)
        
        
#