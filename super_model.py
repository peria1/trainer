# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:15:55 2022

@author: Bill
"""

import torch
from torch import nn

def store_input(self, x):
    self.xlast = x
    
class SuperModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_forward_pre_hook(store_input)
       
    
    def foo(self):
        for p in self.parameters():
            print(p)
            
    def max_eigen_H(self):

        lossfirst = loss_wrt_params(*orig_params)
        gradfirst = capture_gradients(mlp)
        mlp = copy.deepcopy(mlpsave)
        orig_params, orig_grad, names = make_functional(mlp)
        loss_value, v_dot_hessian = \
            torch.autograd.functional.vhp(loss_wrt_params,
                                          params2pass,
                                          v_to_dot, strict=True)
    
    def make_functional(self):
        model = self
        orig_params = tuple(model.parameters())
        orig_grad = capture_gradients(model)
        # Remove all the parameters in the model, because reasons. 
        names = []
        for name, p in list(model.named_parameters()):
            del_attr(model, name.split("."))
            names.append(name)
        return orig_params, orig_grad, names
        
    


def del_attr(obj, names_split): # why, why, why? But it definitely breaks without this. 
    if len(names_split) == 1:
        delattr(obj, names_split[0])
    else:
        del_attr(getattr(obj, names_split[0]), names_split[1:])

def set_attr(obj, names_split, val):
    if len(names_split) == 1:
        setattr(obj, names_split[0], val)
    else:
        set_attr(getattr(obj, names_split[0]), names_split[1:], val)


def load_weights(model, names, params, as_params=False):
    for name, p in zip(names, params):
        if not as_params:
            set_attr(model, name.split("."), p)
        else:
            set_attr(model, name.split("."), torch.nn.Parameter(p))

def restore_model(model, names, params, grad): # (this one is mine)
    load_weights(model, names, params, as_params=True)
    for k, v in model.named_parameters():
        kg = k + '.grad'
        if grad[kg] is not None:
            v.grad = grad[kg].clone().detach()
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

def scale_vect(x,a):  # IN-PLACE!! Returns None...
    try:
        for xi in x:
            xi /= a
    except TypeError:
        x /= a
            
def norm_vect(x):  # IN-PLACE!! Returns None...
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
    
def dict_to_tuple(d):
    t = []
    for k,v in d.items():
        t.append(v)
    return tuple(t)
        

    
if __name__ == "__main__":
    
    class SimpleMLP(SuperModel):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_dim, out_dim),
            )

        def forward(self, x):
            '''Forward pass'''
            return torch.sigmoid(self.layers(x))

    in_dim, out_dim = 3, 2
    mlp = SimpleMLP(in_dim, out_dim) 
    
    xglobal = torch.rand((in_dim,)) 
    out = mlp(xglobal)  # model output

    
    
