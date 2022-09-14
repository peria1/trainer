# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:14:24 2022

@author: Bill
"""
import torch
from torch import nn

# your loss function
def objective(x):
    return torch.sum(0.25 * torch.sum(x)**4)

# Following are utilities to make nn.Module functional
# borrowed from the link I posted in comment
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

# the thing I want the hessian of....
def loss_wrt_params(*new_params):
    # this line replace your for loop
    load_weights(mlp, names, new_params)

    x = torch.ones((Arows,))
    out = mlp(x)
    loss = objective(out)
    return loss

# your simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


if __name__ == '__main__':
    # your model instantiation
    Arows = 2
    Acols = 2
    mlp = SimpleMLP(Arows, Acols)

    # your vector computation
    v = torch.ones((6,))
    v_tensors = []
    idx = 0
    #this code "reshapes" the v vector as needed
    for i, param in enumerate(mlp.parameters()):
        numel = param.numel()
        # sourceTensor.clone().detach().requires_grad_(True) # from torch warning
        # v_tensors.append(torch.reshape(torch.tensor(v[idx:idx+numel]), param.shape))
        v_tensors.append(torch.reshape(v[idx:idx+numel].clone().detach().requires_grad_(True), param.shape))
        idx += numel
    reshaped_v = tuple(v_tensors)

    #make model's parameters functional
    orig_params, names = make_functional(mlp)
    params2pass = tuple(p.detach().requires_grad_() for p in orig_params)

    #compute hvp
    soln = torch.autograd.functional.vhp(loss_wrt_params, params2pass, reshaped_v, strict=True)
    print(soln)
    
    
    