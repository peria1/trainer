# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:02:58 2022

@author: Bill
"""
import torch
from torch.autograd.functional import vhp

def pow_reducer(x):
  return x.pow(3).sum()

inputs = torch.rand(2, 2)
v = torch.ones(2, 2)
vhp(pow_reducer, inputs, v)
vhp(pow_reducer, inputs, v, create_graph=True)

def pow_adder_reducer(x, y):
  return (2 * x.pow(2) + 3 * y.pow(2)).sum()

inputs = (torch.rand(2), torch.rand(2))
v = (torch.zeros(2), torch.ones(2))
vhp(pow_adder_reducer, inputs, v)