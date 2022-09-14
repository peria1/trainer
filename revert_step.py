# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:22:15 2022
THis is from Pytorch discussion group....a question about how to step backwards 
after stepping forwards along the gradient. I want to adapt this to taking my 
own steps, along Hessian eigenvector directions. 

I had to add the three import statements but after that it worked in iPython. 

https://discuss.pytorch.org/t/revert-optimizer-step/70692/5

@author: Bill
"""
import torch
from torchvision import models
import torch.nn as nn

torch.manual_seed(2809)

# Set model to eval to prevent batchnorm and dropout layers of changing the output
model = models.resnet50().eval()
x = torch.randn(2, 3, 224, 224)
target = torch.randint(0, 1000, (2,))

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Dummy update steps
out = model(x)
loss = criterion(out, target)
loss.backward()
print('Initial loss ', loss.item())
optimizer.step()
#fc0 = model.fc.weight.detach().clone()

# Get updated loss
out = model(x)
loss = criterion(out, target)
print('Updated loss ', loss.item())
#fc1 = model.fc.weight.detach().clone()

# Use negative lr
optimizer.param_groups[0]['lr'] = -1. * optimizer.param_groups[0]['lr']
optimizer.step()
out = model(x)
loss = criterion(out, target)
print('Reverted loss ', loss.item())
#fc2 = model.fc.weight.detach().clone()