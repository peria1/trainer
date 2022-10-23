# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:15:55 2022

Trying to encapsulate the stuff in vhp_learn.py into a superclass of a
    Pytorch model. I am trying to use this class as a "global" context, 
    so I can get rid of all the global references that the toy code in 
    vhp_learn uses. 

You can subclass SuperModel instead of torch.nn.Module, and call its 
__init__ when making a model. This does a few things:


1) Calls nn.Module __init__()

2) Makes a default vector attribute, to be loaded later once parameters are 
defined. This is the "vector to dot into the Hessian matrix".

3) Register a hook to be called when forward is called, i.e. during inference. 
This hook takes the input data and stores it in a SuperModel attribute called
xlast, and also sets the default vector attribute to be a tuple of tensors
containing 1's, each with the same shape as the model parameter to which
it corresponds. 

I do not understand how this works in detail. But basically, to use the 
autograd.functional methods, we need to be looking at derivatives with 
respect to inputs, not model parameters. So we need to wrap the function 
that we are training in a function that takes only the model parameters as its 
inputs. 

Furthermore, whenever we want to calculate something with autograd.functional, 
we need to call SuperModel.make_functional(). This function deletes all the
parameters from the model, and stores them in SuperModel attributes. No, I 
am not joking. Once we have done this, we can go ahead with the calculation. 

This is not yet general; it can only calculate the vector-Hessian product. 
But that's all I want so far!

@author: Bill
"""
import copy
import numpy as np
import torch
from torch import nn

class SuperModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_forward_pre_hook(store_inputs)
        
        self.max_iterations = 10000
        self.allowed_angular_error = 0.001 # radians
        self.allowed_cos_error = 1-np.cos(self.allowed_angular_error)
        print(self.allowed_cos_error)
        self.no_warning = False

        self.default_v = None
        self.hessian = None
        self.x_now = None
        self.y_now = None

    def param_id_list(self):
        plist = []
        for p in self.parameters():
            plist.append(id(p))
        return plist
    
    def is_functional(self):
        return len(tuple(self.parameters())) == 0

    def make_functional(self):
        orig_params = tuple(self.parameters())
        
        # print('PARAMETER REFERENCES ARE KEPT?')
        # This pumps out True a bajillion times...
        # for op, sp in zip(orig_params, self.parameters()):
        #     print(op==sp)
        #
        # So, orig_params is a tuple of references to the original model params
            
        # orig_grad = self.capture_gradients()
        # Remove all the parameters from the model, because reasons. 
        names = []
        for name, p in list(self.named_parameters()):
            del_attr(self, name.split("."))
            names.append(name)
            
        self.names = names
        self.orig_params = orig_params
        # self.curr_grad = orig_grad
        
    def load_weights(self, names, params, as_params=False):
        for name, p in zip(names, params):
            if not as_params:
                set_attr(self, name.split("."), p)
            else:
                set_attr(self, name.split("."), torch.nn.Parameter(p))
    
    # def restore_model(self, orig=False): # (this one is mine)
    #     if not self.is_functional():
    #         return  # no need to restore....
        
    #     self.load_weights(self.names, self.orig_params, as_params=True)
        
    #     if orig:
    #         grad = self.orig_grad
    #     else:
    #         grad = self.curr_grad
            
    #     for k, v in self.named_parameters():
    #         kg = k + '.grad'
    #         if grad[kg] is not None:
    #             v.grad = grad[kg].clone().detach()
    #         else:
    #             v.grad = None
                
    
    #  # returns gradients in dict vector form (this one is also mine)
    def capture_gradients(self):
        if self.is_functional():
            self.load_weights(self.names, \
                              self.orig_params, \
                                  as_params=False) 
            # self.restore_model()
        g = {}
        for k, v in self.named_parameters():
            gnext = v.grad
            knext = k + '.grad'
            if gnext is not None:
                next_entry = {knext: gnext.clone().detach()}
            else:
                next_entry = {knext: None}
            g.update(next_entry)
        return g
    
    def count_params(self):
        count = 0
        prod, T = torch.prod, torch.tensor
        for p in self.parameters():
            count += prod(T(p.shape))
        return count.item()
    
    def set_criterion(self, objective):
        assert callable(objective)
        self.objective = objective
    
    def zero_grad(self):
        for v in self.parameters():
            if v.grad is not None:
                v.grad[:] = 0.0
    
    def flatten_v(self, vt=None):
        if vt is None:
            return self.flatten_p()
        
        vf = None
        for v in vt:
            if vf is None:
                vf = torch.flatten(v)
            else:
                vf = torch.cat((vf, torch.flatten(v)))
        return vf        
        
        
    def flatten_p(self):
        pf = None
        for p in self.parameters():
            if pf is None:
                pf = torch.flatten(p)
            else:
                pf = torch.cat((pf, torch.flatten(p)))
        return pf        
        
# vfmin = torch.cat((torch.flatten(vmin[0]),torch.flatten(vmin[1])))

    def flatten_H(self, Ht):
        N = self.count_params()
        H = torch.zeros(N, N)
        ps = [p.shape for p in self.parameters()]

        top,left = 0,0
        for i,si in enumerate(ps):
            for j,sj in enumerate(ps):
                h = copy.deepcopy(Ht[i][j])
                h = torch.flatten(h, end_dim=len(si)-1)
                h = torch.flatten(h, start_dim=1)
                
                height, width = h.shape
                H[top:(top+height), left:(left+width)] = h
                left = (left + width) % N
                
            top += height
        
        
        
        return H
        
       
    def vH(self, v=None, get_hessian=False):  # easy interface to vector-Hessian product. 
        # You can multiply any vector v into the Hessian, but....
        if v is None:
            v = self.default_v  # just 1's, shaped like params.

        # self.orig_grad = self.capture_gradients()
        # This is the loss function that allows Hessian computations
        #   with respect to parameters rather than input. It wraps
        #   whatever loss function we are training (self.objective), via
        #   this SuperModel class. 
        def loss_wrt_params(*new_params):
            if self.is_functional:
                self.load_weights(self.names, new_params, as_params=False) # Weird! We removed the params before. 
            
            pred = self.forward(self.x_now)
            
            # if len(targs) == 0 and self.no_warning is False:
            #     self.no_warning = True
            #     print('When calling SuperModel.vH(), you must include any')
            #     print('targets that are needed to evaluate the loss function.')
            #     print('I have no way to check that you have sent the right')
            #     print('targets for a given input. So be careful!')
            #     _ = input('Hit enter to continue...')
                
            loss = self.objective(pred, self.y_now)    
            
            self.zero_grad()
            
            loss.backward(retain_graph=True)
            return loss
    
        if not self.is_functional():
            self.make_functional() # monkey-patching, step 1...
            
        params2pass = tuple(p.detach().requires_grad_() for p in self.orig_params)
    
        _ = loss_wrt_params(*self.orig_params) # Unpatched inside function...
        # _ = self.capture_gradients()

        self.make_functional() # monkey-patching now complete. Wow.

        _, v_dot_hessian = \
            torch.autograd.functional.vhp(loss_wrt_params,
                                          params2pass,
                                          v, strict=True)
            
        if get_hessian:
            self.hessian = \
            torch.autograd.functional.hessian(loss_wrt_params,
                                              params2pass, 
                                              strict=True)
            # self.hessian = self.flatten_H(Ht)
            
        self.load_weights(self.names, \
                          self.orig_params, \
                              as_params=False) 
        # self.restore_model(orig=True) # did I botch this with grad stuff??
        
        return v_dot_hessian

        
    def ext_eigen_H(self):
        pass
        
    def max_eigen_H(self):
        v_dot_hessian = self.vH()
            
        vnext = copy.deepcopy(v_dot_hessian)
        count = 0
        while True:
            count += 1
            scale_vect(vnext, max_vect_comp(vnext, maxabs=True))
            vprev = vnext # agf_vhp makes a new copy, whew....
            vnext = self.vH(v=vnext)
            cos_dtht = cos_angle_vect(vnext, vprev)
            
            if count % 50 == 0:
                print('cos_dtht=',cos_dtht)
            if torch.abs(torch.abs(cos_dtht) - 1) < self.allowed_angular_error:
                break
            elif count > self.max_iterations:
                print('ACK! Too many iterations  in max_eigen_H...')
                return None
        
        vHmax = vnext
        sign = torch.sign(dot_vect(vHmax,vprev))
        lambda_max = sign*torch.sqrt(dot_vect(vnext, vnext)/ \
                                     dot_vect(vprev, vprev))

        scale_vect(vnext, max_vect_comp(vnext, maxabs=True))

        return vnext, lambda_max

    def min_eigen_H(self, lambda_max):
        v_dot_hessian = self.vH()
            
        vnext = copy.deepcopy(v_dot_hessian)
        count = 0
        while True:
            count += 1
            scale_vect(vnext, max_vect_comp(vnext, maxabs=True))
            vprev = vnext
            
            decr = copy.deepcopy(vprev)
            scale_vect(decr, -1/lambda_max) # scale_vect *divides*! in-place!
            
            vnext = add_vect(self.vH(v=vprev), decr)
            cos_dtht = cos_angle_vect(vnext, vprev)
            
            if torch.abs(torch.abs(cos_dtht) - 1) < self.allowed_cos_error:
                break
            elif count > self.max_iterations:
                print('ACK! Too many iterations  in min_eigen_H...')
                return None
        
        lmin_minus_lmax = dot_vect(vnext, vprev)/dot_vect(vprev,vprev)
        lambda_min = lmin_minus_lmax + lambda_max
                  
        scale_vect(vnext, max_vect_comp(vnext, maxabs=True))
        return vnext, lambda_min

    def line_scan(self, vhat, length=1, npts=100, symmetric=True):
        sd0 = self.state_dict()
        sd1 = copy.deepcopy(sd0)
        vhat0 = copy.deepcopy(vhat)
        norm_vect(vhat)
        dp = copy.deepcopy(vhat)
        scale_vect(dp, npts/length)
        
        if symmetric:
            scale_vect(vhat, 2.0/length)
            for k,vhatk in zip(sd1.keys(), vhat):
                sd1[k] -= vhatk
            scale_vect(vhat, 0.5*length)
        
        lchk = torch.zeros(npts)
        for i in range(npts):
            for k,dpi in zip(sd1.keys(), dp):
                sd1[k]+=dpi
                self.load_state_dict(sd1)
            lchk[i]=self.objective(self(self.x_now), self.y_now)
    
        self.load_state_dict(sd0)
        vhat = copy.deepcopy(vhat0)
        
        x = torch.linspace(-length/2.0, length/2.0, npts)
        
        return x, lchk
    
    def report(self):
        if self.y_now is None:
            print('Need to define targets before calling report().')
            
        vmax, lmax = self.max_eigen_H()
        vmin, lmin = self.min_eigen_H(lmax)
        g = dict_to_tuple(self.capture_gradients())
        
        rad2deg = 180/np.pi
        g2max = angle_vect(g, vmax)*rad2deg
        g2min = angle_vect(g, vmin)*rad2deg
        max2min = angle_vect(vmax, vmin)*rad2deg
        
        print('\n\n===========================================\n')
        print('Max eigenvalue:',lmax)
        print('Min eigenvalue:',lmin)
        
        print('grad to vmax',g2max)
        print('grad to vmin',g2min)
        print('vmin to vmax',max2min)
        print('============================================\n\n')

def store_inputs(self, x): 
    # How is self defined here? This does work! I just don't get why. The
    #   idea is to have the model input available to all SuperModel methods,
    #   which the toy examples do via a global reference. 
    #
    # Anyway, this is the forward_pre_hook that is registered at
    #   instantiation. 
    self.x_now = x[0] # get rid of unused extra arg included in hook call
    
    if self.default_v is None:  # just a startup issue: need to know shape
        self.default_v = tuple([torch.ones_like(p.clone().detach()) \
                                      for p in self.parameters()])

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
   
def add_vect(a,b):
    c = list(copy.deepcopy(a))
    try:
        for i,(ai,bi) in enumerate(zip(a,b)):
            assert(ai.shape == bi.shape)
            c[i] = ai + bi           
    except TypeError:
        assert(a.shape==b.shape)
        c = a + b
    except AssertionError:
        print('OOPS!', a.shape, b.shape)
        c = None
    return tuple(c)

def angle_vect(a,b):
    cos_ab = dot_vect(a,b)/torch.sqrt(dot_vect(a,a)*dot_vect(b,b))
    
    return torch.acos(cos_ab)

def cos_angle_vect(a,b):
    cos_ab = dot_vect(a,b)/torch.sqrt(dot_vect(a,a)*dot_vect(b,b))
    
    return cos_ab
    
def dict_to_tuple(d):
    t = []
    for k,v in d.items():
        t.append(v)
    return tuple(t)

if __name__ == "__main__":
    
    torch.manual_seed(0)

    class SimpleMLP(SuperModel):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_dim, out_dim),
            )

        def forward(self, x):
            '''Forward pass'''
            return self.layers(x)

    class LessSimple(SuperModel):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_dim, out_dim),
            )

        def forward(self, x):
            '''Forward pass'''
            return torch.sigmoid(self.layers(x))
        
    
    in_dim, out_dim = 3, 2
    xglobal = torch.rand((in_dim,)) 

    mlp = SimpleMLP(in_dim, out_dim) 
    mlp.objective = lambda x,y : torch.sum((x-y)**2)
    
    hmlp = LessSimple(in_dim, out_dim)
    hmlp.objective = lambda x,y : torch.sum((x-y)**2)
    
    out = mlp(xglobal)  
    mlp.y_now = torch.randn_like(out)
    
    outh = hmlp(xglobal)
    hmlp.y_now = mlp.y_now
    # print('grads before eigs', mlp.capture_gradients())


    vmax, lmax = mlp.max_eigen_H()
    vmin, lmin = mlp.min_eigen_H(lmax)
    
    mlp.report()
    
    x0, x1, x2 = xglobal
    H =2.0*torch.tensor(\
       [[x0*x0, x1*x0, x2*x0,     0,     0,     0, x0,  0],\
        [x1*x0, x1*x1, x2*x1,     0,     0,     0, x1,  0],
        [x2*x0, x2*x1, x2*x2,     0,     0,     0, x2,  0],
        [    0,     0,     0, x0*x0, x1*x0, x2*x0,  0, x0],
        [    0,     0,     0, x1*x0, x1*x1, x2*x1,  0, x1],
        [    0,     0,     0, x2*x0, x1*x2, x2*x2,  0, x2],
        [   x0,    x1,    x2,     0,     0,     0,  1,  0],
        [    0,     0,     0,    x0,    x1,    x2,  0,  1]])
        
    vpt = tuple(mlp.parameters())
    vp = torch.tensor([vpt[0][0,0],vpt[0][0,1],vpt[0][0,2],
                       vpt[0][1,0],vpt[0][1,1],vpt[0][1,2],
                       vpt[1][0],vpt[1][1]])
    
    vpH = mlp.vH(v=vpt)
    vpHchk = vp@H
    
    hvmax, hlmax = hmlp.max_eigen_H()
    hvmin, hlmin = hmlp.min_eigen_H(hlmax)
    
    vHmax = hmlp.vH(v=hvmax)
    print('hvmax', hvmax)
    scale_vect(hvmax,-1/hlmax)
    print('Hv - lv', add_vect(vHmax, hvmax))
    
    vHmin = hmlp.vH(v=hvmin)
    print('hvmin', hvmin)
    scale_vect(hvmin,-1/hlmin)
    print('Hv - lv', add_vect(vHmin, hvmin))
    
    
    # # grad_tuple = dict_to_tuple(mlp.capture_gradients())
    # print('grad_tuple is', grad_tuple)
    
    # print("eigenvector is", v_eig)
    # print('angle between eigenvector and gradient is', \
    #       int(angle_vect(grad_tuple, v_eig)*180/np.pi),'degrees.')
        
    # print('Largest eigenvalue of the Hessian is', lambda_max)

    # gradmag = torch.sqrt(dot_vect(grad_tuple, grad_tuple)).item()
    # print('gradient magnitude is', gradmag)
    
    # vhpmag = torch.sqrt(dot_vect(v_eig, v_eig)).item()
    # print('VHP magnitude is', vhpmag)
    
    
    # vunit = copy.deepcopy(v_eig)
    # norm_vect(vunit)
    # vuH = mlp.vH(vunit)
        
    
    # fpp = torch.sqrt(dot_vect(vuH, vuH))    
    # # this is the step size, in the vunit direction, that should bring 
    # #   the gradient magnitude to zero, if the gradient varies linearly. 
    # # It's delta_x = y_now/slope...a step that big should bring y to zero. 
    # scale = gradmag/fpp  
    # print('Expect zero gradient after step of',scale.item())
    
    
    # # The following shows that I can step and then step back, with plain
    # #   SGD. This is stepping along the gradient though, not in the max
    # #   eigen direction. 
    
    # if mlp.is_functional():
    #     mlp.restore_model()
        
    # optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-3)
    
    # loss0 = mlp.objective(mlp(xglobal))
    # print('Initial loss ', loss0.item())
   
    # optimizer.step()

    # # Get updated loss
    # out1 = mlp(xglobal)
    # loss1 = mlp.objective(out1)
    # print('Updated loss ', loss1.item())

    # # Use negative lr to revert loss
    # optimizer.param_groups[0]['lr'] = -1. * optimizer.param_groups[0]['lr']
    
    # optimizer.step()
    # out2 = mlp(xglobal)
    # loss2 = mlp.objective(out2)
    # print('Reverted loss ', loss2.item())
