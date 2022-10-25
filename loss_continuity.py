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

from super_model import dict_to_tuple
# from torch import optim
# from torch.autograd.functional import vhp as VHP

def linear_step(optimizer):
    optimizer.step()

def set_learning_rate(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

def reset_model(model, sd, g):
    model.load_state_dict(sd)
    for k, v in model.named_parameters():
        # v.grad = copy.deepcopy(g[k])
        v.grad = g[k].clone().detach()
        
def get_model_state(model):
    return copy.deepcopy(model.state_dict())

# def save_params(model):
#     psave = []
#     for p in model.parameters():
#         psave.append(copy.deepcopy(p))
#     return psave

def build_gradient_vector(model): # not a dict vector, a 1D tensor
    g = None
    for p in model.parameters():
        pcpu = p.grad.flatten().detach().cpu()
        if g is None:
            g = pcpu
        else:
            g = torch.cat((g, pcpu))
    return g
 
def capture_gradients(model):
    return {k: v.grad.clone().detach() for k,v in model.named_parameters()}
          

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

def dict_vect_angle(gdict0, gdict1): # between 2 dict vects, radians
    dot = 0.0
    norm0 = 0.0
    norm1 = 0.0
    for k, v in gdict0.items():
        g0 = gdict0[k]
        g1 = gdict1[k]
        dot += torch.sum(g0*g1)
        norm0 += torch.sum(g0**2)
        norm1 += torch.sum(g1**2)
    
    dot /= torch.sqrt(norm0*norm1)
    return torch.acos(torch.clip(dot, -1.0, 1.0))

def normsq_grad(model):
    absL2 = 0
    for p in model.parameters():
        absL2 += torch.sum(p.grad**2)
        
    return absL2

def vH(v, model):
    pass

def normalize_dict_vector(dv, in_place=False):
    norm = get_norm_dict_vector(dv)
         
    if in_place:
        for k, v in dv.items():
            v /= norm
        ret = dv
    else:     
        ret = {}
        for k, v in dv.items():
            ret.update({k: (v/norm).clone().detach()})
    return ret

def get_norm_dict_vector(dv):
    norm2 = 0.0
    for k, v in dv.items():
        norm2 += torch.sum(v**2)
    return torch.sqrt(norm2)

def dot_dict_vectors(a,b):
    dot_ab = 0.0
    for k, v in a.items():
        dot_ab += torch.sum(a[k]*b[k])
    return dot_ab

def project_dict_vector(a, b):
    factor = dot_dict_vectors(a,b)/dot_dict_vectors(b,b) 
    a_onto_b = {}
    for k,v in b.items(): # vector nature from b
        a_onto_b.update({k : v*factor})            
    return a_onto_b
    
def subtract_dict_vector(a, b):
    c = {}
    for k, v in a.items():
        c.update({k: a[k]-b[k]})
    return c

def add_dict_vector(a, b):
    c = {}
    for k, v in a.items():
        c.update({k: a[k]+b[k]})
    return c

def rand_dict_vector(a):
    c = {}
    for k, v in a.items():
        c.update({k: torch.randn_like(v)})
    return c

def param_dict_vector(model): # current model position as dict vector
    ret = {}
    for k, v in model.named_parameters():
        # ret.update({k: copy.deepcopy(v)})
        ret.update({k: v.clone().detach()})
    return ret
    
def grad_basis(g, g1prev, g2prev):
    xhat = copy.deepcopy(g1prev)    # previous gradient, i.e. step just taken
    
    ypx =  project_dict_vector(g, xhat)
    yhat = subtract_dict_vector(g, ypx) # perp to xhat, in plane spanned by prev 
                                        #   and curr grad. 
    
    zpx = project_dict_vector(g2prev, xhat)
    zpy = project_dict_vector(g2prev, yhat)
    zhat = subtract_dict_vector(g2prev, add_dict_vector(zpx, zpy)) 
    # zhat is perp to xhat and yhat...I am confused about what plane it's in. I've 
    #   been calling it the "out-of-plane" component. 
    
    for ehat in [xhat, yhat, zhat]:
        normalize_dict_vector(ehat, in_place=True)
    
    return xhat, yhat, zhat

def count_params(model):
    count = 0
    for p in model.parameters():
        count += np.prod(p.shape)
    return count

if __name__=="__main__":
    
    # npts kwarg gives degree of polynomials
    tvt = trainer.trainer(models.n_double_nout, problems.roots_of_poly, npts=10)
    
    model = tvt.model
    optimizer = tvt.optimizer
   
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
    
    # assert(1==0)
    
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
        
        
        angle[i] = dict_vect_angle(gsave, capture_gradients(model)) # between current grad and grad in gsave
    
    
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
    
    try: 
        assert(tvt.use_GAlr)
    except AssertionError:
        print('Oops, you turned off gradient-adaptive stepping...')
        # raise AssertionError
    
    neps = 9
    eps = 10**np.linspace(-8,0,neps)
    Lratio = np.zeros(neps)
    for i, e in enumerate(eps):
        tvt.eps = e
        
        reset_model(model, sdsave, gsave)  # back to starting model. 
        loss = tvt.criterion(model(example), target) # starting loss
        # loss.backward() # do I need this here? 
    
        saveloss = copy.copy(loss) # can't do deepcopy yet, "user-created only". Gradients don't flow to copy.
        gradnorm = normsq_grad(model).item() # save this to check if step is right later
    
        tvt.train_step(example, target) # model parameters updated
        loss = tvt.criterion(model(example), target) # new loss
        Lratio[i] = -(loss.item()-saveloss.item())/saveloss.item() # should equal e
        
    niter = 2000
    loss_history = np.zeros(niter)
    grad_history = np.zeros(niter)
    gangle_history = np.zeros(niter)
    step_history = np.zeros(niter)
    sangle_history = np.zeros(niter)
    
    vm1 = np.zeros((niter,3))
    v0 = np.zeros((niter,3))
    vp1 = np.zeros((niter,3))
    xyz = np.zeros((niter, 3))
    
    tvt.eps = 1e-3
    i=0
    is_power_law = True
    g0 = copy.deepcopy(gsave)
    gp2 = rand_dict_vector(g0)
    gp1 = rand_dict_vector(g0)
    s0 = copy.deepcopy(sdsave)
    reset_model(model, sdsave, gsave)
    alpha0 = param_dict_vector(model)
    
    _ = tvt.get_more_data() # needed to set targets in model
    print('before training:')
    print('grad norm is', normsq_grad(model))
    model.report()
    # assert(1==0)
    # _ = model.vH() # causes grad to grow forever
    # calling vH just once means that in future train_step calls, aparently
    # the gradient is never zero'd again. WHy? 
    #
    # I can do train_step() calls by hand, and I see that the normsq_grad() return 
    #   value begins increasing fairly quickly after a vH() call. This seems to 
    #   legitimately reflect increases in the grad attributes. (I capture them 
    #   via capture_gradients()). But what stops happening is optimizer.zero_grad().
    #   WHY?!
    #
    # 17-Oct-2022 I figured this out I think. I can put the optimizer and model 
    #   params in lists; the are equal at first. Then, after a 
    #   call to vH, the model param ids have changed, while the optimizer 
    #   param ids remain the same. 
    #
    #   So, the optimizer still adjusts the same param set, while the params of
    #   the model remain untouched. Since the gradients are attributes of 
    #   these imposter params, the optimizer also leaves those untouched: in
    #   particular, it never zeros them since it doesn't know about them.  
    #
    #   But backward() knows! Backward() keeps incrementing the grads of the 
    #   imposter params, so they grow forever. 
    #
    #   Backward() just references whatever is in the model when it is called.
    #   Optimizer only has access to whatever reference was passed to it when 
    #   it was first created. 
    #
    #   So this is a weird problem in which "the gradient" grew forever, but 
    #   "the model params" never changed! So there had to be two param sets,
    #   probably because I copied somewhere when I should have moved a pointer.
     
    
    # model.report() # this kills parameter updates!
    # reset_model(model, sdsave, gsave) # this does not help!
    # _ = dict_to_tuple(model.capture_gradients()) # does not kill updates?
    # vmax, lmax = model.max_eigen_H() # zeros gradient! 

    print('Training model with', count_params(model), 'parameters.')
    # while is_power_law and (i < niter):
    while (i < niter):
        if i % 100 == 0:
            print(i,'...')
        example, target = tvt.get_more_data()
        loss_history[i] = tvt.train_step(example, target, linearize=False) # updates parameters
        alpha1 = param_dict_vector(model)
        
        dalpha = subtract_dict_vector(alpha1, alpha0)
        ehat = grad_basis(g0, gp1, gp2)
        for j in range(3):
            xyz[i, j] = dot_dict_vectors(ehat[j], dalpha)
        
        grad_history[i] = normsq_grad(model)
        g1 = capture_gradients(tvt.model)   # copies gradients
        gangle_history[i] = dict_vect_angle(g0, g1)

        # s1 = get_model_state()
        # step_history[i] = normsq_step(s0, s1)
        # g1 = capture_gradients(tvt.model)
        # gangle_history[i] = dict_vect_angle(g0, g1)
        
        
        
        lhi = loss_history[i]
        is_power_law = np.abs(loss_history[0]*(1-tvt.eps)**i - lhi)/lhi < 0.2
        
        gp2, gp1, g0 = gp1, g0, g1
        alpha0 = alpha1
        i += 1
    
    print('after training:')
    model.report()
    
    istop = i
    loss_history = loss_history[0:istop]
    grad_history = grad_history[0:istop]
    gangle_history = gangle_history[0:istop]
    xyz = np.cumsum(xyz[0:istop, :], 0)
    sangle_history = sangle_history[0:istop]
    
    npow = np.round(1.33*istop)
    power_law_pred = loss_history[0]*(1-tvt.eps)**np.linspace(0,npow-1,npow)
    plt.figure()
    plt.plot(loss_history)
    plt.plot(power_law_pred)
    plt.title('compare loss to power law')
    plt.xlabel('iteration')
    plt.ylabel(tvt.criterion)
        
    plt.figure()   
    plt.loglog(eps, Lratio,'o-')
    plt.loglog(eps, eps)
    plt.title('Compare actual loss change to goal')
    plt.xlabel('desired change')
    plt.ylabel('measured change')

    
    figg, axg = plt.subplots(3,1, sharex=True)
    axg[0].plot(loss_history)
    # axg[0].plot(loss_history[0]*(1-tvt.eps)**np.linspace(0,npow-1,npow))
    axg[0].plot(power_law_pred)
    axg[0].set_ylabel('loss')
    
    axg[1].plot(grad_history)
    axg[1].set_ylabel('grad magnitude')
    axg[2].plot(gangle_history)
    axg[2].set_ylabel('grad angle')

    # figs, axs = plt.subplots(3,1, sharex=True)
    # axs[0].plot(loss_history)
    # axs[0].plot(loss_history[0]*(1-tvt.eps)**np.linspace(0,npow-1,npow))

    # axs[1].plot(step_history)
    # axs[2].plot(sangle_history)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(xyz[:,0], xyz[:,1], xyz[:,2],'-o')
    ax.scatter3D(xyz[0,0], xyz[0,1], xyz[0,2], 'og', s=80)
    ax.set_xlabel('prev grad')
    ax.set_ylabel('grad rot plane')
    ax.set_zlabel('out of plane')
    
    ptp = max(np.ptp(xyz[:,0]), np.ptp(xyz[:,1]), np.ptp(xyz[:,2]))
    # ax.set_box_aspect((np.ptp(xyz[:,0]), np.ptp(xyz[:,1]), np.ptp(xyz[:,2])))
    ax.set_box_aspect((ptp, ptp, ptp))

    for item in [ax.xaxis.label,ax.yaxis.label,ax.zaxis.label]:
        item.set_fontsize(24)    

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(xyz[:,0], xyz[:,1], loss_history,'-o')
    ax.plot3D(xyz[:,0], xyz[:,1], power_law_pred[0:istop],'-o')

    ax.scatter3D(xyz[0,0], xyz[0,1], loss_history[0], 'og', s=80)
    ax.set_xlabel('prev grad')
    ax.set_ylabel('grad rot plane')
    ax.set_zlabel('Loss')
    # ax.set_box_aspect((np.ptp(xyz[:,0]), np.ptp(xyz[:,0]), np.ptp(loss_history)))

    for item in [ax.xaxis.label,ax.yaxis.label,ax.zaxis.label]:
        item.set_fontsize(24)    



    sdnow = get_model_state(model)
    gnow = capture_gradients(model)
    neps = 9
    eps = 10**np.linspace(-8,0,neps)
    Lratio_now = np.zeros(neps)
    for i, e in enumerate(eps):
        tvt.eps = e
        
        reset_model(model, sdnow, gnow)  # back to starting model. 
        loss = tvt.criterion(model(example), target) # starting loss
        # loss.backward() # do I need this here? 
    
        saveloss = copy.copy(loss) # can't do deepcopy yet, "user-created only". Gradients don't flow to copy.
        gradnorm = normsq_grad(model).item() # save this to check if step is right later
    
        tvt.train_step(example, target) # model parameters updated
        loss = tvt.criterion(model(example), target) # new loss
        Lratio_now[i] = -(loss.item()-saveloss.item())/saveloss.item() # should equal e


    vmax, lmax = model.max_eigen_H()
    vmin, lmin = model.min_eigen_H(lmax)
    
    xpos, ypos = model.line_scan(vmax, length=1, npts=200)
    xneg, yneg = model.line_scan(vmin, length=1, npts=200)
    
    plt.figure()
    plt.plot(xpos, ypos)
    plt.plot(xneg, yneg)

    xr, yr, ras = model.raster_scan(vmin, vmax)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(xr, yr, ras.numpy(), rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('loss in eig-extreme plane');
    ax.set_xlabel('min eig: '+ str(lmin))
    ax.set_xlabel('max eig: '+ str(lmax))
    
    plt.ion()
    plt.show()

    # func, inputs, v
    # mpars = tuple([p for p in model.parameters()])
    # func = lambda mpars : loss
    # alpha_H = VHP(func, mpars, v=mpars, strict=True)

    # print('\nNext two numbers are expected loss and current loss:')
    # print(saveloss.item() - optimizer.param_groups[0]['lr'] * gradnorm) # what we expected
    # print(loss.item())  # what we have now
    
    # print('\n now it''s eps and realized eps:')
    # print(tvt.eps)
    # print(-(loss.item()-saveloss.item())/saveloss.item())
    
    # plt.figure()
    # plt.plot(s, lincheck,'-o')
    
    # plt.figure()
    # plt.plot(s, angle, '-o')
    
    # plt.show()
   
    
    