# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 06:19:45 2019

@author: Bill
"""

def basic_loss_plot(viewer,d):
    d.ax[0].plot(viewer.trainer.train_loss_history)
    d.ax[0].set_title(viewer.trainer.get_problem_name() +': training loss')
    d.ax[1].plot(viewer.trainer.test_loss_history)
    d.ax[1].set_title('test loss')
    ylim = d.ax[1].get_ylim()
    d.ax[1].set_ylim(0,ylim[1])
    
    max_loss = viewer.trainer.max_loss
    if max_loss < ylim[1]:
        xlim = d.ax[1].get_xlim()
        d.ax[1].hlines(max_loss, xlim[0], xlim[1])

def residual_plot(viewer,d):
    residuals = viewer.trainer.yp - \
        viewer.trainer.model(viewer.trainer.xtest).cpu().detach().numpy()
    d.ax.plot(viewer.trainer.yp, residuals,'o')
    title_str = viewer.trainer.get_problem_name() + \
        ': target minus prediction vs. target'
    d.ax.set_title(title_str)
    xlim = d.ax.get_xlim()
    d.ax.hlines(0, xlim[0], xlim[1])
    
def weight_plot(viewer, d):
    if d.first:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons
                
        d.plist = viewer.trainer.get_named_weight_list()
        d.layer_names = [n for n,p in d.plist]
        axcolor = 'lightgoldenrodyellow'
        plt.figure(d.fig.number);
        plt.subplots_adjust(left=0.4)
        d.ax.set_axis_off()
        rax = plt.axes([0.05, 0.05, 0.15, 0.9], facecolor=axcolor)
        d.radio = RadioButtons(rax, d.layer_names)

        d.layer_to_show = 0
        def set_layer_and_update(event):
            d.layer_to_show =\
            [i for i,n in enumerate(d.layer_names) if event in n]
            try:
                assert len(d.layer_to_show)==1
            except AssertionError:
                print('More than one layer matches',event,'. Figure this out!')
            d.layer_to_show = d.layer_to_show[0]
            if not d.first:
                d.update(viewer,d)
                
        d.radio.on_clicked(set_layer_and_update)
        d.plt = plt
        d.cbar_axis = d.plt.axes([0.25, 0.05, 0.05, 0.9])

    im = d.plist[d.layer_to_show][1].cpu().detach().numpy()
    d.ax.set_axis_off()
    d.mappable = d.ax.imshow(im)
    d.ax.set_title(d.layer_names[d.layer_to_show])
    d.cbar_axis.clear()
    d.plt.colorbar(mappable=d.mappable, cax=d.cbar_axis)
    d.fig.canvas.draw()
    d.fig.canvas.flush_events()
    
#def numbers_check(viewer, d):
#    if d.first:
#        import matplotlib.pyplot as plt
#        from matplotlib.widgets import TextBox
#        import torch
#        d.torch = torch
#
#        d.plt = plt
#        plt.figure(d.fig.number, figsize=(12,3));
#        left, bottom, width, row_height = (0.05, 0.05, 0.5, 0.075)
#        
#        net = viewer.trainer.model
#        d.layer_names =  [name for name, module in net.named_modules()\
#                         if (len(module._modules) == 0) and (name != 'custom_loss')]
#        d.plist = viewer.trainer.get_named_weight_list()
#        d.weight_names = [n for n,p in d.plist]
#        
#    
def dataflow_plot(viewer, d):
    if d.first:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons
        import torch
        d.torch = torch

        plt.figure(d.fig.number);
        plt.subplots_adjust(left=0.4)

        net = viewer.trainer.model
        d.layer_names =  [name for name, module in net.named_modules()\
                         if (len(module._modules) == 0) and (name != 'custom_loss')]
        d.modules = [module for name, module in net.named_modules()\
                         if (len(module._modules) == 0) and (name != 'custom_loss')]
        d.layer_to_show = 0
        d.x = viewer.trainer.xtest
        d.plt = plt
        d.cbar_axis = d.plt.axes([0.25, 0.05, 0.05, 0.9])


        axcolor = 'lightgoldenrodyellow'
        rax = plt.axes([0.05, 0.05, 0.15, 0.9], facecolor=axcolor)
        d.radio = RadioButtons(rax, d.layer_names)
        d.layer_to_show = 0
        
        def set_layer_and_update(event):
            d.layer_to_show =\
            [i for i,n in enumerate(d.layer_names) if event in n]
            try:
                assert len(d.layer_to_show)==1
            except AssertionError:
                print('More than one layer matches',event,'. Figure this out!')
            d.layer_to_show = d.layer_to_show[0]
            if not d.first:
                d.update(viewer,d)

        d.radio.on_clicked(set_layer_and_update)


        def capture_data_hook(self, input, output):
            d.current_data = output.cpu().detach().numpy()
        d.hook = capture_data_hook
            
    mp = d.modules[d.layer_to_show]
    chandle = mp.register_forward_hook(d.hook)
    viewer.trainer.model(d.x)
    chandle.remove()
    
#    if not FEATURE_MAPS:
#        print('FEATURE_MAPS not set in capture_data_hook.')
        
    d.ax.imshow(d.current_data)
    d.ax.set_title(d.layer_names[d.layer_to_show])
    d.ax.set_axis_off()
    d.mappable = d.ax.imshow(d.current_data)
    d.ax.set_title(d.layer_names[d.layer_to_show]+ ' output data')
    d.cbar_axis.clear()
    d.plt.colorbar(mappable=d.mappable, cax=d.cbar_axis)
    d.fig.canvas.draw()
    d.fig.canvas.flush_events()
   
def example_plot(viewer, d):
    if d.first:
        import numpy as np
        d.np = np
        predsize = viewer.trainer.yp.shape
        d.n_examples = predsize[0]
        
    pick = int(d.np.random.uniform(0,d.n_examples,size=(1)))
    pred = viewer.trainer.model(viewer.trainer.xtest[pick,:]).cpu().detach().numpy()
#    inp = viewer.trainer.xp[pick,:]
#    ord = d.np.argsort(inp)
    target = viewer.trainer.yp[pick,:]
    pline, = d.ax.plot(pred)
    tline, = d.ax.plot(target)
    d.ax.set_title('Test Data Example '+str(pick))
    d.ax.set_xlabel('input')
    d.ax.legend((pline,tline),('prediction','target'))
   

def svd_weight_plot(viewer, d):
    if d.first:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons
        import numpy as np
        d.np = np
        
        d.plist = viewer.trainer.get_named_weight_list()
        d.layer_names = [n for n,p in d.plist]
        axcolor = 'lightgoldenrodyellow'
        plt.figure(d.fig.number);
        plt.subplots_adjust(left=0.4)
        d.ax.set_axis_off()
        rax = plt.axes([0.05, 0.05, 0.15, 0.9], facecolor=axcolor)
        d.radio = RadioButtons(rax, d.layer_names)

        d.layer_to_show = 0
        def set_layer_and_update(event):
            d.layer_to_show =\
            [i for i,n in enumerate(d.layer_names) if event in n]
            try:
                assert len(d.layer_to_show)==1
            except AssertionError:
                print('More than one layer matches',event,'. Figure this out!')
            d.layer_to_show = d.layer_to_show[0]
            if not d.first:
                d.update(viewer,d)
                
        d.radio.on_clicked(set_layer_and_update)
        d.plt = plt
#        d.cbar_axis = d.plt.axes([0.25, 0.05, 0.05, 0.9])

    im = d.plist[d.layer_to_show][1].cpu().detach().numpy()
#    d.ax.set_axis_off()
#    d.mappable = d.ax.imshow(im)
    _,s,_ = d.np.linalg.svd(im)
    d.ax.plot(s)
    d.ax.set_title(d.layer_names[d.layer_to_show])
#    d.cbar_axis.clear()
#    d.plt.colorbar(mappable=d.mappable, cax=d.cbar_axis)
    d.fig.canvas.draw()
    d.fig.canvas.flush_events()
