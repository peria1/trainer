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
        ': target minus prediction versus target'
    d.ax.set_title(title_str)
    xlim = d.ax.get_xlim()
    d.ax.hlines(0, xlim[0], xlim[1])
    
def weight_plot(viewer, d):
    if d.first:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons
        
        d.first = False
        
        d.plist = viewer.trainer.get_named_weight_list()
        d.names = [n for n,p in d.plist]
        axcolor = 'lightgoldenrodyellow'
        plt.figure(d.fig.number);
        plt.subplots_adjust(left=0.4)        
        rax = plt.axes([0.05, 0.05, 0.15, 0.9], facecolor=axcolor)
        d.radio = RadioButtons(rax, d.names)
        d.layer_to_show = 0
        def set_layer(event):
            d.layer_to_show =\
            [i for i,n in enumerate(d.names) if event in n]
            try:
                assert len(d.layer_to_show)==1
            except AssertionError:
                print('More than one layer matches',event,'. Figure this out!')
            d.layer_to_show = d.layer_to_show[0]
            
        d.radio.on_clicked(set_layer)

    if len(d.plist) == 1:
        im = d.plist[0][1].cpu().detach().numpy()
        d.ax.imshow(im)
    else:
        im = d.plist[d.layer_to_show][1].cpu().detach().numpy()
        d.ax.imshow(im)
        d.ax.set_title(d.names[d.layer_to_show])
#        for i,a in enumerate(d.ax.flatten()):
#            im = d.plist[i][1].cpu().detach().numpy()
#            a.imshow(im)
#            
    

    