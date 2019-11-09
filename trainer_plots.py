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