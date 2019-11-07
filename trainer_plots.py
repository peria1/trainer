# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 06:19:45 2019

@author: Bill
"""

def basic_loss_plot(viewer,d):
    d.ax[0].plot(viewer.trainer.train_loss_history)
    d.ax[1].plot(viewer.trainer.test_loss_history)

def residual_plot(viewer,d):
    from trainer_utils import date_for_filename

    residuals = viewer.trainer.yp - \
        viewer.trainer.model(viewer.trainer.xtest).cpu().detach().numpy()
    d.ax.plot(viewer.trainer.yp, residuals,'o')
    d.ax.set_title(date_for_filename())
    d.ax.hlines(0, min(viewer.trainer.yp),max(viewer.trainer.yp))