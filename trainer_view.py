# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 06:16:10 2019

@author: Bill
"""

import matplotlib.pyplot as plt
import trainer as tr
from models import *
from trainer_utils import getnum
from matplotlib.widgets import TextBox, Button

import asyncio

def fire_and_forget(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, *kwargs)

    return wrapped


class trainer_view():
    def __init__(self, *args, **kwargs):
        self.fig, self.ax = plt.subplots(1,1)       
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)       
        self.fig.canvas.mpl_connect('key_press_event', self.process_key) 
        self.fig.canvas.mpl_connect('button_press_event', self.process_button) 
        
        self.trainer = tr.trainer(*args, **kwargs, viewer=self)
        self.trainer.pause = False
        self.update_plot = False

        axbox = plt.axes([0.7, 0.8, 0.2, 0.075]) # left, bottom, width, height
        self.lr_box = TextBox(axbox, 'learning rate', \
                              initial=str(self.get_learning_rate()))
        self.lr_box.on_submit(self.set_learning_rate)
        
        axbutton = plt.axes([0.7, 0.7, 0.2, 0.075])
        self.start_button = Button(axbutton, 'Start')
        self.start_button.label.set_color('white')
        self.start_button.label.set_fontweight('bold')
        self.start_button.color = 'green'
        self.start_button.on_clicked(self.deal_with_button)


    def set_learning_rate(self,text):
        try:
            self.trainer.optimizer = \
            self.trainer.optimizer_type(self.trainer.model.parameters(), lr=float(text))
        except:
            print('Unable to set learning rate to',text)

        
    def deal_with_button(self, other_arg):
        label = self.start_button.label.get_text()
        if label == 'Start':
            self.start_button.label.set_text('Pause')
            self.start_button.color = 'red'
            self.call_trainer()
        elif label == 'Pause':
            self.start_button.label.set_text('Start')
            self.start_button.color = 'green'
            self.pause_training()
        else:
            print('How did this happen? Start button label is', label)


    @fire_and_forget       # this enables training to run in background
    def call_trainer(self):
        print('hello')
        self.trainer.pause = False
        self.trainer.train()

    def pause_training(self):
        self.trainer.pause = True

    def process_key(self, event): 
        if event.key == 't':
            print('training...')
            self.call_trainer()  # this is decorated with fire_and_Forget
        elif event.key == 'p':
            print('Paused.')
            self.pause_training()
        elif event.key == 'g':
            self.update_plot = True
        else:
            print(event.key,'??!')

    def process_button(self, event): 
        print("Button:", event.x, event.y, event.xdata, event.ydata, event.button) 

    def get_learning_rate(self):
        lrlist = []
        [lrlist.append(p['lr']) for p in self.trainer.optimizer.param_groups]
        return lrlist[0]

#
#class Label(object):
#    def __init__(self,ax,text):        
#        self.b = Button(ax,str(text))
#        self.b.on_clicked(self.clicked)
#    def change_to(self, text):
#        self.b.label.set_text(str(text))
#        plt.gcf().canvas.draw_idle()
#
#    def clicked(self, event):
#        pass
#
