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
        
        self.trainer = tr.trainer(*args, **kwargs)
        self.trainer.pause = False
        
        self.lr_box = TextBox(self.ax, 'learning rate', \
                              initial=str(self.get_learning_rate()))
        self.lr_box.on_submit(self.say_hello)

    def say_hello(self,text):
#        try:
        print('setting lr to',text)
        [print(float(t)) for t in text]
        self.set_learning_rate(float(text))
#        except:
#            print('Unable to set learning rate to',text)

        
    @fire_and_forget
    def call_trainer(self):
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
        elif event.key == 'l':
            self.pause_training()
            lrlist = self.get_learning_rate()
            print('Number of rates to set is',str(len(lrlist)))
            newlr = []
            for l in lrlist:
                print('Learning rate is',l)        
                nlr = getnum('Please enter new learning rate:',l)
                newlr.append(nlr)
            
            if len(newlr)==len(lrlist):
                self.set_learning_rate(newlr)
            
            self.call_trainer()
        else:
            print('I dunno from',event.key)

    def process_button(self, event): 
        print("Button:", event.x, event.y, event.xdata, event.ydata, event.button) 

    def get_learning_rate(self):
        lrlist = []
        [lrlist.append(p['lr']) for p in self.trainer.optimizer.param_groups]
        return lrlist[0]

    def set_learning_rate(self, newlr):
        for i,p in enumerate(self.trainer.optimizer.param_groups):
            p['lr']=newlr
 

class Label(object):
    def __init__(self,ax,text):        
        self.b = Button(ax,str(text))
        self.b.on_clicked(self.clicked)
    def change_to(self, text):
        self.b.label.set_text(str(text))
        plt.gcf().canvas.draw_idle()

    def clicked(self, event):
        pass

