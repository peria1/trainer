# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 06:16:10 2019

@author: Bill
"""

import matplotlib.pyplot as plt
import trainer as tr
from models import *
from trainer_utils import date_for_filename
from matplotlib.widgets import TextBox, Button
import trainer_plots as tp
import asyncio
#
# The following code can be used to decorate a function; such a decorated function 
#   will then run in the background until stopped. I use it to start the training
#   loop in trainer. 
#
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
        
#        self.displays = [loss_graph]
        
        self.trainer = tr.trainer(*args, **kwargs, viewer=self)
        self.trainer.pause = False
        self.update_plots = False

        self.displays = [self.Training_Display(name='loss history',\
                                               nrows=2,ncols=1,\
                                               update=tp.basic_loss_plot),\
                         self.Training_Display(name='residuals',\
                                               update=tp.residual_plot)]

        # Button layout, for a column at the right-hand side of window. 
        left_edge = 0.6
        width = 0.3
        height = 0.075
        text_color = 'white'
        
        plt.figure(self.fig.number)
        axbox = plt.axes([left_edge, 0.8, width, height]) # left, bottom, width, height
        self.lr_box = TextBox(axbox, 'learning rate', \
                              initial=str(self.get_learning_rate()))
        self.lr_box.on_submit(self.set_learning_rate)
        
        axbutton = plt.axes([left_edge, 0.7, width, height])
        self.start_button = Button(axbutton, 'Start')
        self.start_button.label.set_color(text_color)
        self.start_button.label.set_fontweight('bold')
        self.start_button.color = 'green'  # callback will toggle the color
        self.start_button.on_clicked(self.deal_with_start_button)
        
        dispbutton = plt.axes([left_edge, 0.6, width, height])
        self.disp_button = Button(dispbutton, 'Refresh')
        self.disp_button.label.set_color(text_color)
        self.disp_button.label.set_fontweight('bold')
        self.disp_button.color = 'black'
        self.disp_button.on_clicked(self.handle_update_button)

        newbutton = plt.axes([left_edge, 0.5, width, height])
        self.new_button = Button(newbutton, 'New Display')
        self.new_button.label.set_color(text_color)
        self.new_button.label.set_fontweight('bold')
        self.new_button.color = 'black'
        self.new_button.on_clicked(self.add_display)

        clearbutton = plt.axes([left_edge, 0.4, width, height])
        self.clear_button = Button(clearbutton, 'Clear History')
        self.clear_button.label.set_color(text_color)
        self.clear_button.label.set_fontweight('bold')
        self.clear_button.color = 'black'
        self.clear_button.on_clicked(self.clear_history)
        
    def clear_history(self,event):
        self.trainer.zap_history()
    
    def handle_update_button(self, event):
        self.set_update_flag()
        
    def set_update_flag(self):
        print('setting update flag to True...')
        self.update_plots = True

    def update_displays(self):
        if self.update_plots:
            self.update_plots = False
            for d in self.displays:
                try:
                    for a in d.ax:
                        a.clear()
                except TypeError:
                    d.ax.clear()             
                d.update(self,d)   # this is calling the custom plotting function
                d.fig.canvas.draw()
                d.fig.canvas.flush_events()

    def add_display(self, event):
        pass

    def set_learning_rate(self,text):
        try:
            self.trainer.optimizer = \
            self.trainer.optimizer_type(self.trainer.model.parameters(), lr=float(text))
        except:
            print('Unable to set learning rate to',text)

        
    def deal_with_start_button(self, other_arg):  
        # Start or pause training, and toggle the button to the other state. 
        label = self.start_button.label.get_text()
        if label == 'Start':
            self.call_trainer()
            # Grt ready to Pause next time button is pushed. 
            self.start_button.label.set_text('Pause')
            self.start_button.color = 'red'
        elif label == 'Pause':
            self.pause_training()
            # Get ready to Start next time button is pushed. 
            self.start_button.label.set_text('Start')
            self.start_button.color = 'green'
        else:
            print('How did this happen? Start button label is', label)


    @fire_and_forget       # this enables training to run in background
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

            
    class Training_Display():
        def __init__(self, name='no name', nrows=1, ncols=1,update=None):
            self.fig, self.ax = plt.subplots(nrows,ncols)
            if update:
                self.update = update
                
        def update(self):
            print('You need to define an update function in Training_Display objects.')
