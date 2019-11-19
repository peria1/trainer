# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 06:16:10 2019

@author: Bill
"""

import matplotlib.pyplot as plt
import trainer as tr
from models import *
from problems import *
from trainer_utils import best_square, date_for_filename
from matplotlib.widgets import TextBox, Button, CheckButtons
import trainer_plots as tp
import asyncio
from matplotlib.backends.backend_pdf import PdfPages

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
        #
        # Controller window
        #
        self.fig, self.ax = plt.subplots(1,1)       
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)       
        self.fig.canvas.mpl_connect('key_press_event', self.process_key) 
        self.fig.canvas.mpl_connect('button_press_event', self.process_button) 
        self.fig.canvas.mpl_connect('close_event', self.close_it_down)
        self.ax.set_axis_off()
        self.fig.canvas.manager.window.raise_() # Trying to make windows visible right away
        
        self.trainer = tr.trainer(*args, **kwargs, viewer=self)
        self.trainer.pause = False
        self.update_plots = False

        #
        #  Available displays...feel free to add more!
        #
        self.displays = [self.Training_Display(name='loss history',  \
                                               nrows=2,ncols=1,\
                                               update=tp.basic_loss_plot), \
                         self.Training_Display(name='residuals',  \
                                               update=tp.residual_plot) ,\
                         self.Training_Display(name='weights',  \
                                               update=tp.weight_plot),\
                         self.Training_Display(name='dataflow',  \
                                               update=tp.dataflow_plot)]
                         
        if self.trainer.ytest.size() == self.trainer.xtest.size():
            print('adding examples...')
            self.displays.append(self.Training_Display(name='examples', active = False,\
                                               update=tp.example_plot))

#        # Button layout, for a column at the right-hand side of window. 
        text_box_left_edge = 0.75
        text_box_width = 0.15
        left_edge = 0.6
        width = 0.3
        height = 0.075
        text_color = 'white'
        down_from_top = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        
        plt.figure(self.fig.number)
        
        display_pick_ax = plt.axes([0.15, 0.125, width, 0.75])
        disp_names = [d.name for d in self.displays]
        actives = [d.active for d in self.displays]
        
        self.dispradio = CheckButtons(display_pick_ax,  disp_names, actives)
        def toggle_active_display(event):
            for i,n in enumerate(disp_names):
                if n == event:
                    if self.displays[i].active == True:
                        self.displays[i].deactivate()
                    else:
                        self.displays[i].activate()
#                    self.displays[i].active = not self.displays[i].active
        self.dispradio.on_clicked(toggle_active_display)

    
        ax_loss_box = plt.axes([text_box_left_edge, down_from_top[0], text_box_width, height]) # left, bottom, width, height
        self.loss_box = TextBox(ax_loss_box, 'Max Loss: ', \
                              initial=str(1e30))
        self.loss_box.on_submit(self.set_max_loss)

        ax_pval_box = plt.axes([text_box_left_edge, down_from_top[1], text_box_width, height]) # left, bottom, width, height
        self.pval_box = TextBox(ax_pval_box, 'Min p-value: ', \
                              initial=str(0.5))
        self.pval_box.on_submit(self.set_min_pval)
        
        axbox = plt.axes([text_box_left_edge, down_from_top[2], text_box_width, height]) 
        self.lr_box = TextBox(axbox, 'learning rate: ', \
                              initial=str(self.get_learning_rate()))
        self.lr_box.on_submit(self.set_learning_rate)
        
        axbutton = plt.axes([left_edge, down_from_top[3], width, height])
        self.start_button = Button(axbutton, 'Start')
        self.start_button.label.set_color(text_color)
        self.start_button.label.set_fontweight('bold')
        self.start_button.color = 'green'  # callback will toggle the color
        self.start_button.on_clicked(self.deal_with_start_button)
        
        dispbutton = plt.axes([left_edge, down_from_top[4], width, height])
        self.disp_button = Button(dispbutton, 'Update Displays')
        self.disp_button.label.set_color(text_color)
        self.disp_button.label.set_fontweight('bold')
        self.disp_button.color = 'black'
        self.disp_button.on_clicked(self.deal_with_update_button)

        reportbutton = plt.axes([left_edge, down_from_top[5], width, height])
        self.report_button = Button(reportbutton, 'Make Report')
        self.report_button.label.set_color(text_color)
        self.report_button.label.set_fontweight('bold')
        self.report_button.color = 'black'
        self.report_button.on_clicked(self.generate_report)

        clearbutton = plt.axes([left_edge, down_from_top[6], width, height])
        self.clear_button = Button(clearbutton, 'Clear History')
        self.clear_button.label.set_color(text_color)
        self.clear_button.label.set_fontweight('bold')
        self.clear_button.color = 'black'
        self.clear_button.on_clicked(self.clear_history)
        
    def clear_history(self,event):
        self.trainer.zap_history()
    
    def close_it_down(self,event):
        for d in self.displays:
            if d.active:
                plt.close(d.fig)

    def set_update_flag(self, flag=True):
        if flag is True:
            self.update_plots = True
        else:
            self.update_plots = False

    def update_displays(self):
        if self.update_plots:
            active_displays = (d for d in self.displays if d.active)
            for d in active_displays:
                try:
                    for a in d.ax.flatten():
                        a.clear()
                except (TypeError, AttributeError):
                    d.ax.clear()             
                d.update(self,d)   # this is calling the custom plotting function
                d.first = False
                d.fig.canvas.draw()
                d.fig.canvas.flush_events()

    def add_display(self, event):
        pass

    def set_max_loss(self, text):
        try:
            self.trainer.max_loss = float(text)
        except:
            print('Unable to set max loss to',text)

    def set_min_pval(self, text):
        try:
            self.trainer.min_pval = float(text)
        except:
            print('Unable to set min p-value to',text)
                
    def set_learning_rate(self,text):
        try:
            self.trainer.optimizer = \
            self.trainer.optimizer_type(self.trainer.model.parameters(), lr=float(text))
        except:
            print('Unable to set learning rate to',text)

        
    def deal_with_start_button(self, other_arg):  
        # Start or pause training, and toggle the button to the other state. 
        label = self.start_button.label.get_text()
        if label in ['Start','Resume']:
            self.call_trainer()
            # Grt ready to Pause next time button is pushed. 
            self.start_button.label.set_text('Pause')
            self.start_button.color = 'red'
        elif label == 'Pause':
            self.pause_training()
            # Get ready to Start next time button is pushed. 
            self.start_button.label.set_text('Resume')
            self.start_button.color = 'green'
        else:
            print('How did this happen? Start button label is', label)

    def deal_with_update_button(self, other_arg):  
        # Start or pause training, and toggle the button to the other state. 
        label = self.disp_button.label.get_text()
        if label == 'Update Displays':
            self.set_update_flag(flag=True)
            # Get ready to Pause next time button is pushed. 
            self.disp_button.label.set_text('Pause Displays')
            self.disp_button.color = 'red'
        elif label == 'Pause Displays':
            self.set_update_flag(flag=False)
            self.disp_button.label.set_text('Update Displays')
            self.disp_button.color = 'green'
        else:
            print('How did this happen? Update button label is', label)

    def arm_start_button(self):
            self.start_button.label.set_text('Resume')
            self.start_button.color = 'green'
        

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
            pass

    def process_button(self, event): 
        pass
#        print("Button:", event.x, event.y, event.xdata, event.ydata, event.button) 

    def get_learning_rate(self):
        lrlist = []
        [lrlist.append(p['lr']) for p in self.trainer.optimizer.param_groups]
        return lrlist[0]
    
    def generate_report(self, event):
        model_name = self.trainer.get_model_name()
        problem_name = self.trainer.get_problem_name()
        date = date_for_filename()
        report_file = model_name + '_' +  problem_name + '_' + date + '.pdf'
        print('Generating report:',report_file)
        with PdfPages(report_file) as pdf:
            pdf.savefig(self.fig)   # First page is the controller window,
            for d in (d for d in self.displays if d.active):  #  then all the displays in order. 
                if 'layer_names' in dir(d):
                    curr_layer = d.layer_to_show
                    for i,ln in enumerate(d.layer_names):
                        d.layer_to_show = i
                        d.update(self, d)
                        pdf.savefig(d.fig)
                    d.layer_to_show = curr_layer
                    d.update(self, d)
                else:
                    pdf.savefig(d.fig)
    
    class Training_Display():
        def __init__(self, name='no name', nrows=1, ncols=1, nplots=None, \
                     update=None, active = None):
            self.first = True

            if update: 
                self.update = update
            
            if active:
                self.activate()
            else:
                self.active = False

            self.name = name
            self.nrows = nrows
            self.ncols = ncols
            if nplots and (nplots != (nrows*ncols)):
                nrows,ncols = best_square(nplots)
                
            if not update:
                print('You need to define an update function in Training_Display objects.')
                print('Look in trainer_plots.py to see examples of update functions.')

        def update(self):
            pass

        def activate(self):
            self.active = True
            self.fig, self.ax = plt.subplots(self.nrows, self.ncols)
            self.first = True
        
        def deactivate(self):
            self.active = False
            plt.close(self.fig)
        
