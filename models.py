# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:32:52 2019

@author: Bill
"""

# -*- coding: utf-8 -*-billUti
"""
Created on Sun Oct  6 07:32:17 2019

@author: Bill


"""

import torch
import torch.utils.data
from torch import nn
from torchvision.models.vgg import VGG
import numpy as np

from criteria import *



class one_linear_layer(nn.Module):
    def __init__(self, problem):
        super().__init__()

        inp, target = problem.get_input_and_target()

        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        
        self.npts = npts
        self.nbatch = nbatch    
       
        self.L1 = nn.Linear(self.npts,target.size()[1])
       
    def forward(self,x):
        return self.L1(x)

class one_linear_layer_to_n(nn.Module):
    def __init__(self, problem):
        super().__init__()

        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        
        self.L1 = nn.Linear(self.npts,self.npts)
       
    def forward(self,x):
        return self.L1(x)

     
class bisect_to_power_of_two(nn.Module):
    def __init__(self, problem):   
        super().__init__()

        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        nout = target.size()[1]
        nchk = npts*nout
        
        try:
            assert((nchk != 0) and (nchk & (nchk-1) == 0))
        except AssertionError:
            print('Length of both input and output examples must be a power of two.')
        
        self.npts = npts
        self.nbatch = nbatch
        self.nout = nout
        
        n = npts*2
        self.layer_list = nn.ModuleList([nn.Linear(npts,n)])
        while n > nout:
            self.layer_list.append(nn.Linear(n,n//2))
            n//=2
        self.leaky = nn.LeakyReLU()

    def forward(self, xy):
        dataflow = xy
        for L in self.layer_list:
            dataflow = self.leaky(L(dataflow))
        
        return dataflow



class n_double_n_act(nn.Module): # moved to n_double_n
    def __init__(self, problem): 
        super().__init__()

        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
                
        self.npts = npts
        self.nbatch = nbatch
        
        self.L1 = nn.Linear(npts, 2*npts)
        self.L2 = nn.Linear(2*npts, 2*npts)
        self.L3 = nn.Linear(2*npts, 2*npts)
        self.L4 = nn.Linear(2*npts, 2*npts)
        self.L5 = nn.Linear(2*npts, 2*npts)
        self.L6 = nn.Linear(2*npts, 2*npts)
        self.L7 = nn.Linear(2*npts, 2*npts)
        self.Llast = nn.Linear(2*npts, npts)
        self.activation1 = nn.LeakyReLU()
        self.activation2 = nn.Tanh()


    def forward(self, x):
        dataflow = self.activation1(self.L1(x))
        dataflow = self.activation1(self.L2(dataflow))
        dataflow = self.activation1(self.L3(dataflow))
        dataflow = self.activation1(self.L4(dataflow))
        dataflow = self.activation2(self.L5(dataflow))
        dataflow = self.activation2(self.L6(dataflow))
        dataflow = self.activation2(self.L7(dataflow))
        dataflow = self.Llast(dataflow)
        
        return dataflow



class n_double_n(nn.Module): # moved to n_double_n
    def __init__(self, problem): 
        super().__init__()

        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        
        self.npts = npts
        self.nbatch = nbatch
        
        self.L1 = nn.Linear(npts, 2*npts)
        self.L2 = nn.Linear(2*npts, 2*npts)
        self.L3 = nn.Linear(2*npts, 2*npts)
        self.L4 = nn.Linear(2*npts, 2*npts)
        self.L5 = nn.Linear(2*npts, 2*npts)
        self.L6 = nn.Linear(2*npts, 2*npts)
        self.L7 = nn.Linear(2*npts, 2*npts)
        self.Llast = nn.Linear(2*npts, npts)
        self.leaky = nn.LeakyReLU()


    def forward(self, x):
        dataflow = self.leaky(self.L1(x))
        dataflow = self.leaky(self.L2(dataflow))
        dataflow = self.leaky(self.L3(dataflow))
        dataflow = self.leaky(self.L4(dataflow))
        dataflow = self.leaky(self.L5(dataflow))
        dataflow = self.leaky(self.L6(dataflow))
        dataflow = self.leaky(self.L7(dataflow))
        dataflow = self.leaky(self.Llast(dataflow))
        
        return dataflow

class n_double_one(nn.Module):  # moved to n_double_one
    def __init__(self, problem):  # trying to see if machine can tell that y is the sum over x 
        super().__init__()

        self.custom_loss = nn.L1Loss()
        
        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        
        self.npts = npts
        self.nbatch = nbatch
        width_factor = 2
#        self.bn1 = nn.BatchNorm1d(npts)
        self.L1 = nn.Linear(npts, width_factor*npts)
        self.L2 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L3 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L4 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L5 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L6 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L7 = nn.Linear(width_factor*npts, width_factor*npts)
        self.Llast = nn.Linear(width_factor*npts, npts)
        
        self.weight_vector = nn.Linear(npts, 1) # npts is the size of each 1D example
        
        self.leaky = nn.LeakyReLU()

        
    def forward(self, x):
        dataflow = x
        dataflow = self.leaky(self.L1(dataflow))
        dataflow = self.leaky(self.L2(dataflow))
        dataflow = self.leaky(self.L3(dataflow))
        dataflow = self.leaky(self.L4(dataflow))
        dataflow = self.leaky(self.L5(dataflow))
        dataflow = self.leaky(self.L6(dataflow))
        dataflow = self.leaky(self.L7(dataflow))
        dataflow = self.leaky(self.Llast(dataflow))
        result = self.weight_vector(dataflow)
        return result



class n_double_nout(nn.Module):  # moved to n_double_one
    def __init__(self, problem):  # trying to see if machine can tell that y is the sum over x 
        super().__init__()


        self.custom_loss = nn.L1Loss()
        
        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        nout = target.size()[1]
        
        
        self.npts = npts
        self.nbatch = nbatch
        width_factor = 2
#        self.bn1 = nn.BatchNorm1d(npts)
        self.L1 = nn.Linear(npts, width_factor*npts)
        self.L2 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L3 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L4 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L5 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L6 = nn.Linear(width_factor*npts, width_factor*npts)
        self.L7 = nn.Linear(width_factor*npts, width_factor*npts)
        self.Llast = nn.Linear(width_factor*npts, npts)
        
        self.weight_vector = nn.Linear(npts, nout) # npts is the size of each 1D example
        
        self.leaky = nn.LeakyReLU()

        
    def forward(self, x):
        dataflow = x
        dataflow = self.leaky(self.L1(dataflow))
        dataflow = self.leaky(self.L2(dataflow))
        dataflow = self.leaky(self.L3(dataflow))
        dataflow = self.leaky(self.L4(dataflow))
        dataflow = self.leaky(self.L5(dataflow))
        dataflow = self.leaky(self.L6(dataflow))
        dataflow = self.leaky(self.L7(dataflow))
        dataflow = self.leaky(self.Llast(dataflow))
        result = self.weight_vector(dataflow)
        return result



class n_double_one_tanh(nn.Module):    # moved to n_double_one_tanh
    def __init__(self, problem):  # trying to see if machine can tell that y is the sum over x 
        super().__init__()

        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        
        self.npts = npts
        self.nbatch = nbatch
        
        self.L1 = nn.Linear(npts, 2*npts)
        self.L2 = nn.Linear(2*npts, 2*npts)
        self.L3 = nn.Linear(2*npts, 2*npts)
        self.L4 = nn.Linear(2*npts, 2*npts)
        self.L5 = nn.Linear(2*npts, 2*npts)
        self.L6 = nn.Linear(2*npts, 2*npts)
        self.L7 = nn.Linear(2*npts, 2*npts)
        self.Llast = nn.Linear(2*npts, npts)
        
        self.weight_vector = nn.Linear(npts, 1) # npts is the size of each 1D example

    def forward(self, x):
        dataflow = torch.relu(self.L1(x))
        dataflow = torch.relu(self.L2(dataflow))
        dataflow = torch.relu(self.L3(dataflow))
        dataflow = torch.relu(self.L4(dataflow))
        dataflow = torch.relu(self.L5(dataflow))
        dataflow = torch.relu(self.L6(dataflow))
        dataflow = torch.relu(self.L7(dataflow))        
        dataflow = torch.relu(self.Llast(dataflow))
        dataflow = self.weight_vector(dataflow)
        result = torch.tanh(dataflow)
        return result



class vectorVAE(nn.Module):   # moved to vectorVAE
    def __init__(self, problem, dim=2):  # trying to see if machine can tell that y is the sum over x 
        super().__init__()
        
        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
#        nout = target.size()[1]

        self.npts = npts
        self.nbatch = nbatch        
        self.z_dimension = dim
#        self.register_backward_hook(grad_hook)
        self.fc1 = nn.Linear(npts, npts) # stacked MNIST to 400
        self.fc21 = nn.Linear(npts, self.z_dimension) # two hidden low D
        self.fc22 = nn.Linear(npts, self.z_dimension) # layers, same size
        self.fc3 = nn.Linear(self.z_dimension, npts)  
        self.fc4 = nn.Linear(npts, npts)

    def encode(self, x): 
        from torch.nn import functional as F
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # <- Stochasticity!!!
        # How can the previous line allow back propagation? The question is, does the 
        #   loss function depend on eps? For starters, it returns a normal 
        #   with the same dimensions as std, rather than one that has the variances
        #   implied by std. That's why we scale by std, below, when returning. 
        #   Because we are only using the dimensions of std to get eps, does that mean
        #   that the loss function is independent of eps? I am missing something. 
        #
        return mu + eps*std

    def decode(self, z):
        from torch.nn import functional as F
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
#        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z) #, mu, logvar



#--------------------
class TrainerRNN(nn.Module):
    def __init__(self, problem, npts=None, nbatch=None, nout=None):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==2)
        nbatch, npts = inp.size()
        nout = target.size()[1]
        
        self.input_size = npts
        self.seq_len = nbatch
        
        self.hidden_dim = 2*self.input_size  
        self.n_layers = 10
        self.output_size = nout
        
        self.rnn = nn.GRU(self.input_size, self.hidden_dim, self.n_layers)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

        self.forward_count = 0
        
    def forward(self, x):
        
        batch_size = 1

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x.unsqueeze(1), hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
#        out = out.view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        hidden = hidden.to(self.device)
        return hidden
         
#--------------------
#            vgg_model = VGGNet(pretrained = pretrained, requires_grad=True,GPU = GPU)

 
class VGGNet(VGG):
    def __init__(self, problem, in_channels=None, num_classes = None, nbatch = None, \
                 model='vgg16', requires_grad=True, \
                 show_params=False, GPU = False):
        from VGGdefs import ranges, cfg, make_layers


        if not num_classes:
            num_classes = 1000
        if not nbatch:
            nbatch = problem.nbatch

        inp, target = problem.get_input_and_target()
        assert(len(inp.size())==4)
        nbatch, in_channels, nx, ny = inp.size()
        assert(np.prod(target.size())==nbatch)

        if not in_channels:
            in_channels=3
        
    
        super().__init__(make_layers(cfg[model],in_channels),\
              num_classes=num_classes)
        self.ranges = ranges[model]

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.device)
        if GPU:
            for name, param in self.named_parameters():
                param.cuda()
                
        self.custom_loss = self.vggloss
        
        self.criterion = torch.nn.CrossEntropyLoss()

        
    def vggloss(self,pred,target):
        
        return self.criterion(pred, target.to(torch.long))
    


    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x
        
        score = self.classifier(torch.flatten(self.avgpool(output['x5']),1))

        return torch.softmax(score,1)


class YOLAB(nn.Module):
    
    def __init__(self, problem):
        super().__init__()
        
        from utils.augmentations import FastBaseTransform
        self.FastBaseTransform = FastBaseTransform
        
        import cv2
        self.cv2 = cv2
        
        import matplotlib.pyplot as plt
        self.plt = plt
        
        from layers.output_utils import postprocess, undo_image_transformation
        self.postprocess = postprocess
        self.undo_image_transformation = undo_image_transformation
        
        from utils import timer
        self.timer = timer

        import sys
        syspathsave = None
        if not 'yolact' in sys.path[1]:
            import copy
            syspathsave = copy.copy(sys.path)
            sys.path.insert(1, '../yolact/')
            
        from yolact import Yolact
        from train import MultiBoxLoss
        import data as D  
        self.D = D
        
        from collections import defaultdict
        self.color_cache = defaultdict(lambda: {})

        
        net = Yolact()
        net.train()
        net.init_weights(backbone_path='../yolact/weights/' + D.cfg.backbone.path)

        criterion = MultiBoxLoss(num_classes=D.cfg.num_classes,
                                 pos_threshold=D.cfg.positive_iou_threshold,
                                 neg_threshold=D.cfg.negative_iou_threshold,
                                 negpos_ratio=D.cfg.ohem_negpos_ratio)

        self.net = net
        self.criterion = criterion

        if syspathsave:
            sys.path = syspathsave

    def forward(self, images): 
        predsT = self.net(images)
        return predsT
    
    def custom_loss(self, predsT, target):        
        targets, masks, num_crowds = target
        self.net.train()
        losses = self.criterion(self.net, predsT, targets[0], masks[0], num_crowds[0])
        loss = sum([losses[k] for k in losses])
        return loss

    def local_evalimage(self, net, path:str):
        frame = torch.from_numpy(self.cv2.imread(path)).cuda().float()
        print('frame size is', frame.size())
        batch = self.FastBaseTransform()(frame.unsqueeze(0))
        print('Batch size is',batch.size())
        net.eval()
        with torch.no_grad():
            preds = net(batch)
        net.train()
    
        img_numpy = self.local_prep_display(preds, frame, None, None, undo_transform=False)
        
        img_numpy = img_numpy[:, :, (2, 1, 0)]
    
        self.plt.imshow(img_numpy)
        self.plt.title(path)
        self.plt.show()
    


    def local_prep_display(self,dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        
        I don't have args available, so I need to create it and fill it with the defaults. 
        
        
        """
    #    print('local_prep_display, type(dets_out) is',type(dets_out))
        
        D = self.D  #  D contains everything from config.py
        color_cache = self.color_cache
        
        top_k = 5    # May want to adjust this. If there are 6 objects in view, what happens? 
        score_threshold = 0.0
        display_masks = True
        display_text = True
        display_bboxes = True
        display_scores = True
        display_fps = False
    
        if undo_transform:
            img_numpy = self.undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape
        
        with self.timer.env('Postprocess'):
            save =D.cfg.rescore_bbox
            D.cfg.rescore_bbox = True
            t = self.postprocess(dets_out, w, h, visualize_lincomb = False,
                                            crop_masks        = True,
                                            score_threshold   = score_threshold)
    
            D.cfg.rescore_bbox = save
    
        with self.timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:top_k]
            
            if D.cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().detach().numpy() for x in t[:3]]
    
        num_dets_to_consider = min(top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < score_threshold:
                num_dets_to_consider = j
                break
    
        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
#            global color_cache
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(D.COLORS)
            
            if on_gpu is not None and color_idx in color_cache[on_gpu]:
                return color_cache[on_gpu][color_idx]
            else:
                color = D.COLORS[color_idx]
                if not undo_transform:
                    # The image might come in as RGB or BRG, depending
                    color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    color_cache[on_gpu][color_idx] = color
                return color
    
        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        if display_masks and D.cfg.eval_mask_branch and num_dets_to_consider > 0:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]
            
            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
    
            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1
            
            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)
    
            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        
        if display_fps:
                # Draw the box for the fps on the GPU
            font_face = self.cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1
    
            text_w, text_h = self.cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]
    
            img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha
    
    
        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
        if display_fps:
            # Draw the text on the CPU
            text_pt = (4, text_h + 2)
            text_color = [255, 255, 255]
    
            self.cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, self.cv2.LINE_AA)
        
        if num_dets_to_consider == 0:
            return img_numpy
    
        if display_text or display_bboxes:
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]
    
                if display_bboxes:
                    self.cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
    
                if display_text:
                    _class = D.cfg.dataset.class_names[classes[j]]
                    text_str = '%s: %.2f' % (_class, score) if display_scores else _class
    
                    font_face = self.cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1
    
                    text_w, text_h = self.cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
    
                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]
    
                    self.cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    self.cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, self.cv2.LINE_AA)
                
        
        return img_numpy




class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super().__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(1)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
    
