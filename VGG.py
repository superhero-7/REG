import torch
import torch.nn as nn
import numpy as np
from ClassifierHelper import Classifier

class VGG(Classifier):
    """ VGG (or Alexnet) with optional modifications including
        - different output dimension (cfg.IMG_NET.N_LABELS)
        - frozen weights on layers in cfg.IMG_NET.FIX_WEIGHTS
        - removed last fully connected layer (cfg.IMG_NET.IGNORE_CLASSIFICATION = True)
    """

    def __init__(self, cfg, vgg, loss_function):
        super(VGG, self).__init__(cfg, loss_function)
        self.VGG = vgg

        #Change output dimension
        if cfg.IMG_NET_N_LABELS != 1000:
            vgg.classifier._modules['6'] = nn.Linear(4096, cfg.IMG_NET_N_LABELS)

        # Freezes weights for all layers in FIX_WEIGHTS
        if len(cfg.IMG_NET_FIX_WEIGHTS) > 0:
            self.freeze(cfg.IMG_NET_FIX_WEIGHTS)

        self.to(self.device)

    def forward(self, x):
        if self.use_cuda:
            x = x.cuda()

        return self.VGG(x)

    # VGG and Alexnet have similar enough architectures that the same method can be used to freeze layers
    # Remove gradients from network layers to freeze pretrained network
    def freeze(self, fix_weights):
        print("Freeze image network weights")
        child_counter = 0
        for child in self.VGG.modules():
            if child_counter in fix_weights:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1

class ResNet(Classifier):

    def __init__(self, cfg, resnet, loss_function):
        super(ResNet, self).__init__(cfg, loss_function)
        self.ResNet = resnet

        self.freeze()

        self.to(self.device)

    def forward(self, x):
        if self.use_cuda:
            x = x.cuda()

        return self.ResNet(x)

    # Remove gradients from network layers to freeze pretrained network
    def freeze(self):

        for k in self.ResNet.parameters():
            k.requires_grad = False