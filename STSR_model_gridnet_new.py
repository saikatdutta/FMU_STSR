import torch
import cv2

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys

import argparse
import os
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import platform
from tqdm import tqdm

import sys

import os

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pylab as pl
import scipy.misc
import torch.optim as optim
from torch import log
import gc 
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import OrderedDict 


from flow_reversal import FlowReversal
from PWCNetNew import PWCNet
from arch import RSDN9_128 as RSDN
from gridnet import GridNet



def backwarp(img, flow):
    _, _, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

    gridX = torch.tensor(gridX, requires_grad=False,).cuda()
    gridY = torch.tensor(gridY, requires_grad=False,).cuda()
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    # range -1 to 1
    x = 2*(x/W - 0.5)
    y = 2*(y/H - 0.5)
    # stacking X and Y
    grid = torch.stack((x,y), dim=3)

    # Sample pixels using bilinear interpolation.
    imgOut = torch.nn.functional.grid_sample(img, grid)

    return imgOut


class SmallMaskNet(nn.Module):
    """A three-layer network for predicting mask"""
    def __init__(self, input, output):
        super(SmallMaskNet, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, output, 3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.conv3(x)
        return x

def qfe(F_1_0, F_1_2, timestamp_tensor):
    C1 = 0.5*(F_1_2 - F_1_0)
    C2 = 0.5*(F_1_2 + F_1_0)
    F_1_t = C1*timestamp_tensor + C2*timestamp_tensor**2

    return F_1_t        

class STSR_net(nn.Module):
    def __init__(self):
        super(STSR_net, self).__init__()

        self.flownet = PWCNet() 
        self.refinenet = GridNet(20, 8)
        self.masknet = SmallMaskNet(38, 1)
        self.vsrnet = RSDN(4)
        self.synthnet = GridNet(20,3)

    def forward(self,I0,I1,I2,I3,t=0.5):

        device = I0.get_device()

        fwarp = FlowReversal().cuda(device)

        F10 = self.flownet(I1,I0)
        F12 = self.flownet(I1,I2)

        F23 = self.flownet(I2,I3)
        F21 = self.flownet(I2,I1)

        

        B, C, H, W = I0.shape

        LR = torch.cat((I0,I1,I2,I3),dim=1)
        LR_S = F.interpolate(LR,scale_factor=0.5,mode='bilinear')
        LR_S = F.interpolate(LR_S,scale_factor=2,mode='bilinear')
        LR_D = LR - LR_S
        LR = LR.view(B,-1,C,H,W)
        LR_D = LR_D.view(B,-1,C,H,W)
        LR_S = LR_S.view(B,-1,C,H,W)

        pred, pred_d, pred_s = self.vsrnet(LR,LR_D,LR_S)

        I1_h = pred[:,1,:,:,:]
        I2_h = pred[:,2,:,:,:]

        # interpolation

        F1t = qfe(F10,F12, t)      
        F2t = qfe(F23,F21, 1-t)

        # Flow Reversal
        Ft1, norm1 = fwarp(F1t, F1t)
        Ft1 = -Ft1
        Ft2, norm2 = fwarp(F2t, F2t)
        Ft2 = -Ft2

        Ft1[norm1 > 0] = Ft1[norm1 > 0]/norm1[norm1>0].clone()
        Ft2[norm2 > 0] = Ft2[norm2 > 0]/norm2[norm2>0].clone()


        I1t = backwarp(I1, Ft1)
        I2t = backwarp(I2, Ft2)

        output, feature = self.refinenet(torch.cat([I1, I2, I1t, I2t, F12, F21, Ft1, Ft2], dim=1))

        # Adaptive filtering
        Ft1r = backwarp(Ft1, 10*torch.tanh(output[:, 4:6])) + output[:, :2]
        Ft2r = backwarp(Ft2, 10*torch.tanh(output[:, 6:8])) + output[:, 2:4]

        I1tf = backwarp(I1, Ft1r)
        I2tf = backwarp(I2, Ft2r)

        M = torch.sigmoid(self.masknet(torch.cat([I1tf, I2tf, feature], dim=1))).repeat(1, 3, 1, 1)

        It_p = ((1-t) * M * I1tf + t * (1 - M) * I2tf) / ((1-t) * M + t * (1-M))

        Ft1r_h = 4*F.interpolate(Ft1r, scale_factor = 4, mode='bilinear')
        Ft2r_h = 4*F.interpolate(Ft2r, scale_factor = 4, mode='bilinear')

        M_h = F.interpolate(M, scale_factor = 4, mode='bilinear')  

        I1tf_h = backwarp(I1_h, Ft1r_h)
        I2tf_h = backwarp(I2_h, Ft2r_h)

        It_p_h = ((1-t) * M_h * I1tf_h + t * (1 - M_h) * I2tf_h) / ((1-t) * M_h + t * (1-M_h))

        del_, _ = self.synthnet(torch.cat((I1_h, I2_h, I1tf_h, I2tf_h, It_p_h, M_h[:,0:1,:,:], Ft1r_h, Ft2r_h),dim=1))

        It_p_h_f = It_p_h + del_

        return pred, pred_d, pred_s, It_p_h_f, It_p 


if __name__ == "__main__":
    
    model = STSR_net().cuda(7)
    I0 = torch.randn(1,3,128,128).cuda(7)
    I1 = torch.randn(1,3,128,128).cuda(7)
    I2 = torch.randn(1,3,128,128).cuda(7)
    I3 = torch.randn(1,3,128,128).cuda(7)

    out1, out2, out3, out4 = model(I0, I1, I2, I3)

    print (out1.shape)
    print (out2.shape)
    print (out3.shape)
    print (out4.shape)    

