import os 
import pandas as pd 
import torch
import cv2
import time 
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

import os
# from natsort import natsorted
from skimage.measure import compare_psnr,compare_ssim
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

import pylab as pl
import scipy.misc
import torch.optim as optim
from torch import log
import gc 
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import OrderedDict 

def to_img(imtensor):
    im = np.uint8(imtensor.clamp(0,1).detach().cpu().squeeze().numpy().transpose(1,2,0)*255)

    return im 

in_dir = 'REDS/val/val_sharp_bicubic/X4'
gt_dir = 'REDS/val/val_sharp'

# in_dir = '../test/test_sharp_bicubic/X4'

out_dir = 'REDS_val_out/' 
os.makedirs(out_dir, exist_ok = True)

listfolder = os.listdir(in_dir)
listfolder.sort()

from FMU_gridnet import STSR_net

# to handle border sequences 
def clamp(x):
    return min(max(x,0),98)

device = 'cuda:0'

model = STSR_net().to(device)


model.load_state_dict(torch.load('checkpoints/gridnet/stsr-21-11279.pth',map_location=device))

print ('Model loaded!')


transform = transforms.Compose([ transforms.ToTensor()])
 

for folder in tqdm(listfolder): 

    # os.makedirs(out_dir + '/' + folder , exist_ok = True)


    for i in range(0,97,2):
        
        i0 = in_dir + '/' + folder + '/' + str(clamp(i-2)).zfill(8) + '.png'
        i1 = in_dir + '/' + folder + '/' + str(clamp(i)).zfill(8) + '.png'
        i2 = in_dir + '/' + folder + '/' + str(clamp(i+2)).zfill(8) + '.png'
        i3 = in_dir + '/' + folder + '/' + str(clamp(i+4)).zfill(8) + '.png'
        

        I0 = transform(cv2.imread(i0)).unsqueeze(0).to(device)
        I1 = transform(cv2.imread(i1)).unsqueeze(0).to(device)
        I2 = transform(cv2.imread(i2)).unsqueeze(0).to(device)
        I3 = transform(cv2.imread(i3)).unsqueeze(0).to(device)

        
        with torch.no_grad():
            
            pred , _, _ , Itph, _  = model(I0,I1,I2,I3)

        
        I1ph = pred[:,1,:,:,:]
        I2ph = pred[:,2,:,:,:]
        
        I1ph_im = to_img(I1ph)
        I2ph_im = to_img(I2ph)        
        Itph_im = to_img(Itph)

        cv2.imwrite(out_dir +'/'+ str(folder) + '_' + str(i).zfill(8)+ '.png', I1ph_im) 
        cv2.imwrite(out_dir +'/'+ str(folder) + '_' + str(i+1).zfill(8)+ '.png', Itph_im)
        

        if i==96:
            cv2.imwrite(out_dir +'/'+ str(folder) + '_' + str(i+2).zfill(8)+ '.png', I2ph_im)
            


print ("Done!")


