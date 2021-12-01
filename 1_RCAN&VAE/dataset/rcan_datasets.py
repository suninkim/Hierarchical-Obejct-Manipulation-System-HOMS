import glob
import random
import os
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from .data_aug import *


class ImageDataset(Dataset):
    def __init__(self):
        self.data_path = 'rcan'
        self.img_path = os.path.join(self.data_path, 'ori')
        self.img_list = natsorted(os.listdir(self.img_path))
        self.can_path = os.path.join(self.data_path, 'can')
        self.seg_path = os.path.join(self.data_path, 'seg')
        self.dep_path = os.path.join(self.data_path, 'dep')


    def __getitem__(self, index):

        img_A = np.load('%s/%s'%(self.img_path, self.img_list[index]))
        img_B_can = np.load('%s/%s'%(self.can_path, self.img_list[index]))
        img_B_seg = np.load('%s/%s'%(self.seg_path, self.img_list[index]))
        img_B_dep = np.load('%s/%s'%(self.dep_path, self.img_list[index]))
        img_B_dep = np.reshape(img_B_dep,(240,240,1))

        # Data augmentation
        img_A = add_noise(img_A)
        eps1 = np.random.uniform()
        if eps1 < 0.5:
            img_A = add_circle(img_A)
        else:
            img_A = add_shadow(img_A)
        img_A, img_B_can, img_B_seg, img_B_dep = np.transpose(img_A,[2, 0, 1]), np.transpose(img_B_can,[2, 0, 1]), np.transpose(img_B_seg,[2, 0, 1]), np.transpose(img_B_dep,[2, 0, 1])
        img_A = torch.ByteTensor(img_A).float()/255.0
        img_B_can = torch.ByteTensor(img_B_can).float()/255.0
        img_B_seg = torch.ByteTensor(img_B_seg).float()/255.0
        img_B_dep = torch.ByteTensor(img_B_dep).float()/255.0

        return {"A": img_A, "B":img_B_can, "C":img_B_seg, "D":img_B_dep}

    def __len__(self):
        return len( self.img_list)
