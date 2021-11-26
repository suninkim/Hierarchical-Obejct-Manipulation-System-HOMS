import sys
#sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import os
import cv2
import random
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset


class OfflineDataset(Dataset):

    def __init__(self, img_dim):

        self.img_list = []
        self._p = 0
        self.load_data(num_rollout=8)

        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim

    def __getitem__(self, idx):
        image = self.img_list[idx]
        image = np.transpose(image,[1,2,0])

        h, w, c = image.shape
        if h > w:
            top_h = int((h - w) / 2)
            image = image[top_h:top_h + w]
        else:
            left_w = int((w - h) / 2)
            image = image[:, left_w:left_w + h]

        h, w = random.randint(0,h-self.img_dim[0]-1), random.randint(0,w-self.img_dim[1]-1)
        image = image[h:h+self.img_dim[0],w:w+self.img_dim[1],:]

        image = image / 255.

        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

    def load_data(self, num_rollout):

        for task in range(9):
            data_path_list = []
            for roll in range(1, num_rollout):
                data_path_list.append(f"../rollout/task{task}/rollout{roll}.npy")
            self.add_data_to_buffer(data_path_list, last_stage=True)
            print(f"Task{task} data is loaded")
        data_path_list = []

    def get_data(self, data_path_list):
        buffer_size = 0
        data_list = []
        for path in data_path_list:
            data = np.load(path, allow_pickle=True)
            data_list.append(data)
            buffer_size += self.get_buffer_size(data)
            data = []

        return data_list, buffer_size

    def get_buffer_size(self, data):
        num_transitions = 0
        for i in range(len(data)):
            for j in range(len(data[i]['observations'])):
                num_transitions += 1
        return num_transitions

    def add_data_to_buffer(self, data_path_list, last_stage=False):
        data_list, buffer_size = self.get_data(data_path_list)

        for data in data_list:
            for j in range(len(data)):

                for i in range(len(data[j]['observations'])):
                    state = data[j]['observations'][i]['image'][:3,:,:]

                    self.img_list += [None]                       

                    self.append(state)

        print(f"p : {self._p}")

        data_list=[]

    def append(self, state):
        self.img_list[self._p] = state

        self._p += 1

    def __len__(self):
        return len(self.img_list)
