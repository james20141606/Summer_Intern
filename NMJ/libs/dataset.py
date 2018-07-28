from __future__ import print_function, division
import os, sys
import numpy as np
import pickle, h5py, time, argparse, itertools, datetime

import torch
import torch.nn as nn
import torch.utils.data

# -- 0. utils --
def countVolume(data_sz, vol_sz, stride):
    return 1 + np.ceil((data_sz - vol_sz) / stride.astype(float)).astype(int)

def cropVolume(data, sz, st=[0,0,0]): # C*D*W*H
    return data[st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], \
            st[2]:st[2]+sz[2]]

# -- 1. dataset -- 
# dataset class for synaptic cleft inputs
class SynapseDataset(torch.utils.data.Dataset):
    # assume for test, no warping [hassle to warp it back..]
    def __init__(self,
                 volume, label=None,
                 vol_input_size = (8,64,64),
                 vol_label_size = None,
                 sample_stride = (1,1,1),
                 data_aug = None,
                 mode = 'train'):
        
        self.mode = mode

        # data format
        self.input = volume
        self.label = label
        self.data_aug = data_aug # data augmentation
        
        # samples, channels, depths, rows, cols
        self.input_size = [np.array(x.shape) for x in self.input] # volume size, could be multi-volume input
        self.vol_input_size = np.array(vol_input_size) # model input size
        self.vol_label_size = np.array(vol_label_size) # model label size

        # compute number of samples for each dataset (multi-volume input)
        self.sample_stride = np.array(sample_stride, dtype=np.float32)
        self.sample_size = [ countVolume(self.input_size[x], self.vol_input_size, np.array(self.sample_stride)) \
                            for x in range(len(self.input_size))]
        #total number of possible inputs for each volume
        self.sample_num = np.array([np.prod(x) for x in self.sample_size])
        self.sample_num_a = np.sum(self.sample_num)
        self.sample_num_c = np.cumsum([0] + list(self.sample_num))
        #print(self.sample_num_c)
        assert self.sample_num_c[-1] == self.sample_num_a

        # for test
        self.sample_size_vol = [np.array([np.prod(x[1:3]),x[2]]) for x in self.sample_size]

    def __getitem__(self, index):

        if self.mode == 'train':
            # 1. get volume size
            vol_size = self.vol_input_size
            # if self.data_aug is not None: # augmentation
            #     self.data_aug.getParam() # get augmentation parameter
            #     vol_size = self.data_aug.aug_warp[0]
            # train: random sample based on vol_size
            # test: sample based on index
            pos = self.getPos(vol_size, index)

            # 2. get input volume
            out_input = cropVolume(self.input[pos[0]], vol_size, pos[1:])
            out_label = cropVolume(self.label[pos[0]], vol_size, pos[1:])

            # 3. augmentation
            # if self.data_aug is not None: # augmentation
            #     out_input, out_label = self.data_aug.augment(out_input, out_label)

            # 4. class weight
            # add weight to classes to handle data imbalance
            # match input tensor shape
            out_input = torch.Tensor(out_input)
            out_label = torch.Tensor(out_label)
            weight_factor = out_label.float().sum() / torch.prod(torch.tensor(out_label.size()).float())
            weight_factor = torch.clamp(weight_factor, min=0.0001)
            # the fraction of synaptic cleft pixels, can be 0
            weight = out_label*(1-weight_factor)/weight_factor + (1-out_label)

            # include the channel dimension
            out_input = out_input.unsqueeze(0)
            out_label = out_label.unsqueeze(0)
            weight = weight.unsqueeze(0)

            return out_input, out_label, weight, weight_factor

        elif self.mode == 'test':
            # 1. get volume size
            vol_size = self.vol_input_size  
            # test mode
            pos = self.getPosTest(index)
            out_input = cropVolume(self.input[pos[0]], vol_size, pos[1:])
            out_input = torch.Tensor(out_input)
            out_input = out_input.unsqueeze(0)

            return pos, out_input  

    def __len__(self): # number of possible position
        return self.sample_num_a
    
    def getPosDataset(self, index):
        return np.argmax(index<self.sample_num_c)-1 # which dataset

    def getPos(self, vol_size, index):
        pos = [0,0,0,0]
        seed = np.random.RandomState(index)
        did = self.getPosDataset(seed.randint(self.sample_num_a))
        pos[0] = did
        tmp_size = countVolume(self.input_size[did], vol_size, np.array(self.sample_stride))
        pos[1:] = [np.random.randint(tmp_size[x]) for x in range(len(tmp_size))]
        return pos

    def index2zyx(self, index): # for test
        # int division = int(floor(.))
        pos = [0,0,0,0]
        did = self.getPosDataset(index)
        pos[0] = did
        index2 = index - self.sample_num_c[did]
        pos[1:] = self.getPosLocation(index2, self.sample_size_vol[did])
        return pos 

    def getPosLocation(self, index, sz):
        # sz: [y*x, x]
        pos = [0,0,0]
        pos[0] = np.floor(index/sz[0])
        pz_r = index % sz[0]
        pos[1] = np.floor(pz_r/sz[1])
        pos[2] = pz_r % sz[1]
        return pos 

    def getPosTest(self, index):
        pos = self.index2zyx(index)
        for i in range(1,4):
            if pos[i] != self.sample_size[pos[0]][i-1]-1:
                pos[i] = int(pos[i] * self.sample_stride[i-1])
            else:
                pos[i] = int(self.input_size[pos[0]][i-1]-self.vol_input_size[i-1])
        return pos

# -- 2. misc --
# for dataloader

def collate_fn(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    out_input, out_label, weights, weight_factor = zip(*batch)
    out_input = torch.stack(out_input, 0)
    out_label = torch.stack(out_label, 0)
    weights = torch.stack(weights, 0)

    weight_factor = np.stack(weight_factor, 0)

    return out_input, out_label, weights, weight_factor

def collate_fn_test(batch):
    pos, out_input = zip(*batch)
    test_sample = torch.stack(out_input, 0)

    return pos, test_sample    