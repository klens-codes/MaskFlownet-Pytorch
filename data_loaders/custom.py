import os

import torch
from torch.utils.data import Dataset

from data_loaders.chairs.flo import load as load_flow
from data_loaders.chairs.ppm import load as load_ppm


import skimage
import skimage.io
import numpy as np
class CustomDataSet(Dataset):
    def __init__(self, left=[],right=[]):
        super(CustomDataSet, self).__init__()
        self.split = "train"
        self.image_list = {}
        self.image_list['train'] = []
        for i,v in enumerate(left):
            self.image_list['train'].append([v,right[i]])
        self.dataset = self.image_list


    def __len__(self):
        return len(self.image_list[self.split])

    def __getitem__(self, idx):
        im0_path, im1_path = self.dataset[self.split][idx]
        img0 = skimage.io.imread(im0_path)
        img1 = skimage.io.imread(im1_path)
        img0 = torch.tensor(img0/255.).float()
        img1 = torch.tensor(img1/255.).float()
        return img0, img1,np.array([]),np.array([]), [im0_path , im1_path],np.array([])
