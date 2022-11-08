"""
DataSet for image segmentation
  [reference] https://discuss.pytorch.org/t/dataloader-for-semantic-segmentation/48290

Created on Sun Oct 23 2022
@author: ynomura
"""

import glob
from pathlib import Path
from sys import path_hooks
import imageio
import numpy as np
import cv2
import os
import random
import torch


class DataSetSegmentation(torch.utils.data.Dataset):
    def __init__(self, base_path):
        super(DataSetSegmentation, self).__init__()
        self.img_files = sorted(glob.glob(os.path.join(base_path, 'image', '*.png')))
        self.gate = len(self.img_files)
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(base_path, 'mask',
                                                os.path.basename(img_path)))

        self.term = []
        for i in sorted(glob.glob(os.path.join(base_path, 'image', 'aug', '*.png'))):
            self.term.append(i)
            self.img_files.append(i)

        for img_path in self.term:
            self.mask_files.append(os.path.join(base_path, 'mask', 'aug',
                                                os.path.basename(img_path)))


    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        # load image data and normalization
        self.data = imageio.imread(img_path).astype(np.float32) / 255.0

        if self.data.ndim == 2:
            self.data = self.data[np.newaxis, :, :]
        else:
            self.data = self.data.transpose(2, 0, 1)

        # load label self.data
        self.label = imageio.imread(mask_path).astype(np.float32)

        if self.label.ndim == 3:  # RGB to gray-scale
            self.label = 0.299 * self.label[:, :, 2]\
                + 0.587 * self.label[:, :, 1]\
                + 0.114 * self.label[:, :, 0]

        self.label = (self.label > 0.0).astype(np.float32)[np.newaxis, :, :]

        # self.data augmentation

        return torch.from_numpy(self.data).float(), torch.from_numpy(self.label).float()

    def __len__(self):
        return len(self.img_files)

if __name__=="__main__":
    print('')
    # dataset = DataSetSegmentation('training/')
    # dd = dataset.mask_files
    # print(dd)
