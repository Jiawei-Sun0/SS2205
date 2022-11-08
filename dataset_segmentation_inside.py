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
import torchvision.transforms as transform
import torch
import overlay

def augment(img,rotate_angle,scale,x,y):
    # print('sd',img.shape)
    img = img.transpose(1,2,0)
    height,width = img.shape[:2]
    Rmatrix = cv2.getRotationMatrix2D((height / 2, width / 2), rotate_angle, scale)
    Mmatrix = np.float32([[1, 0, x], [0, 1, y]])
    if random.random() < 0.5:
        img = cv2.warpAffine(img, Rmatrix, (width, height))
    if random.random() < 0.5:
        img = cv2.warpAffine(img, Mmatrix, (width, height))
    # print('ss',img.shape)s
    return img

class insideDataSetSegmentation(torch.utils.data.Dataset):
    data_transformer = transform.Compose([
        transform.RandomHorizontalFlip(),
        transform.RandomAffine(degrees=30,translate=(0.2,0.2),scale=(0.7,1)),
    ])

    def __init__(self, base_path):
        super(insideDataSetSegmentation, self).__init__()
        self.img_files = sorted(glob.glob(os.path.join(base_path, 'image', '*.png')))
        self.mask_files = []
        self.gate = len(self.img_files)
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(base_path, 'mask',
                                                os.path.basename(img_path)))
        ids = np.random.randint(0, len(self.img_files), 
                size=random.randint(0, int(len(self.img_files)*random.uniform(2,3))))
        for id in ids:
            self.img_files.append(self.img_files[id])
            self.mask_files.append(self.mask_files[id])


    def __getitem__(self, index):
        
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        
        # load image data and normalization
        data = imageio.imread(img_path).astype(np.float32) / 255.0

        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        else:
            data = data.transpose(2, 0, 1)

        # load label data
        label = imageio.imread(mask_path).astype(np.float32)

        if label.ndim == 3:  # RGB to gray-scale
            label = 0.299 * label[:, :, 2]\
                + 0.587 * label[:, :, 1]\
                + 0.114 * label[:, :, 0]

        label = (label > 0.0).astype(np.float32)[np.newaxis, :, :]

        # data augmentation

        # data = torch.from_numpy(data).float()
        # label = torch.from_numpy(label).float()
        # print(index,end='')
        if index > self.gate:
            # print(' augmented')
            rotate_angle = random.random() * 30
            scale = random.uniform(0.7,1)
            x, y = random.randrange(-20, 20), random.randrange(-30, 30)

            # data = data.transpose(1,2,0)
            # label = label.transpose(1,2,0)
            data = augment(np.asarray(data),rotate_angle,scale,x,y)
            data = data.transpose(2,0,1)
            label = augment(np.asarray(label),rotate_angle,scale,x,y)
            label = np.expand_dims(label, axis=0)

        # data = data.transpose(2,0,1)
        # label = label.transpose(2,0,1)
        # print(label.shape)
        # imageio.imwrite('./testdata/data'+str(index)+'.png',data.transpose(1,2,0)*255)
        # imageio.imwrite('./testdata/mask'+str(index)+'.png',label.transpose(1,2,0)*255)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)

if __name__=="__main__":
    print('')
    # dataset = DataSetSegmentation('training/')
    # dd = dataset.mask_files
    # print(dd)
