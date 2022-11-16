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
import dataset_segmentation

def augment(img,rotate_angle,scale,x,y,poss0,poss1):
    # print('sd',img.shape)
    img = img.transpose(1,2,0)
    height,width = img.shape[:2]
    Rmatrix = cv2.getRotationMatrix2D((height / 2, width / 2), rotate_angle, scale)
    Mmatrix = np.float32([[1, 0, x], [0, 1, y]])
    if poss0 < 0.75:
        Rmatrix = cv2.getRotationMatrix2D((height / 2, width / 2), rotate_angle, scale)
    else:
        Rmatrix = cv2.getRotationMatrix2D((height / 2, width / 2), 0, 1)
    if poss1 < 0.75:
        Mmatrix = np.float32([[1, 0, x], [0, 1, y]])
    else:
        Mmatrix = np.float32([[1, 0, 0], [0, 1, 0]])

    img = cv2.warpAffine(img, Rmatrix, (width, height))
    img = cv2.warpAffine(img, Mmatrix, (width, height))
    # print('ss',img.shape)s
    return img

class DataAugmentation(dataset_segmentation.DataSetSegmentation):
    aug_size = 1.2

    def __init__(self, base_path):
        self.img_files = sorted(glob.glob(os.path.join(base_path, 'image', '*.png')))
        self.gate = len(self.img_files)
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(base_path, 'mask',
                                                os.path.basename(img_path)))

        ids = np.random.randint(0, len(self.img_files), 
                size=int(len(self.img_files)*self.aug_size))
        print('generated data:',len(ids))
        for id in ids:
            self.img_files.append(self.img_files[id])
            self.mask_files.append(self.mask_files[id])

        # for id in range(len(self.img_files)):
        #     self.img_files.append(self.img_files[id])
        #     self.mask_files.append(self.mask_files[id])
    
    def __getitem__(self, index):
        super().__getitem__(index)
        if index > self.gate:
            # print(' augmented')
            rotate_angle = random.random() * 30
            scale = random.uniform(0.7,1)
            x, y = random.randrange(-20, 20), random.randrange(-30, 30)
            poss0 = random.random()
            poss1 = random.random()
            # data = data.transpose(1,2,0)
            # label = label.transpose(1,2,0)
            self.data = augment(np.asarray(self.data),rotate_angle,scale,x,y,poss0,poss1)
            # imageio.imwrite('./testdata/data'+str(index)+'.png',(self.data*255).astype(np.uint8))
            self.data = self.data.transpose(2,0,1)
            self.label = augment(np.asarray(self.label),rotate_angle,scale,x,y,poss0,poss1)
            # imageio.imwrite('./testdata/mask'+str(index)+'.png',self.label.astype(np.uint8)*255)
            self.label = np.expand_dims(self.label, axis=0)

        return torch.from_numpy(self.data).float(), torch.from_numpy(self.label).float()

if __name__=="__main__":
    print('')
