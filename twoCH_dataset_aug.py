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
np.set_printoptions(threshold=np.inf)

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

class twoCH(torch.utils.data.Dataset):
    aug_size = 1.2

    def __init__(self, base_path):
        self.img_files = sorted(glob.glob(os.path.join(base_path, 'image', '*.png')))
        self.gate = len(self.img_files)
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(base_path, 'mask',
                                                os.path.basename(img_path)[:-4]+'.png'))
    
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

        # 分割不同颜色label为labels->分别转换为二值-> || =>数据增强（2,480,640）
        # loss/dice问题：如果不能直接用就分割channel然后平均
        # test输出问题：分成两张输出
        # git branch->2channel 
        # dice + crossentropyloss
        
        final = np.zeros((2,self.label.shape[0],self.label.shape[1],self.label.shape[2]))
        self.outLabel = np.zeros((2,self.label.shape[0],self.label.shape[1]))

        mask = (self.label == [255,200,0,255]).all(axis=2)
        final[0][mask] = [255,200,0,255]
        mask = (self.label == [200,0,0,255]).all(axis=2)
        final[1][mask] = [200,0,0,255]
        # FOR LOOP IS TOO SLOW
        # for i in range(self.label.shape[0]):
        #     for j in range(self.label.shape[1]):
        #         if all(self.label[i,j] == [255,200,0,255]):
        #             final[0,i,j,:] = [255,200,0,255]
        #         if all(self.label[i,j] == [200,0,0,255]):
        #             final[1,i,j,:] = [200,0,0,255]
        

        for i in range(final.shape[0]):
            if final[i].ndim == 3:  # RGB to gray-scale
                self.outLabel[i] = 0.299 * final[i, :, :, 2]\
                    + 0.587 * final[i, :, :, 1]\
                    + 0.114 * final[i, :, :, 0]
            self.outLabel[i] = (self.outLabel[i] > 0.0).astype(np.float32)
            imageio.imwrite('./testdata/mask'+str(index)+'no'+str(i)+'.png',self.outLabel[i].astype(np.uint8)*255)
        # print(torch.from_numpy(self.data).float().shape,torch.from_numpy(self.outLabel).float().shape)

        return torch.from_numpy(self.data).float(), torch.from_numpy(self.outLabel).float()
    def __len__(self):
        return len(self.img_files)

class two_aug(twoCH):
    def __init__(self, base_path):
        super().__init__(base_path)

        ids = np.random.randint(0, len(self.img_files), 
                size=int(len(self.img_files)*self.aug_size))
        print('generated data:',len(ids))
        for id in ids:
            self.img_files.append(self.img_files[id])
            self.mask_files.append(self.mask_files[id])

    def __getitem__(self, index):
        super().__getitem__(index)
        self.label = self.outLabel
        if index > self.gate:
            # print(' augmented')
            rotate_angle = random.random() * 30
            scale = random.uniform(0.7,1)
            x, y = random.randrange(-20, 20), random.randrange(-30, 30)
            poss0 = random.random()
            poss1 = random.random()

            self.data = augment(np.asarray(self.data),rotate_angle,scale,x,y,poss0,poss1)
            # imageio.imwrite('./testdata/data'+str(index)+'.png',(self.data*255).astype(np.uint8))
            self.data = self.data.transpose(2,0,1)
            self.label = augment(np.asarray(self.outLabel),rotate_angle,scale,x,y,poss0,poss1)
            # imageio.imwrite('./testdata/mask'+str(index)+'.png',self.label.astype(np.uint8)*255)
            self.label = self.label.transpose(2,0,1)
        return torch.from_numpy(self.data).float(), torch.from_numpy(self.label).float()

if __name__=="__main__":
    print('')
