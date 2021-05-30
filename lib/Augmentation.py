import time

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch import nn

import torch.nn.functional as F
import random
import math
import timeit
import random
from datetime import datetime

# Random number with system time
seed = datetime.now()
sseed = str(seed)
seed = sseed.split('.')
seed = seed[-1]
random.seed(seed)

class Augmentation(nn.Module):
    def __init__(self, flip=None, offset=0, scale= 0, rotate=0, noise=None):
        #super.__init__()
        super().__init__()
        self.flip = flip
        if offset>1 or offset<0:
            offset = 0.5
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise
    def _build2DTranformMatrix(self):
        mat_t = torch.eye(3)
        seed = time.time()
        #random.seed(seed)
        if self.flip:
            for i in range(2):
                if random.random() > 0.5:
                    mat_t[i,i] *= -1
        if self.rotate:
            # This might be too large
            ang = random.random() * math.pi * 2
            cs = math.cos(ang)
            sn = math.sin(ang)
            rot_t = torch.Tensor([[cs, sn, 0], \
                                  [-sn, cs, 0],\
                                 [0, 0, 1]])
            mat_t @= rot_t
        if self.offset>0:
            for i in range(2):
                offset_val = self.offset
                random_float = (random.random() * 2 - 1)
                #random_float *= 100
                offset_val *= random_float
                mat_t[i, 2] = offset_val
        if self.scale>0:
            for i in range(2):
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                mat_t[i, i] *= 1.0 + scale_float * random_float
        return mat_t

    def forward(self, input_g, label_g):
        N = input_g.shape[0]
        augmented_input_g = torch.clone(input_g)
        augmented_label_g = torch.clone(label_g)
        for n in range(N):
            transform_t = self._build2DTranformMatrix()
            #transform_t = transform_t.expand(input_g.shape[0], -1, -1)
            transform_t = transform_t.expand(1, -1, -1)
            transform_t = transform_t.to(input_g.device, torch.float32)
            inputn = augmented_input_g[n,:]
            inputn = inputn.expand(1, -1, -1, -1)
            affine_t = F.affine_grid(transform_t[:, :2],
                                     inputn.size(), align_corners=False)

            inputn = F.grid_sample(inputn,
                                   affine_t, padding_mode='border',
                                   align_corners=False)
            augmented_input_g[n,:] = inputn[0,:]
            labeln = augmented_label_g[n,:]
            labeln = labeln.expand(1, -1,-1,-1)
            labeln = F.grid_sample(labeln.to(torch.float32),
                                              affine_t, padding_mode='border',
                                              align_corners=False)
            augmented_label_g[n,:] = labeln[0,:]

            if self.noise:
                noise_t = torch.randn_like(augmented_input_g)
                noise_t *= self.noise
                augmented_input_g += noise_t
        return augmented_input_g, augmented_label_g > 0.5



import cv2
import lib.read_cv_imgs

if __name__ == '__main__':
    N = 2
    aug = Augmentation(flip=False, offset=0., scale= 1, rotate=0, noise=False)
    imgname = 'C:/Users/shimon.cohen/Pictures/CondGAN_walking_on_latancy.PNG'
    img = lib.read_cv_imgs.read_color_img(imgname, to_rgb=False)

    cv2.imshow('in_img', img)
    img = lib.read_cv_imgs.convert_to_ternsor(img)

    sp = (N,) + img.shape

    ten = torch.ones(sp)
    imgs = np.zeros(sp)
    imgs[0,:], imgs[1,:] = img,img.copy()
    ten = torch.Tensor(imgs)

    lab = torch.ones(sp)
    aten, alab = aug(ten,lab)
    aten = aten.numpy()

    for i in range(N):
        oimgi = aten[i, :]
        oimgi = np.swapaxes(oimgi, 0, 2)
        oimgi = np.swapaxes(oimgi, 0, 1)
        oimgi = oimgi.astype(np.uint8)
        #oimgi = cv2.cvtColor(oimgi, cv2.COLOR_BGR2RGB)
        cv2.imshow('aug_img_' + str(i), oimgi)
    cv2.waitKey(0)