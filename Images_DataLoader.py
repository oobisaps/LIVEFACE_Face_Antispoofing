import os

import cv2
import torch
import numpy as np
import pandas as pd

import torch
from torch.nn import *
from collections import namedtuple
from torch.utils.data import Dataset


COLOR_SPACES = namedtuple('Color_Spaces',['BGR','HSV','YCrCb','BGR_HSV','BGR_YCrCb','HSV_YCrCb'])

COLOR_SPACES.BGR = {
    'title' : 'BGR',
    'in_channels' : 3
}

COLOR_SPACES.HSV = {
    'title' : 'HSV',
    'in_channels' : 3
}

COLOR_SPACES.YCrCb = {
    'title' : 'YCrCb',
    'in_channels' : 3
}

COLOR_SPACES.BGR_HSV = {
    'title' : 'BGR + HSV',
    'in_channels' : 6
}

COLOR_SPACES.BGR_YCrCb = {
    'title' : 'BGR + YCrCb',
    'in_channels' : 6
}

COLOR_SPACES.HSV_YCrCb = {
    'title' : 'HSV + YCrCb',
    'in_channels' : 6
} 

class AntiSpoofing_Dataset(Dataset):

    def __init__(self, dataframe, use_case, transform = None):
        
        self.dataframe = dataframe
        if use_case == 'Patch_Based_CNN':

            self.labels = list(self.dataframe['LABEL'])
            self.image_path = list(self.dataframe['PATH'])
        
        elif use_case == 'Face':
            
            self.labels = list(self.dataframe['label'])
            self.image_path = list(self.dataframe['image_path_cropped'])
        
        elif use_case == 'Full_Image':

            self.labels = list(self.dataframe['label'])
            self.image_path = list(self.dataframe['image_path'])

        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_path[idx])
        label = self.labels[idx]
        path = self.image_path[idx]

        sample = {
            'image' : image, 
            'label' : label,
            'path'  : path
        }



        if self.transform:

            sample = self.transform(sample)

        return sample

class ToTensor(object):
    
    
    def __call__(self, sample): 
        image, label = sample['image'], sample['label']



        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W


        label = 1 if label == 'LIVE' else 0

        image = image.transpose((2, 0, 1))


        sample['image'] = image
        sample['label'] = label

        return sample

class RandomCrop(object):

    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        images = []

        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]

        new_h, new_w = self.output_size

        tops = np.random.randint(low = 0, high = h - new_h, size = 8)
        lefts = np.random.randint(low = 0, high = w - new_w, size = 8)

        for i in range(len(tops)):
            images.append(image[tops[i] : tops[i] + new_h, lefts[i] : lefts[i] + new_w])
        
        labels = [label] * 8

        sample = {
            'images' : images,
            'labels' : labels
        }

        return sample

class ColorSpace_Transformer(object):

    def __init__(self,color_space):
        self.color_space = color_space

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        if self.color_space == 'BGR':
            pass
        
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2HSV)
            # print(image.shape)
        
        elif self.color_space == 'YCrCb':
            image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2YCR_CB)

        elif self.color_space == 'BGR + HSV':
            hsv_image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2HSV)

            image = np.concatenate((image,hsv_image), axis = 2)

        elif self.color_space == 'BGR + YCrCb':
            y_cr_cb_image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2YCR_CB)

            image = np.concatenate((image,y_cr_cb_image), axis = 2)
    
        
        elif self.color_space == 'HSV + YCrCb':
            hsv_image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2HSV)
            y_cr_cb_image = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2YCR_CB)

            image = np.concatenate((hsv_image, y_cr_cb_image), axis = 2)
        
        else: 
            pass
        

        sample['image'] = image
        sample['label'] = label

        return sample


class Rescale_Image:

    
    def __call__(self, sample):

        image = sample['image']
        image = cv2.resize(src = image, dsize = self.output_size)
    
        sample['image'] = image
        return sample


class Normalizer:

    def __init__(self, mean, std):

        super(Normalizer, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1,1,1)


    def forward(self, img):

        return (img - self.mean) / self.std












