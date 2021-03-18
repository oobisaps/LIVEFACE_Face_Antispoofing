'''
    Standard Modules
'''

import os
import sys
import time
import pickle as pkl
from collections import Counter

'''
    External Modules
'''

import cv2

import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt

import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision
from torchvision import transforms
from PIL import Image

'''
    Internal Modules
'''

from NN_Architecture import *
import LIVEFACE_MOBILE_ML_Tools as ml_tools
from  Images_DataLoader import AntiSpoofing_Dataset, RandomCrop, ToTensor, ColorSpace_Transformer, COLOR_SPACES  

'''
    CONSTANTS
'''

SEED = 17
EPOCHS = 10
BATCH_SIZE = 32
SHUFFLE = True
NUM_CLASSES = 2
NUM_PATCHES = 8
NUM_WORKERS = os.cpu_count()
COLOR_SPACE = COLOR_SPACES.HSV_YCrCb['title']
IN_CHANNELS = COLOR_SPACES.HSV_YCrCb['in_channels']


# uncomment only on my machine

train_dataset = pd.read_csv('Data_Locations/TRAIN_DATASET_PATCHES.csv')
test_dataset = pd.read_csv('Data_Locations/TEST_DATASET_PATCHES.csv')
validation_dataset = pd.read_csv('Data_Locations/VALIDATION_DATASET_PATCHES.csv')

print('TRAIN')
print(train_dataset['LABEL'].value_counts())
print()

print('TEST')
print(test_dataset['LABEL'].value_counts())
print()

print('VALIDATION')
print(validation_dataset['LABEL'].value_counts())
print()

transform_all = transforms.Compose(             
        [
            ColorSpace_Transformer(color_space = COLOR_SPACE), 
            ToTensor()
        ]
    )

train_dataset = AntiSpoofing_Dataset(dataframe = train_dataset, transform = transform_all)
test_dataset = AntiSpoofing_Dataset(dataframe = test_dataset, transform = transform_all)
validation_dataset = AntiSpoofing_Dataset(dataframe = validation_dataset, transform = transform_all)



print()

train_loader = data_utils.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE,shuffle = SHUFFLE)
test_loader = data_utils.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE,shuffle = SHUFFLE)
validation_loader = data_utils.DataLoader(dataset = validation_dataset, batch_size = BATCH_SIZE,shuffle = SHUFFLE)


# print(len(train_dataset) / 32)
# print(train_set['label'].value_counts())


model = Patch_Based_CNN(in_channels = IN_CHANNELS, input_shape = (96, 96), num_classes = NUM_CLASSES)


VERSION = 'V6'
TITLE = 'Patch_Based_CNN'
ABS_PATH = 'Model_Results'

ml_tools.train_model(epochs = EPOCHS, 
                         learning_rate = 0.001, 
                         model_net = model, 
                         batch_size = 32, 
                         num_patches = NUM_PATCHES,
                         loader = train_loader,
                         test_loader = test_loader,
                         val_loader = validation_loader,
                         input_size = (IN_CHANNELS,96,96),  # (C x H x W)
                         color_space = COLOR_SPACE, 
                         abs_path = ABS_PATH,
                         version = VERSION,
                         title = TITLE,
                         filename = 'yeah boy',
                         std_out = '')
# except:

#     current_time = time.ctime()
#     current_time = current_time.replace(' ','_')

#     ml_tools.get_exception()


  