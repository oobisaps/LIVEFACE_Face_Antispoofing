""" 
    Standard Modules
"""

import os
import time
import copy

from PIL import Image


""" 
    Pytorch Modules
"""

import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as Functional


from torch.autograd import Variable
import torch.utils.data as data_utils



"""
    External Modules
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
    Internal Modules
"""

from Loss_Functions import *
from NN_Architecture import *
from Images_DataLoader import *


# Constants

SEED = 17
EPOCHS = 10
BATCH_SIZE = 1
SHUFFLE = True
NUM_CLASSES = 2
NUM_PATCHES = 8
OUTPUT_SIZE = (512,512)
NUM_WORKERS = os.cpu_count()
COLOR_SPACE = COLOR_SPACES.HSV_YCrCb['title']
IN_CHANNELS = COLOR_SPACES.HSV_YCrCb['in_channels']



transform_full_img = transforms.Compose(             
        [
            ColorSpace_Transformer(color_space = COLOR_SPACE),
            Rescale_Image(output_size = OUTPUT_SIZE),
            ToTensor()
        ]
    )


train_dataset = pd.read_csv('Data_Locations/TRAIN_DATASET.csv')
test_dataset = pd.read_csv('Data_Locations/TEST_DATASET.csv')
validation_dataset = pd.read_csv('Data_Locations/VALIDATION_DATASET.csv')


train_dataset = AntiSpoofing_Dataset(dataframe = train_dataset, use_case = 'Full_Image')
test_dataset = AntiSpoofing_Dataset(dataframe = test_dataset, use_case = 'Full_Image')
validation_dataset = AntiSpoofing_Dataset(dataframe = validation_dataset, use_case = 'Full_Image')


train_loader = data_utils.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = SHUFFLE)
test_loader = data_utils.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = SHUFFLE)
validation_loader = data_utils.DataLoader(dataset = validation_dataset, batch_size = BATCH_SIZE, shuffle = SHUFFLE)






























