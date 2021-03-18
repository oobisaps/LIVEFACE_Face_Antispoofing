'''
    Standard Modules
'''

import os

from os import listdir
from os.path import join, isfile

'''
    External Modules
'''

import cv2

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

'''
    Internal Modules
'''

import LIVEFACE_MOBILE_Tools as tl

ABS_PATH = 'Data'

image_case = 'Original'

KASPI_LIVE_PATH = os.path.join(ABS_PATH, 'KASPI/{}/LIVE'.format(image_case))

KASPI_FAKE_DEVICE_VISIBLE = os.path.join(ABS_PATH, 'KASPI/{}/FAKE/DEVICE_VISIBLE'.format(image_case))
KASPI_FAKE_DEVICE_INVISIBLE = os.path.join(ABS_PATH, 'KASPI/{}/FAKE/DEVICE'.format(image_case))
KASPI_FAKE_PHOTO = os.path.join(ABS_PATH, 'KASPI/{}/FAKE/PHOTO'.format(image_case))

YDL_LIVE = os.path.join(ABS_PATH, 'YDL/LIVE/{}'.format(image_case))
YDL_LIVE_VAL_TEST = os.path.join(ABS_PATH, 'YDL/LIVE/Test')

YDL_FAKE = os.path.join(ABS_PATH, 'YDL/FAKE/{}'.format(image_case))
YDL_FAKE_VAL_TEST = os.path.join(ABS_PATH, 'YDL/FAKE/Test')

ALL_PATHES = (
    
    # KASPI_LIVE_PATH,
    # KASPI_FAKE_DEVICE_VISIBLE,
    # KASPI_FAKE_DEVICE_INVISIBLE,
    KASPI_FAKE_PHOTO,
    YDL_LIVE,
    YDL_LIVE_VAL_TEST,
    YDL_FAKE,
    YDL_FAKE_VAL_TEST

)

for path in ALL_PATHES:

    # path = KASPI_LIVE_PATH

    images_generator = ((join(path, file), cv2.imread(filename = join(path, file))) for file in  listdir(path) if isfile(path = join(path, file)))

    scale = 0.5
    limit_images = 100

    for i in range(limit_images):
        image_name,image = next(images_generator)

        print('CURRENT IMAGE : ',image_name)
        print()

        print(image.shape)
        
        blue_scale_image = image[...,0]
        green_scale_image = image[...,1]
        red_scale_image = image[...,2]
        gray_scale_image = blue_scale_image * 0.114 + green_scale_image * 0.587 + red_scale_image * 0.299  # it's possible to round



        channels_code = (0, 0, 1, 2)
        colors_codes = ('silver','deepskyblue','green','red')
        
        images_container = [gray_scale_image,blue_scale_image,green_scale_image,red_scale_image]

        # images_codes = list(zip(images_container,channels_code))
        # params = [param_wrapper(image_input = i[0],channel = i[1]) for i in images_codes]
        # images_hists = [cv2.calcHist(**param) for param in params]

        for i in range(len(images_container)):
            sns.distplot(images_container[i].ravel(),bins = 256, kde = False,
                        hist_kws = {"color": colors_codes[i]})

        plt.legend(labels = ('gray','blue','green','red'), loc = 'best')

        hist_path = join('Data/ADDITIONAL/','_'.join(image_name.split('/')))

        # image_name[image_name.rfind('/')+1:]

        plt.title('Histograms for all channels')
        plt.savefig(hist_path)
