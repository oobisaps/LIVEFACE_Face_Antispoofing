import io
# import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from keras.datasets import fashion_mnist
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

model = Sequential()

model.add(Conv2D(filters = 50, kernel_size = (5,5), strides = 1, input_shape = (96,96,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

model.add(Conv2D(filters = 100, kernel_size = (3,3), strides = (1,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

model.add(Conv2D(filters = 150, kernel_size = (3,3), strides = (1,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

model.add(Conv2D(filters = 200, kernel_size = (3,3), strides = (1,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

model.add(Conv2D(filters = 250, kernel_size = (3,3), strides = (1,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

model.add(Flatten())

model.add(Dense(1000))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(400))
model.add(BatchNormalization())
model.add(Dense(2))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()