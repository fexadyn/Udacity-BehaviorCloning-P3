import os
import shutil
import cv2
import csv
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import sklearn
import numpy as np
from random import shuffle, random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Cropping2D, ELU
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160

def make_preprocess_layers():
    """
    Build first layer of the network, normalize the pixels to [-1,1]
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - .5,
                     input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    return model

def make_lenet():
    """
    Build a LeNet model using keras
    """
    model = make_preprocess_layers()
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model


def make_commaai():
    # model = Sequential()
    # model.add(Lambda(lambda x: x / 255. - .5, input_shape=(160, 320, 3)))
    model = make_preprocess_layers()
    model.add(Convolution2D(3, 1, 1, subsample=(1, 1), border_mode='same',
                            init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(16, 5, 5, subsample=(4, 4), border_mode="same",
                            init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode="same",
                            init = 'he_normal'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same", 
                            init = 'he_normal'))
    model.add(Flatten())
    model.add(Dropout(.2))

    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))

    model.add(ELU())
    model.add(Dense(1))

    return model

def make_nvidia():
    model = make_preprocess_layers()
    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(1))

    return model