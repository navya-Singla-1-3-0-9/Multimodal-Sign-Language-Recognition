
import numpy as np

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from spatial_pyramid_pooling import spp
from keras.utils import np_utils
from matplotlib import pyplot as plt
import os
import random
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')


from keras.layers import Input, Conv2D, concatenate, Dense, UpSampling2D
from keras.models import Model
import tensorflow as tf
def model():
    # Input layer for RGB image
    rgb_input = Input(shape=(64, 64, 3))

    # Convolutional layers for RGB image
    rgb_conv1 = Conv2D(64, 3, activation='relu', padding = 'same')(rgb_input)
    rgb_pool1 = MaxPooling2D(2)(rgb_conv1)
    rgb_conv2 = Conv2D(128, 3, activation="relu", padding='same')(rgb_pool1)
    rgb_conv3 = Conv2D(128, 3, activation="relu", padding='same')(rgb_conv2)
    rgb_pool2 = MaxPooling2D(2)(rgb_conv3)
    rgb_conv4 = Conv2D(256, 3, activation="relu", padding='same')(rgb_pool2)
    rgb_conv5 = Conv2D(256, 3, activation="relu", padding='same')(rgb_conv4)

    ## SPP LAYERS
    rgb_model = spp(rgb_conv5)


    # Input layer for features
    g1_input = Input(shape=(64,64,3))

    # Convolutional layers for depth image
    g1_conv1 = Conv2D(64, 7, activation='relu', padding = 'same')(g1_input)
    g1_pool1 = MaxPooling2D(2)(g1_conv1)
    g1_conv2 = Conv2D(128, 3, activation="relu", padding='same')(g1_pool1)
    g1_conv3 = Conv2D(128, 3, activation="relu", padding='same')(g1_conv2)
    g1_pool2 = MaxPooling2D(2)(g1_conv3)
    g1_conv4 = Conv2D(256, 3, activation="relu", padding='same')(g1_pool2)
    g1_conv5 = Conv2D(256, 3, activation="relu", padding='same')(g1_conv4)
    g1_pool3 =spp(g1_conv5)


    # Concatenate the output of the two streams
    merged = concatenate([rgb_model,  g1_pool3])

    # # Add more layers
    merged = Conv2D(64, (3,3), activation='relu')(merged)
    merged = Conv2D(64, (3,3), activation='relu')(merged)
    merged = Flatten()(merged)
    # Final dense layer for classification
    fcn1 = Dense(128, activation="relu")(merged)
    fcn2 = Dense(64 , activation="relu")(fcn1)
    output = Dense(36, activation='softmax')(fcn2)

    # Create the model
    model = Model(inputs=[rgb_input, g1_input], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])