
import numpy as np

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
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
## SNIPPET TO LOAD DATASET FROM KAGGLE
# from google.colab import files
# files.upload() #this will prompt you to upload the kaggle.json

# !ls -lha kaggle.json

# !pip install -q kaggle

# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/

# !chmod 600 /root/.kaggle/kaggle.json

# !pwd

# !kaggle datasets list

# !kaggle datasets download -d mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out

# !unzip asl-rgb-depth-fingerspelling-spelling-it-out.zip

root = 'drive/My Drive/mu_data'

def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters 
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
     
    filters = []
    num_filters = 16
    ksize = 8 # The local area to evaluate
    sigma = 3.0  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

def apply_filter(img, filters):
# This general function is designed to apply filters to our image
    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)
    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1 # remain depth same as original image
    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv2.filter2D(img, depth, kern)  #Apply filter to image
         
        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            elif img[x + i][y + j] > center:
                val += 2 ** ((i + 1) * 3 + (j + 1))
    return val

def lbp_filter(img):
    lbp_img = np.zeros(img.shape, np.uint8)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            lbp_img[i][j] = lbp_calculated_pixel(img, i, j)
    return lbp_img

def get_ASL_paths(root):
    All_images_paths = []
    labels = []
    for img in os.listdir(f'{root}'):
      All_images_paths.append(f'{root}/{img}')
      label = img.split('_')[1]
      labels.append(label)
    for img in os.listdir('drive/My Drive/mu_part2'):
      All_images_paths.append(f'drive/My Drive/mu_part2/{img}')
      label = img.split('_')[1]
      labels.append(label)
    for img in os.listdir('drive/My Drive/mu_part3'):
      All_images_paths.append(f'drive/My Drive/mu_part3/{img}')
      label = img.split('_')[1]
      labels.append(label)          
    for img in os.listdir('drive/My Drive/mu_part4'):
      All_images_paths.append(f'drive/My Drive/mu_part4/{img}')
      label = img.split('_')[1]
      labels.append(label)
    for img in os.listdir('drive/My Drive/mu_part5'):
      All_images_paths.append(f'drive/My Drive/mu_part5/{img}')
      label = img.split('_')[1]
      labels.append(label)
    return All_images_paths, labels


def PathToImage(image_paths, width=128, height=128):
    images = np.zeros((len(image_paths), width, height, 3,2), dtype=np.ubyte)
    i=0;
    for image_path in image_paths:
        print(image_path)
        #color image
        image_array = cv2.imread(image_path)
        image_resized = cv2.resize(image_array,(width,height))
        images[i,:,:,:,0] = image_resized
        ## FEATURE IMAGE
        img = cv2.imread(image_path)
        gfilters = create_gaborfilter()
        image_g = apply_filter(img, gfilters)
        image_g = cv2.resize(image_g, (128,156))
        gray = cv2.cvtColor(image_g, cv2.COLOR_BGR2GRAY)
        gab_lbp_img = lbp_filter(gray)
        gab_lbp_img = cv2.cvtColor(gab_lbp_img, cv2.COLOR_GRAY2BGR)
        image_resized = cv2.resize(gab_lbp_img,(64,64))
        images[i,:,:,:,1] = image_resized

        i+=1
    return images

def DenseToSparse(labels):
    labels = pd.Series(labels).apply(lambda c: int(c) if c <= '9' else ord(c) - ord('a')+10)
    labels = labels.astype(np.ubyte)
    sparse_labels = np.zeros((labels.shape[0], 36))
    sparse_labels[labels.index,labels] = 1
    return sparse_labels
def split_data():
    images_paths, labels = get_ASL_paths(root)
    images = PathToImage(images_paths, 64, 64)
    sparse_labels = DenseToSparse(labels)
    x_train, x_test, y_train, y_test = train_test_split(images, sparse_labels, test_size=0.2, random_state=20, shuffle=True)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.20) # , random_state=40, shuffle=True)
    return x_train, x_test, x_valid, y_valid, x_test, y_test