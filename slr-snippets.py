# -*- coding: utf-8 -*-

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

from google.colab import files
files.upload() #this will prompt you to upload the kaggle.json

!ls -lha kaggle.json

!pip install -q kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!chmod 600 /root/.kaggle/kaggle.json

!pwd

!kaggle datasets list

!kaggle datasets download -d mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out

!unzip asl-rgb-depth-fingerspelling-spelling-it-out.zip

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

from google.colab.patches import cv2_imshow
images_paths, labels = get_ASL_paths(root)
images = PathToImage(images_paths, 64, 64)
sparse_labels = DenseToSparse(labels)

x_train, x_test, y_train, y_test = train_test_split(images, sparse_labels, test_size=0.2, random_state=20, shuffle=True)
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.20) # , random_state=40, shuffle=True)


"""# **2 STREAM CNN**"""


from keras.layers import Input, Conv2D, concatenate, Dense, UpSampling2D
from keras.models import Model
import tensorflow as tf
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

#@title
class CustomEarlyStopping(keras.callbacks.Callback):
    def __init__(self, patience=0):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
#         self.restore_best_weights = restore_best_weights
#         self.mode = mode
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = 0.015

    def on_epoch_end(self, epoch, logs=None):
        current = logs["accuracy"] - logs["val_accuracy"]
        
        if current <= 0.015:
            self.wait = 0
            if np.less(current, self.best):
                self.best = current
            # Record the best weights if current results is better (less).
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
#                 print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: custom early stopping" % (self.stopped_epoch + 1))

#@title
patience = 3
default_early_stopping_cb = keras.callbacks.EarlyStopping(restore_best_weights=True, patience=patience, monitor="val_accuracy")
custom_early_stopping_cb = CustomEarlyStopping(patience=patience)
checkpoint_cb = keras.callbacks.ModelCheckpoint("ASL_dataset_classification_CNN_ModelCheckpoint.h5", monitor="val_accuracy", save_best_only=True)

#@title
learning_rates = [0.5, 0.3, 0.15, 0.07, 0.05, 0.03, 0.01]
lr = learning_rates[2]
model.compile(optimizer="adam", loss = keras.losses.categorical_crossentropy, metrics="accuracy")

import tensorflow as tf
output_tensors = tf.split(x_train, num_or_size_splits=2, axis=4)
output_tensors_val = tf.split(x_valid, num_or_size_splits=2, axis=4)

x_train_rgb = output_tensors[0]
x_train_g1 = output_tensors[1]

x_val_rgb = output_tensors_val[0]
x_val_g1 = output_tensors_val[1]

save_path = '/content/model2.h5'
try:
    history = model.fit([x_train_rgb, x_train_g1], y_train, epochs=50, validation_data=([x_val_rgb,x_val_g1], y_valid), batch_size =32)
except KeyboardInterrupt:
    model.save(save_path)
    print('Output saved to: "{}./*"'.format(save_path))
model.save('/content/model_massey_lbp.h5')

model.evaluate([x_train_rgb, x_train_g1], y_train, batch_size=64)
model.evaluate([x_val_rgb, x_val_g1], y_valid, batch_size=64)
output_tensors2 = tf.split(x_test, num_or_size_splits=2, axis=4)

# 100% test accuracy in massey dataset
model.evaluate([output_tensors2[0], output_tensors2[1]], y_test, batch_size=64)

def recall_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy', f1_m, precision_m, recall_m, 
                       tf.keras.metrics.AUC(name='auc')])

model.evaluate([output_tensors2[0], output_tensors2[1]], y_test, batch_size=64)

import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
