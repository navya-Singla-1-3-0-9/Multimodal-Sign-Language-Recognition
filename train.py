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
from model import model
import load_data
from google.colab import drive
drive.mount('/content/drive')


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
x_train, x_test, x_valid, y_valid, x_test, y_test = load_data.split_data()
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
model.save('/content/model_mu_lbp.h5')

model.evaluate([x_train_rgb, x_train_g1], y_train, batch_size=64)
model.evaluate([x_val_rgb, x_val_g1], y_valid, batch_size=64)
output_tensors2 = tf.split(x_test, num_or_size_splits=2, axis=4)

# 100% test accuracy in massey dataset
model.evaluate([output_tensors2[0], output_tensors2[1]], y_test, batch_size=64)
