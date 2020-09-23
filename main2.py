import glob
import os
import re

import librosa.display
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import GlobalMaxPooling2D, Dense, BatchNormalization, Activation, Dropout

import warnings

from Utils import *
from properties import *


warnings.filterwarnings('ignore')

classes_to_predict = 147

data_generation = ImageDataGenerator(rescale = 1. / 255)

train_generator = data_generation.flow_from_directory(
    directory = '../birdsong-recognition/output_spectro/output_spectro/',
    target_size = (128, 128),
    color_mode ="rgb",
    batch_size = 32,
    class_mode ="categorical",
    shuffle = True,
    seed = 42)

test_generation = data_generation.flow_from_directory(
    directory = '../birdsong-recognition/output_spectro/output_spectro/',
    target_size = (128, 128),
    color_mode ="grayscale",
    batch_size = 32,
    class_mode ="categorical",
    shuffle = True,
    seed = 5)


def get_model():
    efficient_net_layers = tf.keras.applications.EfficientNetB0(weights=None, include_top=False,
                                                                input_shape=(128, 128, 3))


    for layer in efficient_net_layers.layers:
        layer.trainable = True

    model = tf.keras.Sequential()
    model.add(efficient_net_layers)

    model.add(GlobalMaxPooling2D())
    model.add(Dense(256, use_bias = False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_DENSE_LAYER))

    model.add(Dense(classes_to_predict, activation = "softmax"))

    model.summary()
    return model


callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.7),
             EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]


model = get_model()
model.compile(loss="categorical_crossentropy", optimizer='adam')

history = model.fit(train_generator,
                    epochs = 20,
                    shuffle = True,
                    validation_data = test_generation,
                    callbacks = callbacks)


model.load_weights("best_model.h5")
