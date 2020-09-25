import glob
import os
import re
from matplotlib import pyplot as plt
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

data_generation = ImageDataGenerator(rescale=1. / 255)

train_generator = data_generation.flow_from_directory(
    directory='./output_spectro/',
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42)

validation_generator = data_generation.flow_from_directory(
    directory='./output_spectro/',
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=5)


def get_model():
    efficient_net_layers = tf.keras.applications.EfficientNetB0(weights=None, include_top=False,
                                                                input_shape=(128, 128, 3))

    for layer in efficient_net_layers.layers:
        layer.trainable = True

    model = tf.keras.Sequential()
    model.add(efficient_net_layers)

    model.add(GlobalMaxPooling2D())
    model.add(Dense(256, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_DENSE_LAYER))

    model.add(Dense(classes_to_predict, activation="softmax"))

    model.summary()
    return model


callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.7),
             EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

model = get_model()
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

history = model.fit(train_generator,
                    epochs=20,
                    shuffle=True,
                    validation_data=validation_generator,
                    callbacks=callbacks)

model.save("best_model2.h5")

fig, ax = plt.subplots(2, 1, figsize=(16, 9))
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

final_loss, final_acc = model.evaluate_generator(generator=validation_generator)
print(f'Accuracy = {final_acc} and Loss = {final_loss}')

plot_hist(history)
plot_hist_loss(history)
plot_loss_over_epoch(history)
