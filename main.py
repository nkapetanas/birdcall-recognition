import glob
import os
import re
from random import shuffle

import librosa.display
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import GlobalMaxPooling2D, Dense, Flatten, \
    BatchNormalization, Activation, Dropout, Conv2D, MaxPooling2D

from Utils import *

# a data generator for spectrogram images
def data_augmentation(df, target_size, batch_size, shuffle):
    data_generation = ImageDataGenerator(rescale=1. / 255)

    return data_generation.flow_from_dataframe(
        dataframe=df,
        x_col='bird_call_filepath',
        y_col='bird',
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle,
        class_mode='categorical',
        validate_filenames=False)


# creates our custom model
def get_model_light(input_shape=(128, 128, 3)):
    model = tf.keras.Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(256, use_bias=False))
    model.add(BatchNormalization())

    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_DENSE_LAYER))

    model.add(Dense(len(classes_to_predict), activation="softmax"))

    model.summary()
    return model

# loader of images from folder
def load_created_melspectro():
    birdcall_list = []

    files = glob.glob("./output_data/*.tif")
    shuffle(files)
    current_number = 0
    total = len(files) # * 0.4

    for img in files:
        if current_number <= total:
            filename = img[14:]
            bird_type = re.search('_(.*).tif', filename).group(1)
            birdcall_list.append({"bird_call_filepath": "./output_data/" + filename,
                                  "bird": bird_type})
        current_number += 1

    return pd.DataFrame(birdcall_list)


birdcall_df = load_created_melspectro()


# split data to create train - validation - test
train_df, test_df = train_test_split(birdcall_df,
                                     stratify=birdcall_df.bird.values,
                                     test_size=0.30, random_state=1)

train_df, valid_df = train_test_split(train_df,
                                      stratify=train_df.bird.values,
                                      test_size=0.25, random_state=1)


classes_to_predict = sorted(birdcall_df.bird.unique())
classes_pos_dict = (dict(enumerate(classes_to_predict)))

# create callbacks for reducing Learning rate, saving mode and early stopping.
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.7),
             EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='best_model1.h5', monitor='val_loss', save_best_only=True)]

model = get_model_light()
model.compile(loss="categorical_crossentropy", optimizer='adam')


# create one data generator for each sub set
train_generator = data_augmentation(train_df, TARGET_SIZE, TRAINING_BATCH_SIZE, True)
validation_generator = data_augmentation(valid_df, TARGET_SIZE, VALIDATION_BATCH_SIZE, False)
test_generator = data_augmentation(test_df, TARGET_SIZE, VALIDATION_BATCH_SIZE, False)
# fit the model
history = model.fit(train_generator,
                    epochs=20,
                    validation_data=validation_generator,
                    callbacks=callbacks)
# plot the loss of training
plot_hist_loss(history)    
# predict unseen test 
generated_predictions = model.predict_generator(test_generator)

test_results_df = pd.DataFrame(columns=["prediction", "groundtruth"])

# true labels
y_true = test_df.bird.values
# predicted labels
y_predicted = [classes_pos_dict.get(key) for key in np.argmax(generated_predictions, axis=1)]

# print results
from sklearn.metrics import classification_report
print(classification_report(y_true, y_predicted))


#model.load_weights("best_model1.h5")    
