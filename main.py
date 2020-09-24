import glob
import os
import re
from random import shuffle

import librosa.display
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metric import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import GlobalMaxPooling2D, Dense, Flatten, \
    BatchNormalization, Activation, Dropout, Conv2D, MaxPooling2D

from Utils import *


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


def get_model():
    efficient_net_layers = tf.keras.applications.EfficientNetB0(weights=None, include_top=False,
                                                                input_shape=(128, 128, 3))

    # conv_base = ResNet50(weights='imagenet', include_top=False, pooling=None)
    # conv_base.trainable = False

    for layer in efficient_net_layers.layers:
        layer.trainable = True

    model = tf.keras.Sequential()
    model.add(efficient_net_layers)

    model.add(GlobalMaxPooling2D())
    model.add(Dense(256, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_DENSE_LAYER))

    model.add(Dense(len(classes_to_predict), activation="softmax"))

    model.summary()
    return model


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


def load_created_melspectro():
    birdcall_list = []

    files = glob.glob("./output_data/*.tif")
    shuffle(files)
    current_number = 0
    total = len(files) * 0.9

    for img in files:
        if current_number <= total:
            filename = img[14:]
            bird_type = re.search('_(.*).tif', filename).group(1)
            birdcall_list.append({"bird_call_filepath": "./output_data/" + filename,
                                  "bird": bird_type})
        current_number += 1

    return pd.DataFrame(birdcall_list)


birdcall_df = load_created_melspectro()

"""
x_train, x_test, y_train, y_test =  train_test_split(
    birdcall_df.bird_call_filepath.values, birdcall_df.bird.values,
    test_size=0.30, random_state=1)


x_train, x_val, y_train, y_val =  train_test_split(
    x_train, y_train,
    test_size=0.20, random_state=1)
"""

train_df, test_df = train_test_split(birdcall_df,
                                     stratify=birdcall_df.bird.values,
                                     test_size=0.30, random_state=1)

train_df, valid_df = train_test_split(train_df,
                                      stratify=train_df.bird.values,
                                      test_size=0.25, random_state=1)

"""
training_item_count = int(len(birdcall_df) * TRAINING_PERCENTAGE)
validation_item_count = len(birdcall_df) - int(len(birdcall_df) * TRAINING_PERCENTAGE)
training_df = birdcall_df[:training_item_count]
validation_df = birdcall_df[training_item_count:]
"""
classes_to_predict = sorted(birdcall_df.bird.unique())

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.7),
             EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='best_model1.h5', monitor='val_loss', save_best_only=True)]

model = get_model_light()
model.compile(loss="categorical_crossentropy", optimizer='adam')

# class_weights = class_weight.compute_class_weight("balanced", classes_to_predict, birdcall_df.bird.values)
# class_weights_dict = {i: class_weights[i] for i, label in enumerate(classes_to_predict)}

train_generator = data_augmentation(train_df, TARGET_SIZE, TRAINING_BATCH_SIZE, True)
validation_generator = data_augmentation(valid_df, TARGET_SIZE, VALIDATION_BATCH_SIZE, False)
test_generator = data_augmentation(test_df, TARGET_SIZE, VALIDATION_BATCH_SIZE, False)

history = model.fit(train_generator,
                    epochs=20,
                    validation_data=validation_generator,
                    callbacks=callbacks)

plot_hist_loss(history)    

generated_predictions = model.predict_generator(test_generator)

test_results_df = pd.DataFrame(columns=["prediction", "groundtruth"])

for pred, ground_truth in zip(generated_predictions, test_generator.__getitem__(0)[1]):
    test_results_df = test_results_df.append(
        {"prediction": classes_to_predict[np.argmax(pred)],
         "groundtruth": classes_to_predict[np.argmax(ground_truth)],
         "correct_prediction": np.argmax(pred) == np.argmax(ground_truth)},
        ignore_index=True)

test_results_df = test_results_df.append(
    {"prediction": classes_to_predict[np.argmax(generated_predictions, axis=1)],
     "groundtruth": classes_to_predict[test_df.bird.values]},
     ignore_index=True)


y_true = test_df.bird.values
y_predicted = np.argmax(generated_predictions, axis=1)
    
from sklearn.metric import classification_report     
print(classification_report(y_true, y_predicted))


model.load_weights("best_model1.h5")

    



def predict_submission(df, audio_file_path):
    previous_filename = ""
    audio_time_series = []
    sampling_rate = None
    sample_length = None

    for index, row in df.iterrows():

        try:
            if previous_filename == "" or previous_filename != row.audio_id:
                filename = '{}/{}.mp3'.format(audio_file_path, row.audio_id)
                audio_time_series, sampling_rate = librosa.load(filename)
                sample_length = DURATION * sampling_rate
            previous_filename = row.audio_id

            # basically allows to check if we are running the examples or the test set.
            if "site" in df.columns:
                if row.site == "site_1" or row.site == "site_2":
                    song_sample = np.array(
                        audio_time_series[int(row.seconds - 5) * sampling_rate:int(row.seconds) * sampling_rate])
                elif row.site == "site_3":
                    # for now, I only take the first 5s of the samples from site_3 as they are groundtruthed at file level
                    song_sample = np.array(audio_time_series[0:sample_length])
            else:
                # same as the first condition but I isolated it for later and it is for the example file
                song_sample = np.array(
                    audio_time_series[int(row.seconds - 5) * sampling_rate:int(row.seconds) * sampling_rate])

            predicted_bird = predict_on_melspectrogram(song_sample, sample_length)
            df.at[index, "birds"] = predicted_bird
        except:
            df.at[index, "birds"] = "nocall"
    return df


example_df = read_csv_file(BASE_DIR + EXAMPLE_TEST_AUDIO_SUMMARY)
audio_file_path = BASE_DIR + EXAMPLE_TEST_AUDIO
# Adjusting the example filenames and creating the audio_id column to match with the test file.
example_df["audio_id"] = [
    "BLKFR-10-CPL_20190611_093000.pt540" if filename == "BLKFR-10-CPL" else "ORANGE-7-CAP_20190606_093000.pt623" for
    filename in example_df["filename"]]

if os.path.exists(audio_file_path):
    example_df = predict_submission(example_df, audio_file_path)

test_file_path = BASE_DIR + TEST_AUDIO_FILE
test_df = read_csv_file(BASE_DIR + TEST_DATA)
submission_df = read_csv_file(BASE_DIR + SAMPLE_SUBMISSION_FILE)

if os.path.exists(test_file_path):
    submission_df = predict_submission(test_df, test_file_path)

submission_df[["row_id", "birds"]].to_csv('submission.csv', index=False)
