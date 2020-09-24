import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from keras import backend as K

from properties import *


def read_csv_file(file_path):
    return pd.read_csv(file_path, encoding="utf-8")


def compute_melspectrogram(input_signal, sr, n_mels, f_min, f_max):
    mel_spec_result = librosa.feature.melspectrogram(input_signal, sr=sr, n_mels=n_mels, fmin=f_min, fmax=f_max)

    return librosa.power_to_db(mel_spec_result).astype(np.float32)


def predict_on_melspectrogram(song_sample, sampling_rate, sample_length, model, classes_to_predict):
    mel = compute_melspectrogram(song_sample, sampling_rate, N_MELS, F_MIN, F_MAX)

    db = librosa.power_to_db(mel)

    normalised_db = sklearn.preprocessing.minmax_scale(db)

    db_array = (np.asarray(normalised_db) * 255).astype(np.uint8)

    prediction = model.predict(np.array([np.array([db_array, db_array, db_array]).T]))
    predicted_bird = classes_to_predict[np.argmax(prediction)]
    return predicted_bird


def recall_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

def plot_hist_loss(hist):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model accuracy")
    plt.ylabel("Loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

def plot_loss_over_epoch(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss over epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.show()


def create_plot_train_test_model(history, field):
    plt.plot(history.history[field])
    plt.title('model ' + field)
    plt.ylabel(field)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
