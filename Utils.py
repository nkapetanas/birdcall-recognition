import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
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


def plot_loss_over_epoch(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss over epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.show()

def create_plot_Keras_model(history, field):
    plt.plot(history.history[field])
    plt.title('model ' + field)
    plt.ylabel(field)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
