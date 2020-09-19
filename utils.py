import librosa
import pandas as pd
import numpy as np


def read_csv_file(file_path):
    return pd.read_csv(file_path, encoding="utf-8")


def compute_melspectrogram(input_signal, sr, n_mels, f_min, f_max):
    mel_spec_result = librosa.feature.melspectrogram(input_signal, sr=sr, n_mels=n_mels, fmin=f_min, fmax=f_max)

    return librosa.power_to_db(mel_spec_result).astype(np.float32)
