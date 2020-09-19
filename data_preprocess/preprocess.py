import warnings
from uuid import uuid4
import librosa
import librosa.display
import numpy as np
import sklearn
from PIL import Image
import utils
from properties import *

warnings.filterwarnings("ignore")


def get_song_samples(filename, bird_kind, output_folder):
    audio_time_series, sampling_rate = librosa.load(filename)
    audio_time_series, _ = librosa.effects.trim(audio_time_series)

    song_sliced = []
    sample_length = DURATION * sampling_rate
    samples_from_file = []

    for index in range(0, len(audio_time_series), sample_length):

        song_sliced = audio_time_series[index:index + sample_length]

        if len(song_sliced) >= sample_length:
            mel_db = utils.compute_melspectrogram(song_sliced, sampling_rate, N_MELS, F_MIN, F_MAX)
            normalised_db = sklearn.preprocessing.minmax_scale(mel_db)

            db_array = (np.asarray(normalised_db) * 255).astype(np.uint8)
            db_image = Image.fromarray(np.array([db_array, db_array, db_array]).T)

            filename = str(uuid4()) + '_' + bird_kind + '.tif'
            db_image.save("{}{}".format(output_folder, filename))

            samples_from_file.append({"song_sample": "{}{}".format(output_folder, filename),
                                      "bird": bird_kind})
    return samples_from_file


train_data = utils.read_csv_file(BASE_DIR + TRAIN_DATA)

for index, row in train_data.iterrows():
    file_path = BASE_DIR + TRAIN_AUDIO + row.ebird_code + '/' + row.filename
    get_song_samples(file_path, row.ebird_code, OUTPUT_FOLDER)
