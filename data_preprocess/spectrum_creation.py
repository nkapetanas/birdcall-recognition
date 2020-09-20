# importing appropriate libraries
import pandas as pd
import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt
from glob import glob
from uuid import uuid4

import properties
import utils

import warnings
import os

warnings.filterwarnings("ignore")


#loading the data
data = pd.read_csv('../birdsong-recognition/train.csv')
train_data = data[['ebird_code' , 'filename' ]]
train_target = data['species']

data_dir = '../birdsong-recognition/train_audio/'

for row in range(len(train_data)):
    audio_file = data_dir + train_data.ebird_code[row] + '/' + train_data.filename[row]
    
    x , sr = librosa.load(audio_file)
       
    X = librosa.stft(x)
    
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(1.28, 1.28) , dpi = 100)
    librosa.display.specshow(Xdb, sr = sr, x_axis='time', y_axis='hz')
    plt.axis('off')
    save_dir = '../birdsong-recognition/output_spectro' + '/' + train_data.ebird_code[row]
   
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    flag = '%s' % str(train_data.filename[row]).partition('.')[0]
    filename = os.path.join(save_dir , flag)
    plt.savefig(filename + '.jpg' , dpi = 100)
    
    print(row)

