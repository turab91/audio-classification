import os
import numpy as np

class Config:
    """
    Contains configuration for initial pre-processing
    """
    def __init__(self, file_path, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000, winfunc=np.hamming):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate / 10)          # sample size 1/10th of a second
        self.winfunc = winfunc
        self.winlen = 0.025            # 25 ms
        self.winstep = 0.01               # 10 ms

        self.file_path = file_path
        self.raw_data_path = os.path.join(self.file_path, 'wavfiles')
        self.clean_data_path = os.path.join(self.file_path, 'cleanfiles')
        self.model_path = os.path.join(self.file_path, 'models', mode + '.model')
        self.pickle_path = os.path.join(self.file_path, 'pickles', mode + '.p')
        self.image_path = os.path.join(self.file_path, 'images')