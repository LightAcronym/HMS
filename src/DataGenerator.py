import keras
import numpy as np

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, df, mode='test', start_sec=0):
        self.df = df
        self.mode = mode
        self.start_sec = start_sec


    def __getitem__(self, index):
        'Generate one batch of data'
        X = self.__data_generation()
        return X

    def __data_generation(self):
        'Generates data containing batch_size samples'

        X = np.zeros((1, 128, 256, 4), dtype='float32')
        y = np.zeros((1, 6), dtype='float32')
        img = np.ones((128, 256), dtype='float32')

        r = self.start_sec // 2
        for k in range(4):
            img = self.df[r:r + 300, k * 100:(k + 1) * 100].T
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            m = np.nanmean(img.flatten())
            s = np.nanstd(img.flatten())
            img = (img - m) / (s + ep)
            img = np.nan_to_num(img, nan=0.0)

            X[0, 14:-14, :, k] = img[:, 22:-22]  # / 2.0

        return X
