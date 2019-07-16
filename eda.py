from python_speech_features import mfcc
from audioNet.utils import plot_signals, plot_mfccs, load_sound, read_csv, get_class_dist
from audioNet.utils import envelope, clean_save_data
from audioNet.parameters import Config
import matplotlib.pyplot as plt

"""
    1. handle silence  at beginning and end
    2. downsample the audio to 16KHz
"""

def get_mfccs(config: object) -> tuple:
    """
    Preprocess the signal and calculate the mfcc
    :param config:
    :return: signals, mfccs
    """
    signals = {}
    mfccs = {}
    for c in config.classes:
        # plot the first file of each class
        wav_file = df[df.label == c].iloc[0, 0]
        rate, signal = load_sound(config.raw_data_path, wav_file)
        # Remove silent regions
        mask = envelope(signal, rate, 0.0005)
        signal = signal[mask]
        signals[c] = signal

        # 1 sec of data
        mel = mfcc(signal=signal[:rate], winlen=config.winlen, winstep=config.winstep, winfunc=config.winfunc,
                   samplerate=rate, numcep=13, nfilt=26, nfft=2048).T
        mfccs[c] = mel

    return signals, mfccs


if __name__ == "__main__":
    file_path = '.'
    config = Config(file_path=file_path, mode='conv')

    # Read csv file
    df = read_csv(config.file_path, 'instruments.csv')
    print(df.head())

    classes, class_dist = get_class_dist(df, config.raw_data_path)
    config.classes = classes
    config.class_dist = class_dist

    print(f'\nclasses: \n{classes}')
    print(f'\nclass_dist: \n{class_dist}')

    signals, mfccs = get_mfccs(config)

    plot_signals(config, signals)
    plt.show()

    plot_mfccs(config, mfccs)
    plt.show()

    # applies envelope,  downsample data,  and save  to new folder 'cleanfiles'
    clean_save_data(df, config, sr=16000)

