import argparse
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import librosa, librosa.display
from pathlib import Path
import sounddevice as sd
import h5py

from splits import SPLITS
from config import Config
from util import ProgressBar


class Preprocessor():
    '''
    Arguments
    - dataset_url: Name of dataset split, check SPLITS enum for options
    '''
    def __init__(self, split_train: str=SPLITS.DEV_CLEAN.value, split_test: str=SPLITS.TEST_CLEAN.value):
        # Set MFCC meta parameters
        self.hop_length = Config.HOP_LENGTH  # number of samples to shift
        self.n_fft = Config.N_FFT  # number of samples per fft (window size)
        self.mfcc_depth = Config.MFCC_DEPTH

        self.data_train = None
        self.data_test = None
        self.split_train = split_train
        self.split_test = split_test
        print('Dataset train split:', split_train)
        print('Dataset test split:', split_test)
        self.download_dataset()

    def download_dataset(self):
        if self.split_train not in [item.value for item in SPLITS] or self.split_test not in [item.value for item in SPLITS]:
            print('Invalid train or test split name. Check splits.py for options.')
            return
        data_path = Path('data')
        if not data_path.exists():
            data_path.mkdir(parents=True)
        # Updated dataset from LibriLightLimited -> LibriSpeech (same format)
        self.data_train = torchaudio.datasets.LIBRISPEECH(root=data_path, url=self.split_train, download=True)
        self.data_test = torchaudio.datasets.LIBRISPEECH(root=data_path, url=self.split_test, download=True)

    def show_test_data(self, index: int=0, waveform=False, spectrogram=False, mfcc=False, play=False):
        if self.data_train is None:
            return
        test = self.data_train.__getitem__(index)
        samples, sample_rate, transcript, speaker_id, chapter_id, utterance_id = test
        samples = samples.numpy()

        print('-- Sample Data --')
        print(f'{samples=}')
        print(f'samples shape: {np.shape(samples)}')
        print(f'{sample_rate=}')
        print(f'{transcript=}')
        print(f'{speaker_id=}')
        print(f'{chapter_id=}')
        print(f'{utterance_id=}')
        duration = len(samples[0]) / sample_rate
        print(f'{duration=}')

        # # frames = # samples / hop length
        # Shape after stft: (# channels, 1 + (n_fft / 2), # frames)

        if play is True:
            # need to transpose array from one row [1][n] to one column [n, 1] (one channel)
            sd.play(data=samples.T, samplerate=sample_rate)
            sd.wait()

        if waveform is True:
            librosa.display.waveshow(samples, sr=sample_rate)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()

        if spectrogram is True:
            stft = librosa.core.stft(samples, hop_length=self.hop_length, n_fft=self.n_fft)
            stft = stft[0]  # Remove first dimension (mono channel)
            spectrogram = np.abs(stft)
            log_spectrogram = librosa.amplitude_to_db(spectrogram)
            librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=self.hop_length)
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.colorbar()
            plt.show()

        if mfcc is True:
            # Same as stft, remove first dimension
            mfccs = librosa.feature.mfcc(y=samples, n_fft=self.n_fft, hop_length=self.hop_length, n_mfcc=self.mfcc_depth)
            mfccs = mfccs[0]  # Remove first dimension (mono channel)
            librosa.display.specshow(mfccs, sr=sample_rate, hop_length=self.hop_length)
            plt.xlabel('Time')
            plt.ylabel('MFCC')
            plt.colorbar()
            plt.show()

    def process_audio(self, item, split: str):
        samples, sample_rate, transcript, speaker_id, chapter_id, utterance_id = item

        mfccs = librosa.feature.mfcc(y=samples.numpy(), n_fft=self.n_fft, hop_length=self.hop_length, n_mfcc=self.mfcc_depth)
        mfcc_bands = mfccs[0]
        mfcc_frames = mfcc_bands.T  # (num_bands, num_frames) -> (num_frames, num_bands)

        with h5py.File(Path('mfcc', split, f'{speaker_id}_{chapter_id}_{utterance_id}.hdf5'), 'w') as file:
            dataset = file.create_dataset('mfccs', data=mfcc_frames)
            dataset.attrs['speaker_id'] = speaker_id
            dataset.attrs['chapter_id'] = chapter_id
            dataset.attrs['utterance_id'] = utterance_id
            dataset.attrs['sample_rate'] = sample_rate
            dataset.attrs['transcript'] = transcript

    def preprocess(self):
        if self.data_train is None or self.data_test is None:
            print('Data is empty. Aborted.')
            return

        train_path = Path('mfcc', self.split_train)
        test_path = Path('mfcc', self.split_test)

        print('Hop length:', self.hop_length)
        print('Samples per MFCC:', self.n_fft)
        print('MFCC depth:', self.mfcc_depth)
        print('Processing audio data...')

        print('Train data:')
        if not train_path.exists():
            train_path.mkdir(parents=True)
        if len(list(train_path.glob('*.hdf5'))) == 0:
            progress_bar = ProgressBar()
            for x, item in enumerate(self.data_train):
                self.process_audio(item, split=self.split_train)
                progress_bar.update(x + 1, len(self.data_train))

        print('Test data:')
        if not test_path.exists():
            test_path.mkdir(parents=True)
        if len(list(test_path.glob('*.hdf5'))) == 0:
            progress_bar = ProgressBar()
            for x, item in enumerate(self.data_test):
                self.process_audio(item, split=self.split_test)
                progress_bar.update(x + 1, len(self.data_test))
        
        print('Preprocessing finished.')

    def read_preprocessed_data(self):
        files = list(Path('mfcc', self.split_train).glob('*.hdf5'))
        for file in files[:5]:
            with h5py.File(file, 'r') as file_data:
                mfccs_dataset = file_data['mfccs']
                print()
                print(mfccs_dataset)
                for attr in list(mfccs_dataset.attrs):
                    print(f'{attr}: {mfccs_dataset.attrs[attr]}')

def main():
    parser = argparse.ArgumentParser(
        prog='S2T Preprocessor',
        description='Preprocess audio for the S2T transformer neural network',
        epilog='Epilogue sample text')

    default_split_train = Config.SPLIT_TRAIN if Config.SPLIT_TRAIN is not None else SPLITS.TRAIN_CLEAN_100.value
    default_split_test = Config.SPLIT_TEST if Config.SPLIT_TEST is not None else SPLITS.TEST_CLEAN.value
    parser.add_argument('--split_train', type=str, nargs='?', default=default_split_train, help='Name of dataset split for training')
    parser.add_argument('--split_test', type=str, nargs='?', default=default_split_test, help='Name of dataset split for testing (validation)')
    parser.add_argument('-d', '--display', type=int, default=-1, help='Index of one data sample to display')
    parser.add_argument('-w', '--waveform', action='store_true', help='Display waveform')
    parser.add_argument('-s', '--spectrogram', action='store_true', help='Display spectrogram')
    parser.add_argument('-m', '--mfcc', action='store_true', help='Display MFCCs')
    parser.add_argument('-p', '--play', action='store_true', help='Play audio file')
    parser.add_argument('-r', '--read-mfcc', action='store_true', help='Read preprocessed mfcc data')

    args = parser.parse_args()

    preprocessor = Preprocessor(split_train=args.split_train, split_test=args.split_test)

    if args.display != -1:
        preprocessor.show_test_data(
            index=args.display,
            waveform=True if args.waveform else False,
            spectrogram=True if args.spectrogram else False,
            mfcc=True if args.mfcc else False,
            play=True if args.play else False)
    elif args.read_mfcc:
        preprocessor.read_preprocessed_data()
    else:
        preprocessor.preprocess()

if __name__ == '__main__':
    main()