import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import librosa, librosa.display
from pathlib import Path
import sounddevice as sd
import h5py
import os
import multiprocessing as mp
import time
from enum import Enum


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
device = torch.device(device)


class SPLITS(Enum):
    DEV_CLEAN = 'dev-clean'
    DEV_OTHER = 'dev-other'
    TRAIN_CLEAN_100 = 'train-clean-100'
    TRAIN_CLEAN_360 = 'train-clean-360'
    TRAIN_OTHER_500 = 'train-other-500'
    TEST_CLEAN = 'test-clean'
    TEST_OTHER = 'test-other'


class ProgressBar():
    def __init__(self):
        self.length = 30
        self.fill = '█'
        self.empty = '-'
        print(f'Progress: {self.empty * self.length} | 0/0 | {0:.1f}%', end='\r')

    def update(self, filled, total):
        percentage = min([filled, total]) / total
        filled_length = int(percentage * self.length)
        empty_length = self.length - filled_length
        print(f'Progress: {self.fill * filled_length}{self.empty * empty_length} | {filled}/{total} | {(percentage * 100):.1f}%', end='\r')
        if percentage >= 1:
            print()


class SpeechToTextTrainer():
    '''
    Arguments
    - dataset_url: Name of dataset split, check SPLITS enum for options
    '''
    def __init__(self, dataset_url: str = 'DEV_CLEAN'):
        # Set MFCC meta parameters
        self.hop_length = 512  # number of samples to shift
        self.n_fft = 2048  # number of samples per fft (window size)
        self.n_mfcc = 13  # standard minimum

        self.data = None
        self.download_dataset(url=dataset_url)

    def download_dataset(self, url: str):
        if url not in [item.value for item in SPLITS]:
            print('Invalid dataset url. Check SPLITS enum for options.')
            return
        data_path = Path('data')
        if not data_path.exists():
            data_path.mkdir(parents=True)
        # Updated dataset from LibriLightLimited -> LibriSpeech (same format)
        data = torchaudio.datasets.LIBRISPEECH(root=data_path, url=url, download=True)
        self.data = data

    def show_test_data(self, waveform=False, spectrogram=False, mfcc=False, play=False):
        if self.data is None:
            return
        test = self.data.__getitem__(0)
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
            mfccs = librosa.feature.mfcc(y=samples, n_fft=self.n_fft, hop_length=self.hop_length, n_mfcc=self.n_mfcc)
            mfccs = mfccs[0]  # Remove first dimension (mono channel)
            librosa.display.specshow(mfccs, sr=sample_rate, hop_length=self.hop_length)
            plt.xlabel('Time')
            plt.ylabel('MFCC')
            plt.colorbar()
            plt.show()

    def process_audio(self, item):
        samples, sample_rate, transcript, speaker_id, chapter_id, utterance_id = item

        mfccs = librosa.feature.mfcc(y=samples.numpy(), n_fft=self.n_fft, hop_length=self.hop_length, n_mfcc=self.n_mfcc)
        mfcc_bands = mfccs[0]
        mfcc_frames = mfcc_bands.T  # (num_bands, num_frames) -> (num_frames, num_bands)

        with h5py.File(f'mfcc/{speaker_id}_{chapter_id}_{utterance_id}.hdf5', 'w') as file:
            dataset = file.create_dataset('mfccs', data=mfcc_frames)
            dataset.attrs['speaker_id'] = speaker_id
            dataset.attrs['chapter_id'] = chapter_id
            dataset.attrs['utterance_id'] = utterance_id
            dataset.attrs['sample_rate'] = sample_rate
            dataset.attrs['transcript'] = transcript

    def preprocess(self, multiprocess=False):
        if self.data is None:
            print('Data is empty. Aborted.')
            return
        mfcc_path = Path('mfcc')
        if not mfcc_path.exists():
            mfcc_path.mkdir(parents=True)
        print('Processing audio data...')

        if multiprocess is True:
            num_processes = mp.cpu_count() - 1 or 1
            print('Using number of processes:', num_processes)
            pool = mp.Pool(processes=num_processes)
            progress_bar = ProgressBar()
            for i, _ in enumerate(pool.imap_unordered(self.process_audio, self.data)):
                progress_bar.update(i + 1, len(self.data))
        else:
            print('Using single process.')
            progress_bar = ProgressBar()
            for x, item in enumerate(self.data):
                self.process_audio(item)
                progress_bar.update(x + 1, len(self.data))

    def read_preprocessed_data(self):
        for root, directories, files in os.walk('mfcc'):
            for file in files:
                with h5py.File(f'mfcc/{file}', 'r') as file_data:
                    mfccs_dataset = file_data['mfccs']
                    print(mfccs_dataset)
                    for attr in list(mfccs_dataset.attrs):
                        print(f'{attr}: {mfccs_dataset.attrs[attr]}')

    def collate(self, batch):
        # Pad batches?
        return batch

        # sample_batch, target_batch = [], []
        # for sample, target in batch:
        #     sample_batch.append(sample)
        #     target_batch.append(target)

        # padded_batch = pad_sequence(sample_batch, batch_first=True)
        # padded_to = list(padded_batch.size())[1]
        # padded_batch = padded_batch.reshape(len(sample_batch), padded_to, 1)

        # return padded_batch, torch.cat(target_batch, dim=0).reshape(len(sample_batch))


def main():
    print(f'device: {device}')
    args = sys.argv[1:]

    dataset_url = SPLITS.DEV_CLEAN.value
    print('Dataset split:', dataset_url)
    trainer = SpeechToTextTrainer(dataset_url=dataset_url)

    if '--display' in args:
        trainer.show_test_data(
            waveform=True if '--waveform' in args else False,
            spectrogram=True if '--spectrogram' in args else False,
            mfcc=True if '--mfcc' in args else False,
            play=True if '--play' in args else False)
    if '--preprocess' in args:
        trainer.preprocess(multiprocess=True if '--multiprocess' in args else False)
    if '--read-mfcc' in args:
        trainer.read_preprocessed_data()
    if '--train' in args:
        trainer.train()

if __name__ == '__main__':
    main()