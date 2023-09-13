import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import librosa, librosa.display
from pathlib import Path
import sounddevice as sd
import h5py
import os


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


class SpeechToTextRNN():
    def __init__(self):
        # Set device, use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {self.device}')

        # Set MFCC meta parameters
        self.hop_length = 512  # number of samples to shift
        self.n_fft = 2048  # number of samples per fft (window size)
        self.n_mfcc = 13  # standard minimum

        self.download_dataset()

    def download_dataset(self):
        data_path = Path('data')
        if not data_path.exists():
            data_path.mkdir(parents=True)
        data = torchaudio.datasets.LibriLightLimited(root=data_path, download=True)
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
        # duration = librosa.get_duration(y=samples, sr=sample_rate)
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

    def preprocess(self):
        # For now, take first 10 elements as a test
        subset = list(self.data)[0:10]

        mfcc_path = Path('mfcc')
        if not mfcc_path.exists():
            mfcc_path.mkdir(parents=True)
        
        print('Processing audio data...')
        progress_bar = ProgressBar()
        for i, item in enumerate(subset):
            samples, sample_rate, transcript, speaker_id, chapter_id, utterance_id = item

            mfccs = librosa.feature.mfcc(y=samples.numpy(), n_fft=self.n_fft, hop_length=self.hop_length, n_mfcc=self.n_mfcc)
            mfcc_bands = mfccs[0]
            mfcc_frames = mfcc_bands.T  # (num_bands, num_frames) -> (num_frames, num_bands)

            with h5py.File(f'mfcc/{speaker_id}_{chapter_id}_{utterance_id}.hdf5', 'w') as file:
                dataset = file.create_dataset('mfccs', data=mfcc_frames)
                print(dataset)
                dataset.attrs['speaker_id'] = speaker_id
                dataset.attrs['chapter_id'] = chapter_id
                dataset.attrs['utterance_id'] = utterance_id
                dataset.attrs['sample_rate'] = sample_rate

            progress_bar.update(i + 1, len(subset))
        
    def read_preprocessed_data(self):
        for root, directories, files in os.walk('mfcc'):
            for file in files:
                with h5py.File(f'mfcc/{file}', 'r') as file_data:
                    mfccs_dataset = file_data['mfccs']
                    print(mfccs_dataset)
                    for attr in list(mfccs_dataset.attrs):
                        print(f'{attr}: {mfccs_dataset.attrs[attr]}')

    def train(self):
        # Load data (multiprocessing)

        # Architecture (LSTM -> GRU)
        # - GRU Layer (64)
        # - GRU Layer (64)
        # - Dense Layer
        # - Dropout Layer (to prevent overfitting)
        # - Dense Layer (softmax output)
        pass

def main():
    args = sys.argv[1:]
    rnn = SpeechToTextRNN()
    if '--display' in args:
        rnn.show_test_data(
            waveform=True if '--waveform' in args else False,
            spectrogram=True if '--spectrogram' in args else False,
            mfcc=True if '--mfcc' in args else False,
            play=True if '--play' in args else False)
    if '--preprocess' in args:
        rnn.preprocess()
    if '--read-mfcc' in args:
        rnn.read_preprocessed_data()

if __name__ == '__main__':
    main()