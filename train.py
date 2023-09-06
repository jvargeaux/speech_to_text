import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import librosa, librosa.display
from pathlib import Path
import sounddevice as sd


class SpeechToTextRNN():
    def __init__(self):
        # Set device, use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {self.device}')

        self.download_dataset()

    def download_dataset(self):
        print('downloading dataset...')
        data_path = Path('data')
        if not data_path.exists():
            data_path.mkdir(parents=True)
        data = torchaudio.datasets.LibriLightLimited(root=data_path, download=True)
        self.data = data
    
    def show_test_data(self, waveform=False, spectrogram=False, mfcc=False, play=False):
        if self.data is None:
            return
        test = self.data.__getitem__(0)  # maybe this is questionable?
        samples = test[0].numpy()
        sr = test[1]

        print('-- Showing Test Data --')
        print(test)
        print(f'{samples=}')
        print(np.shape(samples))
        print(f'{sr=}')
        duration = librosa.get_duration(y=samples, sr=sr)
        print(f'{duration=}')

        hop_length = 512  # number of samples to shift
        n_fft = 2048  # number of samples per fft (window size)
        n_mfcc = 13  # standard minimum

        # Samples in test file: 175,920
        # Hop length: 512
        # Number of frames: 175,920 / 512 = 343.59 (344)
        # Shape after stft: (number of channels, 1 + (n_fft / 2), number of frames)

        if play is True:
            # need to transpose array from one row [1][n] to one column [n, 1] (one channel)
            sd.play(data=samples.T, samplerate=sr)
            # sd.wait()

        if waveform is True:
            librosa.display.waveshow(samples, sr=sr)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()

        if spectrogram is True:
            # Fixed!
            # Shape changes after stft: (1, 175920) -> (1, 1025, 344)
            # We don't need num channels, since audio is mono.
            # Remove first dimension: (1, 1025, 344) -> (1025, 344)
            # Should work now with no error.
            stft = librosa.core.stft(samples, hop_length=hop_length, n_fft=n_fft)
            stft = stft[0]
            print('stft shape')
            print(np.shape(stft))

            spectrogram = np.abs(stft)
            log_spectrogram = librosa.amplitude_to_db(spectrogram)
            print('spectrogram shape')
            print(np.shape(log_spectrogram))

            librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.colorbar()
            plt.show()

        if mfcc is True:
            # Same as stft, remove first dimension
            mfccs = librosa.feature.mfcc(y=samples, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
            mfccs = mfccs[0]
            print('mfccs shape')
            print(np.shape(mfccs))
            librosa.display.specshow(mfccs, sr=sr, hop_length=hop_length)
            plt.xlabel('Time')
            plt.ylabel('MFCC')
            plt.colorbar()
            plt.show()

    def preprocess_data(self):
        # Take random subset
        # subset = get_random_subset(self.data)

        # For now, take first 10 elements as a test
        subset = list(self.data)[0:10]

        hop_length = 512  # number of samples to shift
        n_fft = 2048  # number of samples per fft (window size)
        n_mfcc = 13  # standard minimum

        for i, item in enumerate(subset):
            samples, sample_rate, transcript, speaker_id, chapter_id, utterance_id = item
            samples = samples.numpy()

            mfccs = librosa.feature.mfcc(y=samples, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
            mfccs = mfccs[0]

            print('mfccs')
            print(mfccs)

            # Export MFCCs along with other features to file for training


def main():
    args = sys.argv[1:]
    rnn = SpeechToTextRNN()
    if '--show-data' in args:
        rnn.show_test_data(
            waveform=True if '--waveform' in args else False,
            spectrogram=True if '--spectrogram' in args else False,
            mfcc=True if '--mfcc' in args else False,
            play=True if '--play' in args else False)
    if '--preprocess' in args:
        rnn.preprocess_data()
        

if __name__ == '__main__':
    main()


def doc_test(blarg: str):
    """
    :param blarg: A random string
    :return: Returns the string 'yo'

    Subtitle 1
    ----------
    What's upppppp

    - List 1
    - Item 2

    >>> import test.blarg as test
    """
    librosa.core.st

    return 'yo'