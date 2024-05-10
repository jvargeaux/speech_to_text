import argparse
from pathlib import Path

import h5py
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sounddevice as sd
import torchaudio
from omegaconf import OmegaConf

from splits import SPLITS
from util import ProgressBar


class Preprocessor:
    '''
    Arguments
    - dataset_url: Name of dataset split, check SPLITS enum for options
    '''
    def __init__(self, split: str = SPLITS.DEV_CLEAN.value) -> None:
        self.config = OmegaConf.load('config.yaml')
        if split not in [item.value for item in SPLITS]:
            raise ValueError(f'Invalid split name "{split}". Check splits.py for options.')
        print('Dataset split:', split)
        # Set MFCC meta parameters
        self.hop_length = self.config.audio.hop_length  # number of samples to shift
        self.n_fft = self.config.audio.n_fft  # number of samples per fft (window size)
        self.mfcc_depth = self.config.audio.mfcc_depth
        self.data_path = Path('data')
        self.split = split
        self.data = None
        if split == SPLITS.COMMONVOICE_DEV.value:
            self.load_commonvoice()
        else:
            self.load_librispeech()


    def load_librispeech(self) -> None:
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True)
        print('Loading LibriSpeech...')
        raw_data = torchaudio.datasets.LIBRISPEECH(root=self.data_path, url=self.split, download=True)
        progress_bar = ProgressBar()
        data = []
        for index, item in enumerate(raw_data):
            data.append({
                'samples': item[0].numpy(),  # torch.Tensor
                'sample_rate': item[1],
                'transcript': item[2],
                'speaker_id': item[3],
                'chapter_id': item[4],
                'sentence_id': item[5],
            })
            progress_bar.update(index + 1, len(raw_data))
        self.data = data


    def load_commonvoice(self) -> None:
        commonvoice_path = Path(self.data_path, 'CommonVoice', 'cv-corpus-17.0-delta-2024-03-15', 'en')
        validated_path = Path(commonvoice_path, 'validated.tsv')
        if not validated_path.exists():
            print('CommonVoice data does not exist.')
            return
        print('Loading CommonVoice...')
        raw_data = pd.read_csv(validated_path, sep='\t')
        data = []
        progress_bar = ProgressBar()
        for index, item in enumerate(raw_data.itertuples()):
            samples, sample_rate = librosa.load(Path(commonvoice_path, 'clips', item.path))
            data.append({
                'samples': np.expand_dims(samples, axis=0),
                'sample_rate': sample_rate,
                'transcript': item.sentence,
                'speaker_id': item.client_id,
                'chapter_id': 0,
                'sentence_id': item.sentence_id,
                'age': item.age,
                'gender': item.gender,
                'accents': item.accents,
                'variant': item.variant,
                'locale': item.locale,
            })
            progress_bar.update(index + 1, len(raw_data))
        self.data = data


    def output_data_sample(self, index: int = 0, waveform: bool = False, spectrogram: bool = False,
                           mfcc: bool = False, play: bool = False) -> None:
        if self.data is None:
            return
        test: dict = self.data[index]
        samples = test['samples']

        print('-- Sample Data --')
        print(f'{samples=}')
        print(f'samples shape: {np.shape(samples)}')
        print(f'{test["sample_rate"]=}')
        print(f'{test["transcript"]=}')
        print(f'{test["speaker_id"]=}')
        print(f'{test["chapter_id"]=}')
        print(f'{test["sentence_id"]=}')
        duration = len(samples[0]) / test['sample_rate']
        print(f'{duration=}')

        # # frames = # samples / hop length
        # Shape after stft: (# channels, 1 + (n_fft / 2), # frames)

        if play is True:
            # need to transpose array from one row [1][n] to one column [n, 1] (one channel)
            sd.play(data=samples.T, samplerate=test['sample_rate'])
            sd.wait()

        if waveform is True:
            librosa.display.waveshow(samples, sr=test['sample_rate'])
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()

        if spectrogram is True:
            stft = librosa.core.stft(samples, hop_length=self.hop_length, n_fft=self.n_fft)
            stft = stft[0]  # Remove first dimension (mono channel)
            spectrogram = np.abs(stft)
            log_spectrogram = librosa.amplitude_to_db(spectrogram)
            librosa.display.specshow(log_spectrogram, sr=test['sample_rate'], hop_length=self.hop_length)
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.colorbar()
            plt.show()

        if mfcc is True:
            # Same as stft, remove first dimension
            mfccs = librosa.feature.mfcc(y=samples, n_fft=self.n_fft, hop_length=self.hop_length, n_mfcc=self.mfcc_depth)
            mfccs = mfccs[0]  # Remove first dimension (mono channel)
            librosa.display.specshow(mfccs, sr=test['sample_rate'], hop_length=self.hop_length)
            plt.xlabel('Time')
            plt.ylabel('MFCC')
            plt.colorbar()
            plt.show()


    def write_mfcc_data(self, item: dict, split: str) -> None:
        mfccs = librosa.feature.mfcc(y=item['samples'], n_fft=self.n_fft, hop_length=self.hop_length, n_mfcc=self.mfcc_depth)
        mfcc_bands = mfccs[0]
        mfcc_frames = mfcc_bands.T  # (num_bands, num_frames) -> (num_frames, num_bands)

        with h5py.File(Path('mfcc', split, f"{item['speaker_id']}_{item['chapter_id']}_{item['sentence_id']}.hdf5"), 'w') as file:
            dataset = file.create_dataset('mfccs', data=mfcc_frames)
            dataset.attrs['speaker_id'] = item['speaker_id']
            dataset.attrs['chapter_id'] = item['chapter_id']
            dataset.attrs['sentence_id'] = item['sentence_id']
            dataset.attrs['sample_rate'] = item['sample_rate']
            dataset.attrs['transcript'] = item['transcript']


    def preprocess(self) -> None:
        if self.data is None:
            print('Data is empty. Aborted.')
            return

        mfcc_path = Path('mfcc', self.split)
        print('Hop length:', self.hop_length)
        print('Samples per MFCC:', self.n_fft)
        print('MFCC depth:', self.mfcc_depth)
        print('Processing audio data...')
        if not mfcc_path.exists():
            mfcc_path.mkdir(parents=True)
        if len(list(mfcc_path.glob('*.hdf5'))) == 0:
            progress_bar = ProgressBar()
            for x, item in enumerate(self.data):
                self.write_mfcc_data(item, split=self.split)
                progress_bar.update(x + 1, len(self.data))
        print('Preprocessing finished.')


    def read_preprocessed_data(self) -> None:
        files = list(Path('mfcc', self.split_train).glob('*.hdf5'))
        for file in files[:5]:
            with h5py.File(file, 'r') as file_data:
                mfccs_dataset = file_data['mfccs']
                print()
                print(mfccs_dataset)
                for attr in list(mfccs_dataset.attrs):
                    print(f'{attr}: {mfccs_dataset.attrs[attr]}')


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='S2T Preprocessor',
        description='Preprocess audio for the S2T transformer neural network',
        epilog='Epilogue sample text')

    parser.add_argument('-s', '--split', type=str, nargs='?', help='Name of dataset split')
    parser.add_argument('-d', '--display', type=int, default=-1, help='Index of one data sample to display')
    parser.add_argument('-w', '--waveform', action='store_true', help='Display waveform')
    parser.add_argument('-g', '--spectrogram', action='store_true', help='Display spectrogram')
    parser.add_argument('-m', '--mfcc', action='store_true', help='Display MFCCs')
    parser.add_argument('-p', '--play', action='store_true', help='Play audio file')
    parser.add_argument('-r', '--read-mfcc', action='store_true', help='Read preprocessed mfcc data')

    args = parser.parse_args()

    preprocessor = Preprocessor(split=args.split)

    if args.display != -1:
        preprocessor.output_data_sample(
            index=args.display,
            waveform=bool(args.waveform),
            spectrogram=bool(args.spectrogram),
            mfcc=bool(args.mfcc),
            play=bool(args.play))
    elif args.read_mfcc:
        preprocessor.read_preprocessed_data()
    else:
        preprocessor.preprocess()


if __name__ == '__main__':
    main()
