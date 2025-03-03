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
    def __init__(self) -> None:
        self.config = OmegaConf.load('config.yaml')
        # Set MFCC meta parameters
        self.hop_length = self.config.audio.hop_length  # number of samples to shift
        self.n_fft = self.config.audio.n_fft  # number of samples per fft (window size)
        self.mfcc_depth = self.config.audio.mfcc_depth
        self.data_path = Path('data')
        self.commonvoice_path = Path(self.data_path, 'CommonVoice')
        self.data = None


    def download_librispeech(self, split: str) -> None:
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True)
        raw_data = torchaudio.datasets.LIBRISPEECH(root=self.data_path, url=split, download=True)


    def download_commonvoice(self, split: str) -> None:
        # raw_data = torchaudio.datasets.COMMONVOICE(root=self.data_path, tsv='train.tsv')
        # "test.tsv"
        # "dev.tsv"
        # "invalidated.tsv"
        # "validated.tsv"
        # "other.tsv"
        commonvoice_info = {
            'commonvoice-dev': {
                'default_name': "cv-corpus-20.0-delta-2024-12-06",
                'url': "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-20.0-delta-2024-12-06/cv-corpus-20.0-delta-2024-12-06-en.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250303%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250303T091447Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=7a22ef188358cda58a3af93ab49d39e6fc965d21716e8afe4200c4c258045008b8a292f360726a8cf56c8cdad33e99cce9fdd2a8df8ffb88beb89e0f8e098dc8728e5f23b5fba94a3c65facf09c60ff0e4e391980242d222046559508854be7e25b93399a1805088d3beb411f8b0aa284ca3e0a22384adb5d00a37d3a0550ee5757f247c04b8a8ef1281245ecaa48b99a0a1cfaf5e4090c8b0c2e47d2d4f36cf7ce230fd9dc2a59ca0644c2efa376803f2884a7346744bf78e695e234237a87af2b035334e51a1005404df6bf0e752e4dabe0c0054ae37aa07ccb7c9ed21f9f2bf178b67971b5b89cf9d41805cc7e1bb540746deebda9d6cf359f288169cad4d",
            },
            'commonvoice-train': {
                'default_name': "cv-corpus-20.0-2024-12-06",
                'url': "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-20.0-2024-12-06/cv-corpus-20.0-2024-12-06-en.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250303%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250303T091357Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=22783fae5e7154251295166eef97bfa485ff0328981c42b307248b9138955084e21c23ecb55a0d0e09f7fc0618474e00a3bd5fedf6360f1b11a096cafe014b3dd0cbc010414467ca0f5bbf8426535ce70b23553381ec21c3a3ec5e39fd75831cc23dd0d27030b5fa7ff9e772b91cd9cddcdfe3a118c4c7af67522f4429b9475a56f8d54954ecd94070b71c33a316c2ee1b470a19af8ee387d3053cf1da27d3dfb4a2e2ae3ae3d2720c69dfd4915849c45b58ebc6ade2267afcce927f0adaecb5266f1cfb9481808b903c1e00a65c7df121cd53d7e6212164c561927d63ff4e5b09349dc1d7de0d717c6e2ec32c876c17f1f5ba83a778e44d5294014ea98c74da" 
            }
        }

        # Download & extract dataset if data is missing
        download_path = Path(self.commonvoice_path, f'{split}.tar.gz')
        split_path = Path(self.commonvoice_path, split)
        validated_path = Path(split_path, 'en', 'validated.tsv')
        other_path = Path(split_path, 'en', 'other.tsv')

        if not validated_path.exists() or not other_path.exists():
            import subprocess

            if not download_path.exists():
                print('\nDownloading CommonVoice data...')
                if not self.commonvoice_path.exists():
                    self.commonvoice_path.mkdir(parents=True)
                subprocess.run(['curl',
                                '-o', download_path.as_posix(),
                                commonvoice_info[split]["url"]])
                print("Download complete.")

            print('\nExtracting CommonVoice data...')
            if not split_path.exists():
                split_path.mkdir(parents=True)
            subprocess.run(['tar',
                            '-xvzf', f'{split_path.as_posix()}.tar.gz',
                            '-C', self.commonvoice_path.as_posix(),
                            '-s', f'/^{commonvoice_info[split]["default_name"]}/{split}/'])
            print("Extraction complete.")

            if not validated_path.exists():
                print(f'\n{validated_path.as_posix()} does not exist. Try updating the download URL.')
                return
            if not other_path.exists():
                print(f'\n{validated_path.as_posix()} does not exist. Try updating the download URL.')
                return


    def load_librispeech(self) -> None:
        # print('Loading LibriSpeech...')
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


    def load_commonvoice(self, split: str) -> None:
        split_path = Path(self.commonvoice_path, split)
        validated_path = Path(split_path, 'en', 'validated.tsv')
        other_path = Path(split_path, 'en', 'other.tsv')

        # Load data
        print('Loading CommonVoice...')
        validated_raw_data = pd.read_csv(validated_path, sep='\t')
        other_raw_data = pd.read_csv(other_path, sep='\t')
        data = []
        progress_bar = ProgressBar()
        for index, item in enumerate(validated_raw_data.itertuples()):
            samples, sample_rate = librosa.load(Path(split_path, 'en', 'clips', item.path))
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
            progress_bar.update(index + 1, len(validated_raw_data) + len(other_raw_data))
        for index, item in enumerate(other_raw_data.itertuples()):
            samples, sample_rate = librosa.load(Path(split_path, 'en', 'clips', item.path))
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
            progress_bar.update(index + 1, len(validated_raw_data) + len(other_raw_data))
        self.data = data


    def load_librispeech_item(self, item) -> dict:
        return {
            'samples': item[0].numpy(),  # torch.Tensor
            'sample_rate': item[1],
            'transcript': item[2],
            'speaker_id': item[3],
            'chapter_id': item[4],
            'sentence_id': item[5],
        }


    def load_commonvoice_item(self, item, split: str) -> dict:
        samples, sample_rate = librosa.load(Path(self.commonvoice_path, split, 'en', 'clips', item.path))
        return {
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
        }


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


    def preprocess(self, split: str = SPLITS.DEV_CLEAN.value) -> None:
        '''
            Arguments
            - dataset_url: Name of dataset split, check SPLITS enum for options
        '''

        if split not in [item.value for item in SPLITS]:
            raise ValueError(f'Invalid split name "{split}". Check splits.py for options.')
        print('\nDataset split:', split)
        print('\n=== Config ===\n')
        print('Hop length:', self.hop_length)
        print('Samples per MFCC:', self.n_fft)
        print('MFCC depth:', self.mfcc_depth)
        is_commonvoice = split == SPLITS.COMMONVOICE_DEV.value or split == SPLITS.COMMONVOICE_TRAIN.value
        split_mfcc_path = Path('mfcc', split)
        if not split_mfcc_path.exists():
            split_mfcc_path.mkdir(parents=True)


        # Download split data
        if is_commonvoice:
            self.download_commonvoice(split=split)
        else:
            self.download_librispeech(split=split)


        # Load split metadata
        print(f'\nProcessing {split}...')
        raw_data = None
        if is_commonvoice:
            validated_raw_data = pd.read_csv(Path(self.commonvoice_path, split, 'en', 'validated.tsv'), sep='\t')
            other_raw_data = pd.read_csv(Path(self.commonvoice_path, split, 'en', 'other.tsv'), sep='\t')
            raw_data = pd.concat([validated_raw_data, other_raw_data])
            raw_data = raw_data.itertuples()
            raw_data_length = len(validated_raw_data) + len(other_raw_data)
        else:
            raw_data = torchaudio.datasets.LIBRISPEECH(root=self.data_path, url=split, download=True)
            raw_data_length = len(raw_data)


        if len(list(split_mfcc_path.glob('*.hdf5'))) == raw_data_length:
            print('Preprocessed data already exists.')
            return


        # Load & process each data item, one at a time
        progress_bar = ProgressBar()
        for index, item in enumerate(raw_data):
            if is_commonvoice:
                data = self.load_commonvoice_item(item, split=split)
            else:
                data = self.load_librispeech_item(item)
            self.write_mfcc_data(data, split=split)
            progress_bar.update(index + 1, raw_data_length)

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

    preprocessor = Preprocessor()

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
        preprocessor.preprocess(split=args.split)


if __name__ == '__main__':
    main()
