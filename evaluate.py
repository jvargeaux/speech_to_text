import argparse
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import torch
from src.transformer import Transformer
from src.vocabulary import Vocabulary
from config import config


def main():
    parser = argparse.ArgumentParser(
             prog='S2T Evaluator',
             description='Evaluate the S2T model',
             epilog='Epilogue sample text')

    parser.add_argument('files', type=str, nargs='+', help='Audio file to evaluate')
    args = parser.parse_args()

    # Preprocess
    processed_files = []
    for file in args.files:
        input_audio_path = Path(file)
        model_path = Path('models/2023_10_13_06_13_56')

        # NOTE: Resample

        hop_length = 512  # number of samples to shift
        n_fft = 2048  # number of samples per fft (window size)
        n_mfcc = 13  # standard minimum

        # Preprocess audio
        samples, sr = sf.read(input_audio_path)
        mfccs = librosa.feature.mfcc(y=samples, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
        mfccs = mfccs.T  # (num_bands, num_frames) -> (num_frames, num_bands)
        processed_files.append(mfccs)

    # Build model
    device = torch.device('cpu')
    vocabulary = Vocabulary(vocab=torch.load(f'{model_path}/vocabulary.pt'), device=device)
    vocab_size = vocabulary.vocab_size
    d_model = config['d_model']
    dropout = config['dropout']
    num_heads = config['num_heads']
    max_length = config['max_length']
    num_layers = config['num_layers']

    model = Transformer(vocab_size=vocab_size, d_model=d_model, dropout=dropout,
                        num_heads=num_heads, max_length=max_length, num_layers=num_layers,
                        device=device).to(device)

    state_dict = torch.load(f'{model_path}/model.pt')
    model.load_state_dict(state_dict)

    total_params = 0
    for param in model.parameters():
        total_params += sum(list(param.size()))
    print()
    print('Total model parameters:', total_params)
    print()
    model.eval()

    # Evaluate
    for file in processed_files:
        source = torch.tensor(file, dtype=torch.float32, device=device).unsqueeze(0)
        target_sequence = torch.zeros(size=(1, 13), device=device).long()
        print('Source shape:', source.shape, source.dtype)
        print('Target shape:', target_sequence.shape, target_sequence.dtype)

        result = model(source=source, target_sequences=target_sequence)
        print(result.shape)

        print()
        prediction_indices = torch.argmax(result, dim=-1)[0]
        prediction_sequence = vocabulary.get_sequence_from_tensor(prediction_indices)
        print('Prediction:', ' '.join(prediction_sequence))
        # target_tokens = vocabulary.get_sequence_from_tensor(target_sequence[0])
        # print('Target:', ' '.join(target_tokens))
        print()


if __name__ == '__main__':
    main()