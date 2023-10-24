import argparse
import math
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from src.transformer import Transformer
from src.vocabulary import Vocabulary
from config import Config


def main():
    parser = argparse.ArgumentParser(
             prog='S2T Evaluator',
             description='Evaluate the S2T model',
             epilog='Epilogue sample text')

    parser.add_argument('--files', '-f', type=Path, nargs='+', help='Path to directory containing audio files to evaluate')
    parser.add_argument('--model', '-m', type=Path, help='Path to model used for inference')
    args = parser.parse_args()

    # Preprocess
    processed_files = []
    files = list(args.files.glob('*.*'))
    for file in files:
        samples, sr = sf.read(file)
        # Resample if sample rate doesn't match
        if sr != Config.MODEL_SAMPLE_RATE:
            samples = librosa.resample(y=samples, orig_sr=sr, target_sr=Config.MODEL_SAMPLE_RATE)

        # Preprocess audio
        mfccs = librosa.feature.mfcc(y=samples, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH, n_mfcc=Config.N_MFCC)
        mfccs = mfccs.T  # (num_bands, num_frames) -> (num_frames, num_bands)
        processed_files.append(mfccs)

    # Build model
    device = torch.device('cpu')
    vocabulary = Vocabulary(vocab=torch.load(Path(args.model, 'vocabulary.pt')), device=device)
    d_model = Config.D_MODEL
    dropout = 0.  # Do not drop any values! We are not training.
    num_heads = Config.NUM_HEADS
    max_length = Config.MAX_LENGTH
    num_layers = Config.NUM_LAYERS

    model = Transformer(vocabulary=vocabulary, d_model=d_model, dropout=dropout,
                        num_heads=num_heads, max_length=max_length, num_layers=num_layers,
                        device=device).to(device)
    model.load_state_dict(torch.load(Path(args.model, 'model.pt')))

    total_params = 0
    for param in model.parameters():
        total_params += sum(list(param.size()))
    print()
    print('Total model parameters:', total_params)
    print()
    model.eval()

    # Evaluate
    for mfccs in processed_files:
        # Init result tensor, we don't know how long it is yet
        result: Tensor = vocabulary.get_tensor_from_sequence('<sos>')

        source = torch.tensor(mfccs, dtype=torch.float32, device=device)
        print('Source shape:', source.shape)

        is_end_of_sentence = False
        max_tokens = 1000
        while not is_end_of_sentence:
            prediction, *_ = model(encoder_source=source.unsqueeze(0), decoder_source=result.unsqueeze(0))

            # Get predicted tokens
            prediction = prediction.view(-1, prediction.shape[-1])
            prediction_indices = torch.argmax(prediction, dim=-1)
            prediction_tokens = vocabulary.get_sequence_from_tensor(prediction_indices)
            print()
            print('Input:', vocabulary.get_sequence_from_tensor(result))
            print('Output:', prediction_tokens)

            # Append last output token and feed back into decoder input
            last_token = vocabulary.get_tensor_from_sequence(prediction_tokens[-1])
            result = torch.cat((result, last_token), dim=-1)

            if prediction_tokens[-1] == vocabulary.eos_token:
                print()
                print('End of sentence token detected.')
                is_end_of_sentence = True
            if len(prediction_indices) > max_tokens:
                print()
                print('Max output limit exceeded.')
                is_end_of_sentence = True

        # final_result = vocabulary.get_sequence_from_tensor(result)
        # print()
        # print('Result:', final_result)
        # print('Result:', ' '.join(final_result[1:-1]))
        # print()
        # return
        prediction_indices = torch.argmax(result, dim=-1)[0]
        prediction_sequence = vocabulary.get_sequence_from_tensor(prediction_indices)
        print('Prediction:', ' '.join(prediction_sequence))
        # target_tokens = vocabulary.get_sequence_from_tensor(target_sequence[0])
        # print('Target:', ' '.join(target_tokens))
        print()


if __name__ == '__main__':
    main()