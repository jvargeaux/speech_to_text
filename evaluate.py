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

    parser.add_argument('--files', '-f', type=Path, help='Path to directory containing audio files to evaluate')
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
        mfccs = librosa.feature.mfcc(y=samples, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH, n_mfcc=Config.MFCC_DEPTH)
        mfccs = mfccs.T  # (num_bands, num_frames) -> (num_frames, num_bands)
        processed_files.append({
            'name': file,
            'mfccs': mfccs
        })

    # Build model
    device = torch.device('cpu')
    vocabulary = Vocabulary(vocab=torch.load(Path(args.model, 'vocabulary.pt')), device=device)

    model = Transformer(vocabulary=vocabulary, d_model=Config.D_MODEL, dropout=None, batch_size=Config.BATCH_SIZE,
                        num_heads=Config.NUM_HEADS, max_length=Config.MAX_LENGTH, num_layers=Config.NUM_LAYERS,
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
    for file in processed_files:
        print(f'=====  {file["name"]}  =====')
        mfccs = file['mfccs']

        # Init result tensor, we don't know how long it is yet
        result: Tensor = vocabulary.get_tensor_from_sequence('<sos>')

        source = torch.tensor(mfccs, dtype=torch.float32, device=device).unsqueeze(0)
        if Config.BATCH_SIZE > 1:
            # Pad batch with zeros
            # source_pad: Tensor = torch.zeros((Config.BATCH_SIZE - 1, source.shape[1], source.shape[2]))
            # source = torch.cat((source, source_pad))

            # Duplicate across batch
            source = source[0]
            source = source.expand((Config.BATCH_SIZE, source.shape[0], source.shape[1]))

        is_end_of_sentence = False
        MAX_OUTPUT_TOKENS = 200
        result_length = 1
        while not is_end_of_sentence:
            expanded_result = result
            if Config.BATCH_SIZE > 1:
                # Pad batch with pad tokens
                # pad_sequence = [vocabulary.sos_token] + [vocabulary.pad_token] * (result.shape[0] - 1)
                # result_pad: Tensor = vocabulary.get_tensor_from_sequence(' '.join(pad_sequence)).unsqueeze(0)
                # result_pad = result_pad.expand(Config.BATCH_SIZE - 1, result_pad.shape[1])
                # expanded_result = torch.cat((result.unsqueeze(0), result_pad))

                # Duplicate across batch
                expanded_result = result.expand(Config.BATCH_SIZE, result.shape[0])

            prediction, *_ = model(encoder_source=source, decoder_source=expanded_result)

            # Take only the first sequence of the prediction batch, the source batch was padded
            prediction = prediction[3]

            # Get predicted tokens
            prediction_indices = torch.argmax(prediction, dim=-1)
            print()
            print('Input:', ' '.join(vocabulary.get_sequence_from_tensor(result)))
            print('Output:', ' '.join(vocabulary.get_sequence_from_tensor(prediction_indices)))

            # Set result to current prediction with prepended sos token, and trim to length + 1
            result_length += 1
            result = torch.cat((vocabulary.get_tensor_from_sequence('<sos>'), prediction_indices), dim=-1)
            result = result[:result_length]
            # last_token = vocabulary.get_tensor_from_sequence(prediction_tokens[-1])
            # result = torch.cat((result, last_token), dim=-1)

            if vocabulary.eos_token in vocabulary.get_sequence_from_tensor(prediction_indices):
                print()
                print('End of sentence token detected.')
                is_end_of_sentence = True
            if len(prediction_indices) > MAX_OUTPUT_TOKENS:
                print()
                print('Max output limit exceeded.')
                is_end_of_sentence = True

        print()
        final_output = vocabulary.get_sequence_from_tensor(result)
        print('Final Output:', ' '.join(final_output))
        print()
        print()


if __name__ == '__main__':
    main()