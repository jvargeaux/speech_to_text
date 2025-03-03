import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor

from src.model import Transformer
from src.trainer import pad_source, pad_target
from src.vocabulary import Vocabulary

if TYPE_CHECKING:
    import numpy as np


def print_model_parameters(model: torch.nn.Module):
    total_params = 0
    total_size = 0
    params = []
    for name, param in model.named_parameters():
        total_params += param.nelement()
        size = param.nelement() * param.element_size()
        total_size += size
        params.append((name, size))
    params = sorted(params, key=lambda x: x[1], reverse=True)
    print()
    print('Total model parameters:', total_params)
    print('Total model size:', total_size)
    print()
    # for param in params[:50]:
    #     print(param)


def evaluate_model(model: torch.nn.Module, vocabulary: Vocabulary, batch_size: int, source_batch: Tensor, max_target_length : int):
    model.eval()

    # Init starting tensor with <sos> token, pad to max length, and expand to batch size
    empty_target = F.pad(input=vocabulary.sos_token_tensor, pad=(0, max_target_length - 1), mode='constant', value=vocabulary.pad_token_tensor.item())
    target_batch = empty_target.expand(batch_size, max_target_length)

    # print('Source')
    # print(source_batch[0][12][:])
    # print()

    # Iterate through the target sequence and output the prediction
    # for i in range(1, max_target_length + 1):
    for i in range(1, 2):

        # Get prediction from model
        # sample_out, *_ = model(encoder_source=source_batch, decoder_source=target_batch)
        with torch.no_grad():
            out, *_ = model(encoder_source=source_batch, decoder_source=target_batch)
        sample_out_prediction_indices = torch.argmax(out, dim=-1)

        # Collapse all batches into a single average
        average_prediction = torch.mean(out, dim=0)
        average_prediction_indices = torch.argmax(average_prediction, dim=-1)

        # Take average of all sequences
        # for i in range(len(out) - 1):
        #     avg_out += out[i + 1]
        # avg_out /= len(out)

        # Output the prediction for each sample in the batch
        print()
        print(f'Decoder Input[0]: ', ' '.join(vocabulary.get_sequence_from_tensor(target_batch[0][:i])))
        print(f'Prediction[avg]:  ', ' '.join(vocabulary.get_sequence_from_tensor(average_prediction_indices[:i])))
        print(out[0][i])
        # for n in range(batch_size):
        #     print(f'Prediction[{n}]:    ', ' '.join(vocabulary.get_sequence_from_tensor(sample_out_prediction_indices[n][:i])))
        print()

        # Add predicted token to end of decoder input
        target_batch[:, i] = sample_out_prediction_indices[0][i - 1]

        if sample_out_prediction_indices[0][i - 1] == vocabulary.eos_token_tensor.item():
            print('End of sentence.')
            print(f'Final Prediction[0]:    ', ' '.join(vocabulary.get_sequence_from_tensor(average_prediction_indices[:i - 1])))
            print()
            break


def main() -> None:
    config = OmegaConf.load('config.yaml')

    assert config.audio.model_sample_rate % config.audio.hop_length == 0
    mfcc_per_second = config.audio.model_sample_rate // config.audio.hop_length
    max_source_length = config.model.max_source_length * mfcc_per_second

    use_fixed_padding = True


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
        if sr != config.audio.model_sample_rate:
            samples = librosa.resample(y=samples, orig_sr=sr, target_sr=config.audio.model_sample_rate)

        # Preprocess audio
        mfccs = librosa.feature.mfcc(y=samples, n_fft=config.audio.n_fft, hop_length=config.audio.hop_length, n_mfcc=config.audio.mfcc_depth)
        mfccs = mfccs.T  # (num_bands, num_frames) -> (num_frames, num_bands)
        processed_files.append({
            'name': file,
            'mfccs': mfccs,
        })

    # Build model
    device = torch.device('cpu')
    vocabulary = Vocabulary(vocab=torch.load(Path(args.model, 'vocabulary.pt'), map_location=device), device=device)

    model = Transformer(vocabulary=vocabulary,
                        d_model=config.model.d_model,
                        dropout=None,
                        batch_size=config.model.batch_size,
                        num_heads=config.model.num_heads,
                        max_source_length=max_source_length,
                        max_target_length=config.model.max_target_length,
                        num_encoder_layers=config.model.num_encoder_layers,
                        num_decoder_layers=config.model.num_decoder_layers,
                        mfcc_depth=config.audio.mfcc_depth,
                        device=device).to(device)
    # model.load_state_dict(torch.load(Path(args.model, 'model.pt'), map_location=device))
    model.eval()


    print_model_parameters(model=model)


    for file in processed_files:
        print(f'=====  {file["name"]}  =====')

        # Pad source to max length & expand across batch
        mfccs: np.ndarray = file['mfccs']
        source = torch.tensor(mfccs, dtype=torch.float32, device=device)
        padded_source = F.pad(input=source, pad=(0, 0, 0, max_source_length - source.shape[0]), mode='constant', value=0)
        source_batch = padded_source.expand(config.model.batch_size, max_source_length, source.shape[1])

        evaluate_model(model=model,
                    vocabulary=vocabulary,
                    batch_size=config.model.batch_size,
                    source_batch=source_batch,
                    max_target_length=config.model.max_target_length)

    return


    # Evaluate
    for file in processed_files:
        print(f'=====  {file["name"]}  =====')
        mfccs = file['mfccs']

        # Test with random tensor
        # mfccs = torch.rand((200, Config.MFCC_DEPTH))

        # Init result tensor, we don't know how long it is yet
        result: Tensor = vocabulary.get_tensor_from_sequence('<sos>')

        source = torch.tensor(mfccs, dtype=torch.float32, device=device).unsqueeze(0)
        print(source.size())
        if config.model.batch_size > 1:
            # Pad batch with zeros
            # source_pad: Tensor = torch.zeros((Config.BATCH_SIZE - 1, source.shape[1], source.shape[2]))
            # source = torch.cat((source, source_pad))

            # Duplicate across batch
            source = source[0]
            source = source.expand((config.model.batch_size, source.shape[0], source.shape[1]))
        print(source.size())
        print(source.size(dim=-1))
        padded_source = torch.stack(
            [pad_source(source=item, max_length=max_source_length, device=device) for item in source]).to(device)

        is_end_of_sentence = False
        MAX_OUTPUT_TOKENS = 50
        result_length = 1
        while not is_end_of_sentence:
            expanded_result = result
            if config.model.batch_size > 1:
                # Pad batch with pad tokens
                # pad_sequence = [vocabulary.sos_token] + [vocabulary.pad_token] * (result.shape[0] - 1)
                # result_pad: Tensor = vocabulary.get_tensor_from_sequence(' '.join(pad_sequence)).unsqueeze(0)
                # result_pad = result_pad.expand(Config.BATCH_SIZE - 1, result_pad.shape[1])
                # expanded_result = torch.cat((result.unsqueeze(0), result_pad))

                # Duplicate across batch
                expanded_result = result.expand(config.model.batch_size, result.shape[0])

            out, *_ = model(encoder_source=padded_source, decoder_source=expanded_result)

            # Take only the first sequence of the prediction batch, the source batch was padded
            avg_out = out[0]

            # Take average of all sequences
            # for i in range(len(out) - 1):
            #     avg_out += out[i + 1]
            # avg_out /= len(out)

            # Get predicted tokens
            prediction = avg_out
            prediction_indices = torch.argmax(prediction, dim=-1)
            print()
            print('Input:', ' '.join(vocabulary.get_sequence_from_tensor(result)))
            for i in range(len(out)):
                print(f'Prediction[{i}]:', ' '.join(vocabulary.get_sequence_from_tensor(torch.argmax(out[i], dim=-1))))
            print('Prediction (average):', ' '.join(vocabulary.get_sequence_from_tensor(prediction_indices)))

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
