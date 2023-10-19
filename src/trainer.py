from datetime import datetime
from pathlib import Path
from typing import List
import h5py
import matplotlib.pyplot as plt
import time
import torch
from torch import Tensor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocess import Preprocessor, SPLITS
from src.transformer import Transformer
from src.vocabulary import Vocabulary
from src.metrics import Metrics


class Trainer():
    def __init__(self, d_model: int, num_layers: int, dropout: float, num_heads: int,
                 max_length: int, device, debug = False):
        self.device = device
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.max_length = max_length
        self.debug = debug
        self.vocabulary = None

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

    def pad_source(self, source, max_length: int, mfcc_dim: int) -> Tensor:
        source_tensor = torch.tensor(source, device=self.device)
        pad_tensor = torch.zeros((max_length - source_tensor.shape[0], mfcc_dim), device=self.device)
        return torch.cat((source_tensor, pad_tensor))

    def pad_target(self, target: Tensor, max_length: int) -> (Tensor, int):
        num_pad_tokens = max_length - target.shape[0]
        pad_sequence = ' '.join([self.vocabulary.pad_token] * num_pad_tokens)
        pad_tensor = self.vocabulary.get_tensor_from_sequence(pad_sequence)
        pad_index = len(target)
        return torch.cat((target, pad_tensor)), pad_index

    def padded_source_from_batch(self, batch) -> Tensor:
        mfcc_dim = len(batch[0][0][0])
        lengths = [len(item[0]) for item in batch]
        max_length = max(lengths)

        padded_source = torch.stack(
            [self.pad_source(source=item[0], max_length=max_length, mfcc_dim=mfcc_dim) for item in batch]).to(self.device)
        return padded_source

    def padded_target_from_batch(self, batch) -> (Tensor, List[int]):
        target_indices = list(map(self.vocabulary.build_tokenized_target, [item[2] for item in batch]))
        lengths = [item.shape[0] for item in target_indices]
        max_length = max(lengths)

        # Produces [(padded_target, pad_index), ...]
        padded = [self.pad_target(target=item, max_length=max_length) for item in target_indices]
        # Convert to ((padded_target, ...), (pad_index, ...)), need to convert to lists
        padded_targets, pad_indices = zip(*padded)

        return torch.stack(list(padded_targets)).to(self.device), list(pad_indices)

    def unpad_and_flatten_batch(self, target_batch: Tensor, prediction_batch: Tensor, pad_indices: List[int]):
        unpadded_targets = []
        unpadded_predictions = []
        for i in range(0, len(pad_indices)):
            pad_index = pad_indices[i]
            unpadded_target = target_batch[i][:pad_index]
            unpadded_prediction = prediction_batch[i][:pad_index]
            # Compare against next word in sequence
            unpadded_targets.append(unpadded_target[1:])
            unpadded_predictions.append(unpadded_prediction[:-1])
        # for target in unpadded_targets:
        #     print('Unpadded targets:', target.shape)
        # for prediction in unpadded_predictions:
        #     print('Unpadded predictions:', prediction.shape)
        return torch.cat(unpadded_targets).to(self.device), torch.cat(unpadded_predictions).to(self.device)

    def train(self, num_epochs: int, batch_size: int, optimizer, learning_rate: float, lr_gamma: float,
              num_warmup_steps: int, num_files: int | None = None):
        # Import preprocessed mfcc data
        data = []
        files = list(Path('.').glob('mfcc/*.hdf5'))

        if len(files) == 0:
            print('No preprocessed MFCC folder detected. Preprocessing now...')
            preprocessor = Preprocessor(dataset_url=SPLITS.DEV_CLEAN.value)
            preprocessor.preprocess()
            files = list(Path('.').glob('mfcc/*.hdf5'))
            print()

        if num_files is not None:
            files = files[:num_files]
        for file in files:
            with h5py.File(file, 'r') as file_data:
                data.append([file_data['mfccs'][:],
                            file_data['mfccs'].attrs['sample_rate'],
                            file_data['mfccs'].attrs['transcript'],
                            file_data['mfccs'].attrs['speaker_id']])

        # Build vocabulary from all transcripts
        print('Building vocabulary...')
        transcripts = [item[2] for item in data]
        self.vocabulary = Vocabulary(batch=transcripts, device=self.device)
        vocab_size = self.vocabulary.vocab_size
        print('Vocabulary built.')
        print()
        print('Files:', len(data))
        print('Model Vocab Size:', vocab_size)
        # all_words = vocab.get_itos()

        # Prepare training data
        train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=self.collate)
        num_steps = len(train_loader)
        print('Number of batches:', num_steps)
        print()

        # Build model
        self.model = Transformer(vocabulary=self.vocabulary, d_model=self.d_model, dropout=self.dropout,
                                 num_heads=self.num_heads, max_length=self.max_length, num_layers=self.num_layers,
                                 device=self.device, debug=self.debug).to(self.device)

        # Set optimizer and criterion
        optimizer = optimizer(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)

        # Create LR schedulers
        warmup_scheduler = lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1e-9, end_factor=1.0,
                                                 total_iters=num_warmup_steps)
        training_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
        scheduler = warmup_scheduler  # SequentialLR uses deprecated pattern, produces warning

        run_path = Path(f'runs/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
        if self.debug:
            run_path = Path('runs/debug')
        if not Path.exists(run_path):
            Path.mkdir(run_path, parents=True)
        summary_writer = SummaryWriter(run_path)

        if self.debug:
            num_epochs = 1
            train_loader = DataLoader(dataset=data[4:8], batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=self.collate)

        num_steps_to_print = 10
        print('Starting training...')
        print()
        self.model.train()
        try:
            for epoch in range(num_epochs):
                start = time.time()
                epoch_loss = 0
                epoch_tokens = 0
                for i, batch in enumerate(train_loader):
                    global_step = epoch * num_steps + i + 1
                    scheduler = warmup_scheduler if global_step <= num_warmup_steps else training_scheduler

                    padded_sources = self.padded_source_from_batch(batch=batch)
                    padded_targets, pad_indices = self.padded_target_from_batch(batch=batch)

                    (out, embedded_source, pos_encoded_source, encoder_out, embedded_target, pos_encoded_target,
                     target_mask, decoder_out) = self.model(encoder_source=padded_sources, decoder_source=padded_targets)

                    # NOTE: Remove padding before calc loss!

                    target_flat, prediction_flat = self.unpad_and_flatten_batch(padded_targets, out, pad_indices)

                    # Calculate loss & perform backprop
                    loss = criterion(prediction_flat, target_flat)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # For showing prediction
                    target_indices = padded_targets[0]  # Take first of batch
                    target_tokens = self.vocabulary.get_sequence_from_tensor(target_indices)
                    prediction_indices = torch.argmax(out, dim=-1)[0]  # Take first of batch
                    prediction_tokens = self.vocabulary.get_sequence_from_tensor(prediction_indices)

                    # print('Target:', target_flat)
                    # print('Target:', target_flat.shape)
                    # prediction_raw = torch.argmax(prediction_flat, dim=-1)
                    # print('Prediction:', prediction_raw)
                    # print('Prediction:', prediction_raw.shape)

                    epoch_loss += loss.item()
                    non_zero_tokens = (target_flat != 0) * 1
                    epoch_tokens += torch.sum(non_zero_tokens).item()

                    # Print every x steps
                    if (i + 1) % num_steps_to_print == 0:
                        elapsed = time.time() - start
                        avg_loss = epoch_loss / (i + 1)
                        tokens_per_sec = epoch_tokens / elapsed
                        summary_writer.add_scalar('Loss (CE)', avg_loss, global_step=global_step)
                        summary_writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step=global_step)
                        summary_writer.add_scalar('Tokens/sec', tokens_per_sec, global_step=global_step)
                        print(f'Epoch: {(epoch+1):>3}/{num_epochs}  |  '
                            f'Step: {(i+1):>4}/{num_steps}  |  '
                            f'Tokens/sec: {tokens_per_sec:>6.1f}  |  '
                            f'Avg. Loss: {avg_loss:.4f}  |  '
                            f'LR: {scheduler.get_last_lr()[0]:.2e}  |  '
                            f'Epoch Time: {elapsed:>5.1f}s')

                    if self.debug:
                        print()
                        print('Out shape:', out.shape)
                        print('Loss inputs:', prediction_flat.shape, target_flat.shape)
                        print()
                        print('=== Decoder Input ===')
                        print(target_indices.shape)
                        print(target_indices)
                        print(target_tokens)
                        print()
                        print('=== Prediction ===')
                        print(prediction_indices.shape)
                        print(prediction_indices)
                        print(prediction_tokens)
                        print()


                # print()
                # print('Decoder Input: ', ' '.join(target_tokens))
                # print('Prediction:    ', ' '.join(prediction_tokens))
                # print()

                # if (epoch + 1) % 10 == 0:
                self.model.eval()
                # Remove padding
                pad_token_tensor = self.vocabulary.get_tensor_from_sequence(self.vocabulary.pad_token)
                target_pad_index = torch.argmax((target_indices == pad_token_tensor) * 1, dim=-1)
                target_no_pad = target_indices[:target_pad_index]
                for i in range(1, len(target_no_pad) + 1):
                    encoder_in = padded_sources[:1]
                    decoder_in = padded_targets[:1,:i]  # (N, seq_len) -> (1, i)
                    out, *_ = self.model(encoder_source=padded_sources, decoder_source=decoder_in)
                    decoder_out_indices = torch.argmax(out, dim=-1)
                    print()
                    print('Decoder Input: ', ' '.join(self.vocabulary.get_sequence_from_tensor(decoder_in[0])))
                    print('Prediction:    ', ' '.join(self.vocabulary.get_sequence_from_tensor(decoder_out_indices[0])))
                    print()
                self.model.train()

                # Write heatmaps
                metrics = Metrics(debug=self.debug)
                metrics.add_heatmap(data=padded_sources[0], ylabel='Source MFCCs')
                metrics.add_heatmap(data=embedded_source[0], ylabel='Source audio embedding')
                metrics.add_heatmap(data=pos_encoded_source[0], ylabel='Source pos encoding')
                metrics.add_heatmap(data=encoder_out[0], ylabel='Encoder out')
                metrics.add_heatmap(data=embedded_target[0], ylabel='Target word embedding')
                metrics.add_heatmap(data=pos_encoded_target[0], ylabel='Target pos encoding')
                metrics.add_heatmap(data=target_mask[0], ylabel='Target mask')
                metrics.add_heatmap(data=decoder_out[0], ylabel='Decoder out')
                metrics.add_heatmap(data=out[0], ylabel='Out')
                metrics.draw_heatmaps()
                summary_writer.add_figure('Heatmaps', plt.gcf(), global_step=global_step)
                plt.clf()

                # Write confusion matrix
                prediction_flat_collapsed = torch.argmax(prediction_flat, dim=-1)
                metrics.draw_confusion_matrix(target=target_flat, predicted=prediction_flat_collapsed)
                summary_writer.add_figure('Confusion Matrix', plt.gcf(), global_step=global_step)
                plt.clf()

                # prediction_no_grad = prediction.clone().detach().requires_grad_(False)
                # # prediction_display = F.softmax(prediction_no_grad, dim=-1)
                # metrics.show_heatmap(data=prediction_no_grad, xlabel='Vocab', ylabel='Sequence')

        except KeyboardInterrupt:
            print('\r  ')

        print()
        print('Training finished. Saving models...')

        if not self.debug:
            # Save model & optimizer
            save_directory = Path(run_path, 'models')
            if not Path.exists(save_directory):
                Path.mkdir(save_directory, parents=True)
            torch.save(self.model.state_dict(), f'{save_directory}/model.pt')
            torch.save(optimizer.state_dict(), f'{save_directory}/optimizer.pt')
            torch.save(self.vocabulary.vocab, f'{save_directory}/vocabulary.pt')

        print()
        print('Models saved.')
        print()
        summary_writer.close()