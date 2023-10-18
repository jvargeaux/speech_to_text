from datetime import datetime
from pathlib import Path
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
    
    def train(self, num_epochs: int, batch_size: int, optimizer, learning_rate: float, lr_gamma: float,
              num_files: int | None = None):
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
        vocabulary = Vocabulary(batch=transcripts, device=self.device)
        vocab_size = vocabulary.vocab_size
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
        self.model = Transformer(vocab_size=vocab_size, d_model=self.d_model, dropout=self.dropout,
                                 num_heads=self.num_heads, max_length=self.max_length, num_layers=self.num_layers,
                                 device=self.device, debug=self.debug).to(self.device)

        optimizer = optimizer(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)

        run_path = Path(f'runs/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
        if self.debug:
            run_path = Path('runs/debug')
        if not Path.exists(run_path):
            Path.mkdir(run_path, parents=True)
        summary_writer = SummaryWriter(run_path)

        if self.debug:
            num_epochs = 1
            train_loader = DataLoader(dataset=data[:1], batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=self.collate)

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

                    source = torch.stack([torch.tensor(item[0]) for item in batch]).to(self.device)
                    transcripts = [item[2] for item in batch]
                    target_sequences = torch.stack(list(map(vocabulary.build_tokenized_target, transcripts))).to(self.device)

                    (out, embedded_source, pos_encoded_source, encoder_out, embedded_target, pos_encoded_target,
                     target_mask, decoder_out) = self.model(encoder_source=source, decoder_source=target_sequences)

                    # Compare against next word in sequence
                    target: Tensor = target_sequences[:,1:]
                    prediction: Tensor = out[:,:-1]

                    # Compare against same word in sequence
                    # prediction = out
                    # target = target_sequences

                    # Flatten batches
                    target_flat = target.view(-1)
                    prediction_flat = prediction.view(-1, out.shape[-1])

                    # Calculate loss & perform backprop
                    loss = criterion(prediction_flat, target_flat)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # For showing prediction
                    target_indices = target_sequences[0]
                    target_tokens = vocabulary.get_sequence_from_tensor(target_indices)
                    prediction_indices = torch.argmax(out, dim=-1)[0]
                    prediction_tokens = vocabulary.get_sequence_from_tensor(prediction_indices)

                    epoch_loss += loss.item()
                    epoch_tokens += len(target_flat)

                    # Print every x steps
                    if (i + 1) % num_steps_to_print == 0:
                        elapsed = time.time() - start
                        avg_loss = epoch_loss / (i + 1)
                        tokens_per_sec = epoch_tokens / elapsed
                        summary_writer.add_scalar('Loss (CE)', avg_loss, global_step=global_step)
                        summary_writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step=global_step)
                        summary_writer.add_scalar('Tokens/sec', tokens_per_sec, global_step=global_step)
                        print(f'Epoch: {(epoch+1):>3}/{num_epochs} | '
                            f'Step: {(i+1):>4}/{num_steps} | '
                            f'Tokens/sec: {tokens_per_sec:>6.1f} | '
                            f'Avg. Loss: {avg_loss:.4f} | '
                            f'LR: {scheduler.get_last_lr()[0]:.2e} | '
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

                if (epoch + 1) % 10 == 0:
                    self.model.eval()
                    for i in range(1, len(target_indices) + 1):
                        decoder_in = target_sequences[:,:i]
                        out, *_ = self.model(encoder_source=source, decoder_source=decoder_in)
                        decoder_out_indices = torch.argmax(out, dim=-1)[0]
                        print()
                        print('Decoder Input: ', ' '.join(vocabulary.get_sequence_from_tensor(decoder_in[0])))
                        print('Prediction:    ', ' '.join(vocabulary.get_sequence_from_tensor(decoder_out_indices)))
                        print()
                    self.model.train()

                # # Write heatmaps
                # metrics = Metrics(debug=self.debug)
                # metrics.add_heatmap(data=source[0], ylabel='Source MFCCs')
                # metrics.add_heatmap(data=embedded_source[0], ylabel='Source audio embedding')
                # metrics.add_heatmap(data=pos_encoded_source[0], ylabel='Source pos encoding')
                # metrics.add_heatmap(data=encoder_out[0], ylabel='Encoder out')
                # metrics.add_heatmap(data=embedded_target[0], ylabel='Target word embedding')
                # metrics.add_heatmap(data=pos_encoded_target[0], ylabel='Target pos encoding')
                # metrics.add_heatmap(data=target_mask[0], ylabel='Target mask')
                # metrics.add_heatmap(data=decoder_out[0], ylabel='Decoder out')
                # metrics.add_heatmap(data=out[0], ylabel='Out')
                # metrics.draw_heatmaps()
                # summary_writer.add_figure('Heatmaps', plt.gcf(), global_step=global_step)
                # plt.clf()

                # # Write confusion matrix
                # prediction_flat_collapsed = torch.argmax(prediction_flat, dim=-1)
                # metrics.draw_confusion_matrix(target=target_flat, predicted=prediction_flat_collapsed)
                # summary_writer.add_figure('Confusion Matrix', plt.gcf(), global_step=global_step)
                # plt.clf()

                # prediction_no_grad = prediction.clone().detach().requires_grad_(False)
                # # prediction_display = F.softmax(prediction_no_grad, dim=-1)
                # metrics.show_heatmap(data=prediction_no_grad, xlabel='Vocab', ylabel='Sequence')

        except KeyboardInterrupt:
            print('\r  ')

        print()
        print('Training finished.')
        print()

        if not self.debug:
            # Save model & optimizer
            save_directory = Path(run_path, 'models')
            if not Path.exists(save_directory):
                Path.mkdir(save_directory, parents=True)
            torch.save(self.model.state_dict(), f'{save_directory}/model.pt')
            torch.save(optimizer.state_dict(), f'{save_directory}/optimizer.pt')
            torch.save(vocabulary.vocab, f'{save_directory}/vocabulary.pt')

        summary_writer.close()