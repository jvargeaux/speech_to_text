from datetime import datetime
from pathlib import Path
from typing import List
import h5py
import json
import matplotlib.pyplot as plt
import time
import torch
from torch import Tensor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from splits import SPLITS
from preprocess import Preprocessor
from config import Config
from src.transformer import Transformer
from src.vocabulary import Vocabulary
from src.metrics import Metrics


class Trainer():
    def __init__(self,
                 d_model: int,
                 num_layers: int,
                 batch_size: int,
                 dropout: float,
                 num_heads: int,
                 max_length: int,
                 max_vocab_size: int | None,
                 mfcc_depth: int,
                 num_epochs: int,
                 lr: float,
                 lr_gamma: float,
                 lr_min: float,
                 weight_decay: float | None,
                 num_warmup_steps: int | None,
                 output_lines_per_epoch: int,
                 checkpoint_after_epoch: int | None,
                 checkpoint_path: Path | None,
                 reset_lr: bool,
                 cooldown: int | None,
                 split_train: str,
                 split_test: str,
                 subset: int | None=None,
                 device: str='cpu',
                 debug=False):
        # Model
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.max_length = max_length
        self.max_vocab_size = max_vocab_size
        self.batch_size = batch_size
        self.mfcc_depth = mfcc_depth

        # Training
        self.device = device
        self.debug = debug
        self.run_path = Path(f'runs/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
        self.num_epochs = num_epochs
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.lr_min = lr_min
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.cooldown = cooldown
        self.output_lines_per_epoch = output_lines_per_epoch
        self.checkpoint_after_epoch = checkpoint_after_epoch
        self.checkpoint_path = checkpoint_path
        self.reset_lr = reset_lr
        self.split_train = split_train
        self.split_test = split_test
        self.subset = subset

        self.data_train = []
        self.data_test = []
        self.vocabulary = None
        self.model = None
        self.optimizer = None

        self.use_amp = True if self.device == 'cuda' else False
        self.scaler = None
        self.use_multiple = True

    def import_data(self):
        train_path = Path('mfcc', self.split_train)
        test_path = Path('mfcc', self.split_test)
        train_files = list(train_path.glob('*.hdf5'))
        test_files = list(test_path.glob('*.hdf5'))
        if len(train_files) == 0 or len(test_files) == 0:
            print(f'No data folder {train_path} detected. Preprocessing now...')
            preprocessor = Preprocessor(split_train=self.split_train, split_test=self.split_test)
            preprocessor.preprocess()
            train_files = list(train_path.glob('*.hdf5'))
            test_files = list(test_path.glob('*.hdf5'))
            print()
        if self.subset is not None:
            train_files = train_files[:self.subset]

        print('Loading mfcc data...')
        data_train = []
        for file in train_files:
            with h5py.File(file, 'r') as file_data:
                data_train.append([file_data['mfccs'][:],
                            file_data['mfccs'].attrs['sample_rate'],
                            file_data['mfccs'].attrs['transcript'],
                            file_data['mfccs'].attrs['speaker_id']])
        self.data_train = data_train
        data_test = []
        for file in test_files:
            with h5py.File(file, 'r') as file_data:
                data_test.append([file_data['mfccs'][:],
                            file_data['mfccs'].attrs['sample_rate'],
                            file_data['mfccs'].attrs['transcript'],
                            file_data['mfccs'].attrs['speaker_id']])
        self.data_test = data_test
        print('Data loaded.')
        print()

    def build_vocabulary(self):
        print('Building vocabulary...')
        transcripts = [item[2] for item in self.data_train]
        self.vocabulary = Vocabulary(batch=transcripts, max_size=self.max_vocab_size, device=self.device)
        print('Vocabulary built.')
        print()
        print('Dataset Size (Train):', len(self.data_train))
        print('Dataset Size (Test):', len(self.data_test))
        print('Model Vocab Size:', self.vocabulary.vocab_size)
        print()

    def load_checkpoint_vocabulary(self):
        print('Loading checkpoint vocabulary...')
        vocabulary_path = Path(self.checkpoint_path, 'vocabulary.pt')
        self.vocabulary = Vocabulary(vocab=torch.load(vocabulary_path), device=self.device)
        print('Vocabulary loaded.')
        print()
        print('Dataset Size (Train):', len(self.data_train))
        print('Dataset Size (Test):', len(self.data_test))
        print('Model Vocab Size:', self.vocabulary.vocab_size)
        print()

    def verify_longest_sequence(self):
        longest = 0
        for item in self.data_train:
            if len(item[0]) > longest:
                longest = len(item[0])
        print('Longest sequence length:', longest)
        print('Longest sequence length (after embedding):', longest // 4)

    def save_config(self):
        config_path = Path(self.run_path, 'config.json')
        config = { key: value for key, value in vars(Config).items() if not '__' in key }
        with open(config_path, 'w') as file:
            file.write(json.dumps(config))

    def collate(self, batch):
        return batch

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

    def check_model_for_randomness(self):
        random_source = torch.rand((self.batch_size, 80, self.mfcc_depth), device=self.device)
        random_target = torch.randint(low=0, high=20, size=(self.batch_size, 22), device=self.device).to(torch.long)

        self.model.eval()
        with torch.no_grad():
            out1, *_ = self.model(encoder_source=random_source, decoder_source=random_target)
            out2, *_ = self.model(encoder_source=random_source, decoder_source=random_target)
        self.model.train()

        difference = torch.sum(out2 - out1).item()
        if difference != 0:
            print('Warning! Model randomness detected:', difference)

    def save_models(self, epoch: int, global_step: int):
        save_directory = Path(self.run_path, f'models_{epoch}')
        if not Path.exists(save_directory):
            Path.mkdir(save_directory, parents=True)
        torch.save(self.model.state_dict(), f'{save_directory}/model.pt')
        torch.save(self.optimizer.state_dict(), f'{save_directory}/optimizer.pt')
        torch.save(self.vocabulary.vocab, f'{save_directory}/vocabulary.pt')
        with open(Path(save_directory, 'global_step.json'), 'w') as file:
            file.write(json.dumps({ 'global_step': global_step }))

    def save_images(self):
        pass
        # metrics = Metrics(debug=self.debug)
        # metrics.add_heatmap(data=padded_sources[0], ylabel='Source MFCCs')
        # metrics.add_heatmap(data=embedded_source[0], ylabel='Source audio embedding')
        # metrics.add_heatmap(data=pos_encoded_source[0], ylabel='Source pos encoding')
        # metrics.add_heatmap(data=encoder_out[0], ylabel='Encoder out')
        # metrics.add_heatmap(data=embedded_target[0], ylabel='Target word embedding')
        # metrics.add_heatmap(data=pos_encoded_target[0], ylabel='Target pos encoding')
        # metrics.add_heatmap(data=target_mask[0], ylabel='Target mask')
        # metrics.add_heatmap(data=decoder_out[0], ylabel='Decoder out')
        # metrics.add_heatmap(data=out[0], ylabel='Out')
        # metrics.draw_heatmaps()
        # train_writer.add_figure('Heatmaps', plt.gcf(), global_step=global_step)
        # plt.clf()

        # # Write confusion matrix
        # prediction_flat_collapsed = torch.argmax(prediction_flat, dim=-1)
        # metrics.draw_confusion_matrix(target=target_flat, predicted=prediction_flat_collapsed)
        # train_writer.add_figure('Confusion Matrix', plt.gcf(), global_step=global_step)
        # plt.clf()

    def train(self):
        try:
            self.import_data()
        except:
            return
        self.verify_longest_sequence()
        print()

        if self.checkpoint_path is not None:
            if not Path.exists(self.checkpoint_path):
                print('Error: Provided checkpoint path does not exist. Aborting...')
                print()
                return
            self.load_checkpoint_vocabulary()
        else:
            self.build_vocabulary()

        # Prepare training data
        train_loader = DataLoader(dataset=self.data_train, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=self.collate)
        test_loader = DataLoader(dataset=self.data_test, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=self.collate)
        num_steps = len(train_loader)

        # Build model
        self.model = Transformer(vocabulary=self.vocabulary, d_model=self.d_model, batch_size=self.batch_size,
                                 dropout=self.dropout, num_heads=self.num_heads, max_length=self.max_length,
                                 num_layers=self.num_layers, device=self.device, mfcc_depth=self.mfcc_depth,
                                 debug=self.debug).to(self.device)
        if self.checkpoint_path is not None:
            self.model.load_state_dict(torch.load(Path(self.checkpoint_path, 'model.pt')))
        self.check_model_for_randomness()

        # Set optimizer and criterion
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9,
                                          weight_decay=self.weight_decay)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)

        if self.checkpoint_path is not None:
            self.optimizer.load_state_dict(torch.load(Path(self.checkpoint_path, 'optimizer.pt')))
            if self.reset_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

        # Create LR schedulers
        warmup_scheduler = None
        if self.checkpoint_path is None and self.num_warmup_steps > 0:
            warmup_scheduler = lr_scheduler.LinearLR(optimizer=self.optimizer, start_factor=1e-9, end_factor=1.0,
                                                    total_iters=self.num_warmup_steps)
        # training_scheduler = lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.lr_gamma)
        training_scheduler = lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.num_epochs * num_steps * 2, eta_min=self.lr_min)
        # training_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, factor=1e-4, patience=1)
        scheduler = training_scheduler  # SequentialLR uses deprecated pattern, produces warning

        # Create grad scaler, becomes no-op if enabled is False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Create tensorboard summary writer
        if not Path.exists(self.run_path):
            Path.mkdir(self.run_path, parents=True)
        self.save_config()
        train_writer = SummaryWriter(self.run_path)
        test_writer = SummaryWriter(self.run_path)
        graph_source = self.padded_source_from_batch(batch=self.data_train[:self.batch_size])
        graph_target, _ = self.padded_target_from_batch(batch=self.data_train[:self.batch_size])
        train_writer.add_graph(self.model, (graph_source, graph_target))

        print_step = num_steps // self.output_lines_per_epoch
        if print_step <= 0:
            print_step = None
        print()
        print()
        start_step = 0
        if self.checkpoint_path is not None:
            step_path = Path(self.checkpoint_path, 'global_step.json')
            if Path.exists(step_path):
                with open(step_path, 'r') as file:
                    step_file = json.load(file)
                    start_step = step_file['global_step']
            print('Continuing training from checkpoint model...')
        else:
            print('Starting training...')
        print()
        self.model.train()
        try:
            for epoch in range(self.num_epochs):
                start = time.time()
                epoch_count = 0
                epoch_tokens = 0
                epoch_loss = 0
                epoch_error = 0

                # Train
                for i, batch in enumerate(train_loader):
                    global_step = start_step + (epoch * num_steps + i + 1)
                    if self.checkpoint_path is None and self.num_warmup_steps > 0 and global_step <= self.num_warmup_steps:
                        scheduler = warmup_scheduler
                    else:
                        scheduler = training_scheduler

                    padded_sources = self.padded_source_from_batch(batch=batch)
                    padded_targets, pad_indices = self.padded_target_from_batch(batch=batch)

                    # Becomes no-op if self.use_amp is False
                    # NOTE: passing self.device to device_type gives error, keep on 'cuda' even if device is cpu
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                        (out, embedded_source, pos_encoded_source, encoder_out, embedded_target, pos_encoded_target,
                        target_mask, decoder_out) = self.model(encoder_source=padded_sources, decoder_source=padded_targets)

                        target_flat, prediction_flat = self.unpad_and_flatten_batch(padded_targets, out, pad_indices)

                        # Calculate loss & perform backprop
                        self.optimizer.zero_grad(set_to_none=True)
                        loss = criterion(prediction_flat, target_flat)
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scheduler.step()

                    epoch_count += len(batch)
                    epoch_tokens += len(target_flat)
                    epoch_loss += loss.item()
                    prediction_indices = torch.argmax(prediction_flat, dim=-1)
                    epoch_error += torch.sum((prediction_indices != target_flat).float()).item()

                    if self.cooldown is not None and self.cooldown > 0:
                        time.sleep(self.cooldown)

                    if print_step is not None and (i + 1) % print_step == 0:
                        elapsed = time.time() - start
                        step_time = elapsed / (i + 1)
                        tokens_per_sec = epoch_tokens / elapsed
                        avg_loss = epoch_loss / epoch_count
                        word_error_rate = epoch_error / epoch_tokens
                        train_writer.add_scalar('Metrics/1 WER', word_error_rate, global_step=global_step)
                        train_writer.add_scalar('Metrics/2 Loss (CE)', avg_loss, global_step=global_step)
                        train_writer.add_scalar('Metrics/3 LR', scheduler.get_last_lr()[0], global_step=global_step)
                        train_writer.add_scalar('Performance/Tokens Per Second', tokens_per_sec, global_step=global_step)
                        train_writer.add_histogram('Vocab Distribution', torch.mean(prediction_flat, dim=0), global_step=global_step)
                        print(f'Epoch: {(epoch+1):>3}/{self.num_epochs}  |  '
                            f'Step: {(i+1):>4}/{num_steps}  |  '
                            f'Tokens/sec: {tokens_per_sec:>6.1f}  |  '
                            f'Loss: {avg_loss:.5f}  |  '
                            f'WER: {word_error_rate:>6.1%}  |  '
                            f'LR: {scheduler.get_last_lr()[0]:.2e}  |  '
                            f'Time: {step_time:>6.3f}s / {(elapsed / 60):>4.1f}m')

                # Test (Validation)
                print('Running validation...')
                test_start = time.time()
                test_count = 0
                test_tokens = 0
                test_loss = 0
                test_error = 0
                self.model.eval()
                for i, batch in enumerate(test_loader):
                    padded_sources = self.padded_source_from_batch(batch=batch)
                    padded_targets, pad_indices = self.padded_target_from_batch(batch=batch)

                    # Becomes no-op if self.use_amp is False
                    # NOTE: passing self.device to device_type gives error, keep on 'cuda' even if device is cpu
                    with torch.no_grad():
                        (out, embedded_source, pos_encoded_source, encoder_out, embedded_target, pos_encoded_target,
                        target_mask, decoder_out) = self.model(encoder_source=padded_sources, decoder_source=padded_targets)

                        target_flat, prediction_flat = self.unpad_and_flatten_batch(padded_targets, out, pad_indices)

                        loss = criterion(prediction_flat, target_flat)

                    test_count += len(batch)
                    test_tokens += len(target_flat)
                    test_loss += loss.item()
                    prediction_indices = torch.argmax(prediction_flat, dim=-1)
                    test_error += torch.sum((prediction_indices != target_flat).float()).item()

                    if self.cooldown is not None and self.cooldown > 0:
                        time.sleep(self.cooldown)

                test_elapsed = time.time() - test_start
                test_step_time = test_elapsed / (i + 1)
                test_tokens_per_sec = test_tokens / test_elapsed
                test_avg_loss = test_loss / test_count
                test_word_error_rate = test_error / test_tokens
                test_writer.add_scalar('Metrics/1 WER', test_word_error_rate, global_step=global_step)
                test_writer.add_scalar('Metrics/2 Loss (CE)', test_avg_loss, global_step=global_step)
                test_writer.add_scalar('Performance/Tokens Per Second', test_tokens_per_sec, global_step=global_step)
                print(f'VALIDATION RESULTS  |  '
                    f'Tokens/sec: {test_tokens_per_sec:>6.1f}  |  '
                    f'Loss: {test_avg_loss:.5f}  |  '
                    f'WER: {test_word_error_rate:>6.1%}  |  '
                    f'Time: {test_step_time:>6.3f}s / {(test_elapsed / 60):>4.1f}m')
                self.model.train()

                if self.checkpoint_after_epoch is not None and (epoch + 1) % self.checkpoint_after_epoch == 0:
                    self.save_models(epoch=epoch+1, global_step=global_step)
                    print()
                    print('Models saved.')
                    self.model.eval()

                    # Take random sample from dataset
                    random_index = torch.randint(low=0, high=len(self.data_train), size=(1,)).item()
                    random_sample_source = torch.tensor(self.data_train[random_index][0], device=self.device).unsqueeze(0)
                    random_sample_target = self.vocabulary.build_tokenized_target(self.data_train[random_index][2]).unsqueeze(0)

                    # Copy data across batch size
                    random_sample_source = random_sample_source.expand((self.batch_size, random_sample_source.shape[1], random_sample_source.shape[2]))
                    random_sample_target = random_sample_target.expand((self.batch_size, random_sample_target.shape[1]))

                    # Iterate through the random sample target sequence and output the prediction
                    for i in range(1, random_sample_target.shape[1] + 1):
                        decoder_in = random_sample_target[:, :i]
                        sample_out, *_ = self.model(encoder_source=random_sample_source, decoder_source=decoder_in)
                        sample_out_indices = torch.argmax(sample_out, dim=-1)
                        print()
                        print('Decoder Input: ', ' '.join(self.vocabulary.get_sequence_from_tensor(decoder_in[0])))
                        print('Prediction:    ', ' '.join(self.vocabulary.get_sequence_from_tensor(sample_out_indices[0])))
                        print()
                    self.model.train()

        except KeyboardInterrupt:
            print('\r  ')

        print()
        print('Training finished.')
        print()
        train_writer.close()
        test_writer.close()