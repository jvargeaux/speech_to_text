from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocess import Preprocessor
from src.model import Transformer
from src.vocabulary import Vocabulary
from util import ProgressBar

if TYPE_CHECKING:
    import numpy as np


class Trainer:
    def __init__(self,
                 config: OmegaConf | dict,
                 d_model: int,
                 num_layers: int,
                 batch_size: int,
                 dropout: float,
                 num_heads: int,
                 max_source_length: int,
                 max_target_length: int,
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
                 tests_per_epoch: int | None,
                 checkpoint_path: Path | None,
                 reset_lr: bool,
                 cooldown: int | None,
                 splits_train: list[str],
                 splits_test: list[str],
                 subset: int | None = None,
                 device: str = 'cpu',
                 debug: bool = False) -> None:

        self.device = device
        self.config = config

        # self.mfcc_depth = config['audio']['mfcc_depth']

        # self.d_model: int = config['model']['d_model']
        # self.num_layers: int = config['model']['num_layers']
        # self.dropout: float = config['model']['dropout']
        # self.num_heads: int = config['model']['num_heads']
        # self.max_source_length: int = config['model']['max_source_length']
        # self.max_target_length: int = config['model']['max_target_length']
        # self.max_vocab_size: int | None = config['model']['max_vocab_size']
        # self.batch_size: int = config['model']['batch_size']

        # self.num_epochs: int = config['training']['num_epochs']
        # self.lr: float = config['training']['lr']
        # self.lr_gamma: float = config['training']['lr_gamma']
        # self.lr_min: float = config['training']['lr_min']
        # self.weight_decay: float | None = config['training']['weight_decay']
        # self.num_warmup_steps: int | None = config['training']['num_warmup_steps']
        # self.cooldown: int | None = config['training']['cooldown']
        # self.checkpoint_path: Path | None = config['training']['checkpoint_path']
        # self.reset_lr: bool = config['training']['reset_lr']
        # self.splits_train: list[str] = config['training']['splits_train']
        # self.splits_test: list[str] = config['training']['splits_test']
        # self.subset: int | None = config['training']['subset']

        # self.output_lines_per_epoch = config['output']['output_lines_per_epoch']
        # self.checkpoint_after_epoch = config['output']['checkpoint_after_epoch']
        # self.tests_per_epoch = config['output']['tests_per_epoch']

        # self.debug: bool = debug
        # self.run_path = Path('runs', datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

        self.mfcc_depth = mfcc_depth

        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_vocab_size = max_vocab_size
        self.batch_size = batch_size

        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.lr_min = lr_min
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.cooldown = cooldown
        self.checkpoint_path = checkpoint_path
        self.reset_lr = reset_lr
        self.splits_train = splits_train
        self.splits_test = splits_test
        self.subset = subset

        self.output_lines_per_epoch = output_lines_per_epoch
        self.checkpoint_after_epoch = checkpoint_after_epoch
        self.tests_per_epoch = tests_per_epoch

        self.debug = debug
        self.run_path = Path('runs', datetime.now(tz=timezone(timedelta(hours=+9))).strftime("%Y_%m_%d_%H_%M_%S"))

        self.data_train = []
        self.data_test = []
        self.vocabulary = None
        self.model = None
        self.optimizer = None

        self.use_amp = self.device == 'cuda'
        self.scaler = None
        self.use_multiple = True
        self.use_fixed_padding = False


    def import_data(self) -> None:
        print('Loading mfcc data...')
        print('Train data:')
        train_files = []
        for split in self.splits_train:
            files = list(Path('mfcc', split).glob('*.hdf5'))
            if len(files) == 0:
                preprocessor = Preprocessor(split=split)
                preprocessor.preprocess()
                files = list(Path('mfcc', split).glob('*.hdf5'))
            train_files += files
            print(f'\t{len(files)} files from {split}')

        print('Test data:')
        test_files = []
        for split in self.splits_test:
            files = list(Path('mfcc', split).glob('*.hdf5'))
            if len(files) == 0:
                preprocessor = Preprocessor(split=split)
                preprocessor.preprocess()
                files = list(Path('mfcc', split).glob('*.hdf5'))
            test_files += files
            print(f'\t{len(files)} files from {split}')

        if self.subset is not None:
            train_files = train_files[:self.subset]
            test_files = test_files[:self.subset]

        progress_bar = ProgressBar(title='Train')
        data_train = []
        for i, file in enumerate(train_files):
            with h5py.File(file, 'r') as file_data:
                data_train.append([file_data['mfccs'][:], file_data['mfccs'].attrs['transcript']])
            progress_bar.update(i + 1, len(train_files))
        self.data_train = data_train

        progress_bar = ProgressBar(title='Test')
        data_test = []
        for i, file in enumerate(test_files):
            with h5py.File(file, 'r') as file_data:
                data_test.append([file_data['mfccs'][:], file_data['mfccs'].attrs['transcript']])
            progress_bar.update(i + 1, len(test_files))
        self.data_test = data_test

        print('Data loaded.')
        print(f'\tTrain: {len(train_files)} files' + (' (subset)' if self.subset is not None else ''))
        print(f'\tTest: {len(test_files)} files' + (' (subset)' if self.subset is not None else ''))
        print()


    def build_vocabulary(self) -> None:
        print('Building vocabulary...')
        transcripts = [item[1] for item in self.data_train]
        self.vocabulary = Vocabulary(batch=transcripts, max_size=self.max_vocab_size, device=self.device)
        print('Vocabulary built.')
        print('\tModel Vocab Size:', self.vocabulary.vocab_size)
        print()


    def load_checkpoint_vocabulary(self) -> None:
        print('Loading checkpoint vocabulary...')
        vocabulary_path = Path(self.checkpoint_path, 'vocabulary.pt')
        self.vocabulary = Vocabulary(vocab=torch.load(vocabulary_path), device=self.device)
        print('Vocabulary loaded.')
        print('\tModel Vocab Size:', self.vocabulary.vocab_size)
        print()


    def verify_longest_sequence(self) -> None:
        longest_source_train = 0
        longest_target_train = 0
        longest_source_test = 0
        longest_target_test = 0
        for item in self.data_train:
            source_length = len(item[0])
            target_length = len(self.vocabulary.tokenize_sequence(item[1]))
            longest_source_train = max(source_length, longest_source_train)
            longest_target_train = max(target_length, longest_target_train)
        for item in self.data_test:
            source_length = len(item[0])
            target_length = len(self.vocabulary.tokenize_sequence(item[1]))
            longest_source_test = max(source_length, longest_source_test)
            longest_target_test = max(target_length, longest_target_test)
        print('Longest source length (train):', f'{longest_source_train} (compressed 4x to {longest_source_train // 4})')
        print('Longest target length (train):', longest_target_train)
        print('Longest source length (test):', f'{longest_source_test} (compressed 4x to {longest_source_test // 4})')
        print('Longest target length (test):', longest_target_test)


    def save_config(self) -> None:
        OmegaConf.save(config=self.config, f=Path(self.run_path, 'config.json'))


    @staticmethod
    def collate(batch: Tensor) -> Tensor:
        return batch


    def pad_source(self, source: np.ndarray, max_length: int, mfcc_dim: int) -> Tensor:
        print(type(source))
        source_tensor = torch.tensor(source, device=self.device)
        pad_tensor = torch.zeros((max_length - source_tensor.shape[0], mfcc_dim), device=self.device)
        return torch.cat((source_tensor, pad_tensor))


    def pad_target(self, target: Tensor, max_length: int) -> tuple[Tensor, int]:
        num_pad_tokens = max_length - target.shape[0]
        pad_index = len(target)
        return torch.cat((target, self.vocabulary.pad_token_tensor.repeat(num_pad_tokens))), pad_index


    def padded_source_from_batch(self, batch: Tensor) -> Tensor:
        mfcc_dim = len(batch[0][0][0])
        max_length = self.max_source_length
        if not self.use_fixed_padding:
            lengths = [len(item[0]) for item in batch]
            max_length = max(lengths)

        padded_source = torch.stack(
            [self.pad_source(source=item[0], max_length=max_length, mfcc_dim=mfcc_dim) for item in batch]).to(self.device)
        return padded_source


    def padded_target_from_batch(self, batch: Tensor) -> tuple[Tensor, list[int]]:
        target_indices = list(map(self.vocabulary.build_tokenized_target, [item[1] for item in batch]))
        max_length = self.max_target_length
        if not self.use_fixed_padding:
            lengths = [item.shape[0] for item in target_indices]
            max_length = max(lengths)

        # Produces [(padded_target, pad_index), ...]
        padded = [self.pad_target(target=item, max_length=max_length) for item in target_indices]
        # Convert to ((padded_target, ...), (pad_index, ...)), need to convert to lists
        padded_targets, pad_indices = zip(*padded)

        return torch.stack(list(padded_targets)).to(self.device), list(pad_indices)


    def unpad_and_flatten_batch(self, target_batch: Tensor, prediction_batch: Tensor, pad_indices: list[int]) -> Tensor:
        unpadded_targets = []
        unpadded_predictions = []
        for i in range(len(pad_indices)):
            pad_index = pad_indices[i]
            unpadded_target = target_batch[i][:pad_index]
            unpadded_prediction = prediction_batch[i][:pad_index]
            # Compare against next word in sequence
            unpadded_targets.append(unpadded_target[1:])
            unpadded_predictions.append(unpadded_prediction[:-1])
        return torch.cat(unpadded_targets).to(self.device), torch.cat(unpadded_predictions).to(self.device)


    def flatten_batch(self, target_batch: Tensor, prediction_batch: Tensor) -> Tensor:
        return target_batch.reshape(-1), prediction_batch.reshape(-1, self.vocabulary.vocab_size)


    def check_model_for_randomness(self) -> None:
        print()
        print('Checking model for randomness...')
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
        else:
            print('Success. No randomness detected.')
        print()


    def save_models(self, epoch: int, global_step: int) -> None:
        save_directory = Path(self.run_path, f'models_{epoch}')
        if not Path.exists(save_directory):
            Path.mkdir(save_directory, parents=True)
        torch.save(self.model.state_dict(), f'{save_directory}/model.pt')
        torch.save(self.optimizer.state_dict(), f'{save_directory}/optimizer.pt')
        torch.save(self.vocabulary.vocab, f'{save_directory}/vocabulary.pt')
        with Path(save_directory, 'global_step.json').open('w') as file:
            file.write(json.dumps({ 'global_step': global_step }))


    def save_images(self) -> None:
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


    def train(self) -> None:
        print('-----  Init  -----')
        try:
            self.import_data()
        except FileNotFoundError:
            return
        if self.checkpoint_path is not None:
            if not Path.exists(self.checkpoint_path):
                print('Error: Provided checkpoint path does not exist. Aborting...')
                print()
                return
            self.load_checkpoint_vocabulary()
        else:
            self.build_vocabulary()
        self.verify_longest_sequence()

        # Prepare training data
        train_loader = DataLoader(dataset=self.data_train, batch_size=self.batch_size,
                                  shuffle=True, drop_last=True, collate_fn=self.collate)
        test_loader = DataLoader(dataset=self.data_test, batch_size=self.batch_size,
                                 shuffle=True, drop_last=True, collate_fn=self.collate)
        num_steps = len(train_loader)

        # Build model
        self.model = Transformer(vocabulary=self.vocabulary,
                                 d_model=self.d_model,
                                 batch_size=self.batch_size,
                                 dropout=self.dropout,
                                 num_heads=self.num_heads,
                                 max_source_length=self.max_source_length,
                                 max_target_length=self.max_target_length,
                                 num_layers=self.num_layers,
                                 device=self.device,
                                 mfcc_depth=self.mfcc_depth,
                                 debug=self.debug)
        if torch.cuda.device_count() > 1:
            print('Multiple GPUs detected. GPU count:', torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        if self.checkpoint_path is not None:
            self.model.load_state_dict(torch.load(Path(self.checkpoint_path, 'model.pt')))
        self.check_model_for_randomness()

        # Set optimizer and criterion
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9,
                                          weight_decay=self.weight_decay)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1,
                                              ignore_index=self.vocabulary.pad_token_tensor.item()).to(self.device)

        if self.checkpoint_path is not None:
            self.optimizer.load_state_dict(torch.load(Path(self.checkpoint_path, 'optimizer.pt')))
            if self.reset_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

        # Create LR schedulers
        warmup_scheduler = None
        if self.checkpoint_path is None and self.num_warmup_steps > 0:
            def warmup(step: float) -> float:
                return step / self.num_warmup_steps
            warmup_scheduler = lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=warmup)
        training_scheduler = lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.lr_gamma)
        # training_scheduler = lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
        #                                                     T_max=(self.num_epochs * num_steps) - self.num_warmup_steps,
        #                                                     eta_min=self.lr_min)
        # training_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, factor=1e-4, patience=1)
        scheduler = training_scheduler  # SequentialLR uses deprecated pattern, produces warning

        # Create grad scaler, becomes no-op if enabled is False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Create tensorboard summary writer
        test_path = Path(self.run_path, 'test')
        if not Path.exists(self.run_path):
            Path.mkdir(self.run_path, parents=True)
        if not Path.exists(test_path):
            Path.mkdir(test_path, parents=True)
        self.save_config()
        train_writer = SummaryWriter(self.run_path)
        test_writer = SummaryWriter(test_path)
        graph_source = self.padded_source_from_batch(batch=self.data_train[:self.batch_size])
        graph_target, _ = self.padded_target_from_batch(batch=self.data_train[:self.batch_size])
        print('Creating tensorboard graph...')
        train_writer.add_graph(self.model, (graph_source, graph_target))
        print('Graph created.')

        self.save_models(epoch=0, global_step=0)

        print_every_step = num_steps // self.output_lines_per_epoch
        if print_every_step <= 0:
            print_every_step = num_steps
        test_every_step = num_steps // self.tests_per_epoch if self.tests_per_epoch is not None else None
        if test_every_step <= 0:
            test_every_step = num_steps
        print()
        print()

        print('-----  Training  -----')
        start_step = 0
        if self.checkpoint_path is not None:
            step_path = Path(self.checkpoint_path, 'global_step.json')
            if Path.exists(step_path):
                with step_path.open('r') as file:
                    step_file = json.load(file)
                    start_step = step_file['global_step']
            print('Continuing training from checkpoint model...')
        else:
            print('Starting training...')
        print()

        self.model.train()
        try:
            for epoch in range(self.num_epochs):
                epoch_start = time.time()

                # Train
                running_start = time.time()
                running_count = 0
                running_tokens = 0
                running_loss = 0
                running_error = 0
                for i, batch in enumerate(train_loader):
                    step_start = time.time()
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

                        target_flat, prediction_flat = self.flatten_batch(padded_targets, out)
                        prediction_indices = torch.argmax(prediction_flat, dim=-1)

                        # Calculate loss & perform backprop
                        self.optimizer.zero_grad(set_to_none=True)
                        loss = criterion(prediction_flat, target_flat)
                        # NOTE: Mask pad indices from loss before backward, instead of unpadding before loss calculation
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scheduler.step()

                    elapsed = time.time() - epoch_start
                    step_time = time.time() - step_start
                    running_time = time.time() - running_start

                    running_count += len(batch)
                    running_tokens += len(target_flat)
                    running_loss += loss.item()
                    running_error += torch.sum((prediction_indices != target_flat).float()).item()

                    if self.cooldown is not None and self.cooldown > 0:
                        time.sleep(self.cooldown)

                    if print_every_step is not None and (i + 1) % print_every_step == 0:
                        word_error_rate = running_error / running_tokens
                        running_tokens_per_sec = running_tokens / running_time
                        running_tokens_per_sequence = running_tokens / running_count
                        running_loss_per_sequence = running_loss / running_count

                        train_writer.add_scalar('Metrics/1 WER', word_error_rate, global_step=global_step)
                        train_writer.add_scalar('Metrics/2 Loss (CE)', running_loss_per_sequence, global_step=global_step)
                        train_writer.add_scalar('Metrics/3 LR', scheduler.get_last_lr()[0], global_step=global_step)
                        train_writer.add_scalar('Other/1 Tokens Per Second', running_tokens_per_sec, global_step=global_step)
                        train_writer.add_scalar('Other/2 Tokens Per Sequence', running_tokens_per_sequence, global_step=global_step)
                        train_writer.add_scalar('Other/3 Step Time', step_time, global_step=global_step)
                        train_writer.add_histogram('Vocab Distribution', torch.mean(prediction_flat, dim=0), global_step=global_step)
                        print(f'Epoch: {(epoch + 1):>3}/{self.num_epochs}  {(elapsed // 60):>3.0f}m {(elapsed % 60):>2.0f}s  |  '
                              f'Step: {(i + 1):>4}/{num_steps}  {step_time:>6.2f}s  |  '
                              f'Tokens/sec: {running_tokens_per_sec:>6.1f}  |  '
                              f'Loss: {running_loss_per_sequence:>8.5f}  |  '
                              f'WER: {word_error_rate:>6.1%}  |  '
                              f'LR: {scheduler.get_last_lr()[0]:.2e}')

                        running_start = time.time()
                        running_count = 0
                        running_tokens = 0
                        running_loss = 0
                        running_error = 0

                    if test_every_step is not None and (i + 1) % test_every_step == 0:
                        # Test (Validation)
                        print('Running validation...')
                        test_start = time.time()
                        test_count = 0
                        test_tokens = 0
                        test_loss = 0
                        test_error = 0
                        self.model.eval()
                        for test_batch in test_loader:
                            padded_sources = self.padded_source_from_batch(batch=test_batch)
                            padded_targets, pad_indices = self.padded_target_from_batch(batch=test_batch)

                            with torch.no_grad():
                                # Becomes no-op if self.use_amp is False
                                # NOTE: passing self.device to device_type gives error, keep on 'cuda' even if device is cpu
                                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                                    (out, embedded_source, pos_encoded_source, encoder_out, embedded_target, pos_encoded_target,
                                    target_mask, decoder_out) = \
                                        self.model(encoder_source=padded_sources, decoder_source=padded_targets)

                                    target_flat, prediction_flat = self.unpad_and_flatten_batch(padded_targets, out, pad_indices)
                                    prediction_indices = torch.argmax(prediction_flat, dim=-1)

                                    loss = criterion(prediction_flat, target_flat)

                                    test_count += len(test_batch)
                                    test_tokens += len(target_flat)
                                    test_loss += loss.item()
                                    test_error += torch.sum((prediction_indices != target_flat).float()).item()

                                if self.cooldown is not None and self.cooldown > 0:
                                    time.sleep(self.cooldown)

                        test_elapsed = time.time() - test_start
                        test_word_error_rate = test_error / test_tokens
                        # test_tokens_per_sequence = test_tokens / test_count
                        test_loss_per_sequence = test_loss / test_count

                        test_writer.add_scalar('Metrics/1 WER', test_word_error_rate, global_step=global_step)
                        test_writer.add_scalar('Metrics/2 Loss (CE)', test_loss_per_sequence, global_step=global_step)
                        print(f'VALIDATION RESULTS  |  '
                            f'Loss: {test_loss_per_sequence:>10.5f}  |  '
                            f'WER: {test_word_error_rate:>6.1%}  |  '
                            f'Time: {test_elapsed:>4.2f}s')
                        print()
                        self.model.train()

                # train_writer.add_hparams({ 'lr': self.lr }, { 'WER': word_error_rate }, run_name=str(self.run_path))
                # train_writer.add_embedding for embedding projector

                # Save models & output model sample
                if self.checkpoint_after_epoch is not None and (epoch + 1) % self.checkpoint_after_epoch == 0:
                    self.save_models(epoch=epoch + 1, global_step=global_step)
                    print()
                    print('Models saved.')
                    self.model.eval()

                    # Take random sample from dataset
                    random_index = torch.randint(low=0, high=len(self.data_train), size=(1,)).item()
                    random_sample_source = torch.tensor(self.data_train[random_index][0], device=self.device).unsqueeze(0)
                    random_sample_target = self.vocabulary.build_tokenized_target(self.data_train[random_index][1]).unsqueeze(0)

                    # Copy data across batch size
                    random_sample_source = random_sample_source.expand((self.batch_size,
                                                                        random_sample_source.shape[1], random_sample_source.shape[2]))
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
