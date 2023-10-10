from datetime import datetime
from glob import glob
import h5py
from pathlib import Path
import time
import torch
from torch import nn, Tensor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from preprocess import Preprocessor, SPLITS
from src.transformer import Transformer
from src.metrics import Metrics


class Vocabulary():
    def __init__(self, batch, device):
        # Go through all batches of data, and expand the vocabulary
        self.device = device
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = None
        self.vocab_size = 0
        if batch is not None:
            self.init_vocab(batch)

    def init_vocab(self, batch):
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, batch), specials=['<unk>'])
        # Index of 0 = <unk> (unfamiliar word)
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.vocab_size = len(self.vocab)

    def get_tensor_from_sequence(self, source_sequence: str):
        # Convert tokens -> vocab indices -> tensor
        return torch.tensor(self.vocab(self.tokenizer(source_sequence)), dtype=torch.long).to(self.device)

    def get_sequence_from_tensor(self, indices: Tensor):
        return self.vocab.lookup_tokens(indices=list(indices))
        # return self.vocab.lookup_token(index)


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')  # size_average=False will be deprecated
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.true_dist = None

    def forward(self, x: Tensor, target_sequence: Tensor):
        '''
        x: Model output, shape (batch_size, sequence_length, vocab_size)
        '''
        print('x shape:', x.shape)
        true_dist = x.data.clone()[0]
        size = x.size(-1)
        true_dist.fill_(self.smoothing / (size - 2))
        true_dist.scatter_(1, target_sequence.data.unsqueeze(1), self.confidence)
        true_dist[:, 0] = 0
        mask = torch.nonzero(target_sequence.data == 0)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        # print('True dist:', true_dist)
        return self.criterion(x, true_dist)


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
        files = glob('mfcc/*.hdf5')

        if len(files) == 0:
            print('No preprocessed MFCC folder detected. Preprocessing now...')
            preprocessor = Preprocessor(dataset_url=SPLITS.DEV_CLEAN.value)
            preprocessor.preprocess()
            files = glob('mfcc/*.hdf5')
            print()

        if num_files is not None:
            files = files[:num_files]
        for file in files:
            with h5py.File(file, 'r') as file_data:
                data.append([file_data['mfccs'][:],
                            file_data['mfccs'].attrs['sample_rate'],
                            file_data['mfccs'].attrs['transcript'],
                            file_data['mfccs'].attrs['speaker_id']])
        print('Files:', len(data))

        # Build vocabulary from all transcripts
        transcripts = [item[2] for item in data]
        vocabulary = Vocabulary(batch=transcripts, device=self.device)
        vocab_size = vocabulary.vocab_size
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

        # criterion = LabelSmoothing(smoothing=0.1)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)

        metrics = Metrics(debug=self.debug)

        print('Starting training...')
        print()
        self.model.train()
        for epoch in range(num_epochs):
            start = time.time()
            epoch_loss = 0
            epoch_tokens = 0
            for i, batch in enumerate(train_loader):
                source = torch.stack([torch.tensor(item[0]) for item in batch]).to(self.device)
                transcripts = [item[2] for item in batch]
                target_sequences = torch.stack(list(map(vocabulary.get_tensor_from_sequence, transcripts))).to(self.device)

                out = self.model(source=source, target_sequences=target_sequences)

                # Compare against next word in sequence
                # prediction = out[:-1]
                # target = target_sequence[1:]

                # Compare against same word in sequence
                # prediction = out
                # target = target_sequences

                # Flatten batches
                prediction_flat = out.view(-1, out.shape[-1])
                target_flat = target_sequences.view(-1)

                # Calculate loss & perform backprop
                loss = criterion(prediction_flat, target_flat)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # For showing prediction
                prediction_indices = torch.argmax(out, dim=-1)[0]
                prediction_sequence = vocabulary.get_sequence_from_tensor(prediction_indices)

                epoch_loss += loss.item()
                epoch_tokens += len(target_flat)

                if self.debug:
                    print()
                    print('Out shape:', out.shape)
                    print('Loss inputs:', prediction_flat.shape, target_flat.shape)
                    print()
                    print('Target indices:', target_sequences)
                    print('Target sequence:', transcripts[0])
                    print('Target shape:', target_sequences.shape)
                    print()
                    print('Prediction indices:', prediction_indices)
                    print('Prediction sequence:', prediction_sequence)
                    print('Prediction shape:', prediction_indices.shape)
                    print()

                    # metrics.show_confusion_matrix(target=target, predicted=prediction_indices)

                    # prediction_no_grad = prediction.clone().detach().requires_grad_(False)
                    # # prediction_display = F.softmax(prediction_no_grad, dim=-1)
                    # metrics.show_heatmap(data=prediction_no_grad, xlabel='Vocab', ylabel='Sequence')

                    return

                # Print every x steps
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start
                    avg_loss = epoch_loss / (i + 1)
                    tokens_per_sec = epoch_tokens / elapsed
                    print(f'Epoch: {(epoch+1):>3}/{num_epochs} | Step: {(i+1):>4}/{num_steps} | Tokens/sec: {tokens_per_sec:>6.1f} | Avg. Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Epoch Time: {elapsed:>5.1f}s')

                # Show prediction at end of training
                if (epoch + 1) % 20 == 0 and i >= (num_steps - 3):
                    print()
                    print('Transcript:', transcripts[0].lower())
                    print('Prediction:', ' '.join(prediction_sequence))
                    print()

        print()
        print('Training finished.')
        print()

        # Save model & optimizer
        save_directory = Path(f'models/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
        if not Path.exists(save_directory):
            Path.mkdir(save_directory, parents=True)
        torch.save(self.model.state_dict(), f'{save_directory}/model.pt')
        torch.save(optimizer.state_dict(), f'{save_directory}/optimizer.pt')

