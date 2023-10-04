import os
import h5py
import time
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

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

    def train(self, num_epochs: int, batch_size: int, optimizer, learning_rate):
        num_files = 100  # Reduce dataset to subset, None = all
        # Import preprocessed mfcc data
        data = []
        for _, _, files in os.walk('mfcc'):
            if num_files is not None:
                files = files[:num_files]
            for file in files:
                with h5py.File(f'mfcc/{file}', 'r') as file_data:
                    data.append([file_data['mfccs'][:],
                                file_data['mfccs'].attrs['sample_rate'],
                                file_data['mfccs'].attrs['transcript'],
                                file_data['mfccs'].attrs['speaker_id']])
        print('Files:', len(data))

        # sample_input = data[0]
        # mfccs = torch.tensor(sample_input[0])  # mfccs
        # transcript = sample_input[2]  # transcript
        # print(mfccs.shape)
        # print('Number of audio chunks:', mfccs.shape[0])
        # print('MFCC dimension:', mfccs.shape[1])
        # print('Target sequence:', transcript)
        # print('Target sequence length:', len(transcript.split()))

        # Build vocabulary from all transcripts
        transcripts = [item[2] for item in data]
        vocabulary = Vocabulary(batch=transcripts, device=self.device)
        vocab_size = vocabulary.vocab_size
        print('Model Vocab Size:', vocab_size)
        # all_words = vocab.get_itos()
        print()

        # Prepare training data
        train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=self.collate)
        num_steps = len(train_loader)

        # Build model
        self.model = Transformer(vocab_size=vocab_size, d_model=self.d_model, dropout=self.dropout,
                            num_heads=self.num_heads, max_length=self.max_length, num_layers=self.num_layers,
                            device=self.device, debug=self.debug).to(self.device)

        optimizer = optimizer(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

        # NOTE: need to fix data shaping of source & target vector distribution
        # criterion = LabelSmoothing(smoothing=0.1)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)
        
        metrics = Metrics(debug=self.debug)

        self.model.train()
        for epoch in range(num_epochs):
            start = time.time()
            total_tokens = 0
            total_loss = 0
            tokens = 0

            for i, batch in enumerate(train_loader):
                source_mfccs = [torch.tensor(item[0]).to(self.device) for item in batch]
                transcripts = [item[2] for item in batch]
                target_sequences = list(map(vocabulary.get_tensor_from_sequence, transcripts))

                # For now, take only first in batch
                source_mfccs = source_mfccs[0]
                target_sequence = target_sequences[0]

                # Train
                out = self.model(source_mfccs=source_mfccs, target_sequence=target_sequence)
                prediction = out[0]  # batch of 1 for now
                target_y = target_sequence

                pred_x = prediction[:-1]
                target_y = target_sequence[1:]

                # Calculate loss & perform backprop
                loss = criterion(pred_x, target_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # For showing prediction
                prediction_indices = torch.argmax(out, dim=-1)[0]
                prediction_sequence = vocabulary.get_sequence_from_tensor(prediction_indices)

                if self.debug:
                    elapsed = time.time() - start
                    print(f'Epoch: {epoch+1}/{num_epochs} | Step: {i}/{num_steps} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}')

                    print()
                    print('Loss inputs:', prediction.shape, target_y.shape)
                    print()
                    print('Prediction indices:', prediction_indices)
                    print('Prediction sequence:', prediction_sequence)
                    print('Prediction shape:', prediction_indices.shape)
                    print()
                    print('Target indices:', target_y)
                    print('Target sequence:', transcripts[0])
                    print('Target shape:', target_y.shape)
                    print()

                    # metrics.show_confusion_matrix(target=target_y, predicted=prediction_indices)

                    # prediction_no_grad = prediction.clone().detach().requires_grad_(False)
                    # # prediction_display = F.softmax(prediction_no_grad, dim=-1)
                    # metrics.show_heatmap(data=prediction_no_grad, xlabel='Vocab', ylabel='Sequence')

                    return

                if i % 10 == 0:
                    elapsed = time.time() - start
                    print(f'Epoch: {epoch+1}/{num_epochs} | Step: {i}/{num_steps} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.1f}s')
                
                # Show prediction on final step of last epoch
                if epoch == (num_epochs - 1) and i >= (num_steps - 10):
                    print()
                    print('Prediction indices:', prediction_indices)
                    print('Prediction sequence:', prediction_sequence)
                    print()
                    print('Target indices:', target_y)
                    print('Target sequence:', transcripts[0])
                    print()
