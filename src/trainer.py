import os
import h5py
import time
import torch

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from src.transformer import Transformer


class Vocabulary():
    def __init__(self, batch):
        # Go through all batches of data, and expand the vocabulary
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
        return torch.tensor(self.vocab(self.tokenizer(source_sequence)), dtype=torch.long)

    def get_word_from_index(self, index: int):
        return self.vocab.lookup_token(index)


class Trainer():
    def __init__(self, d_model: int, num_layers: int, dropout: float,
                 num_heads: int, max_length: int, device: str):
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.max_length = max_length
        self.device = device

    def train(self, num_epochs: int):
        # Import preprocessed mfcc data
        inputs = []
        for _, _, files in os.walk('mfcc'):
            file = files[0]
            with h5py.File(f'mfcc/{file}', 'r') as file_data:
                inputs.append([file_data['mfccs'][:],
                            file_data['mfccs'].attrs['sample_rate'],
                            file_data['mfccs'].attrs['transcript'],
                            file_data['mfccs'].attrs['speaker_id']])
        sample_input = inputs[0]
        mfccs = torch.tensor(sample_input[0])  # mfccs
        transcript = sample_input[2]  # transcript
        print(mfccs.shape)
        print('Number of audio chunks:', mfccs.shape[0])
        print('MFCC dimension:', mfccs.shape[1])
        print('Target sequence:', transcript)
        print('Target sequence length:', len(transcript.split()))

        # Build vocabulary from all transcripts
        all_sequences = [transcript]
        vocabulary = Vocabulary(batch=all_sequences)
        vocab_size = vocabulary.vocab_size
        print('Model Vocab Size:', vocab_size)
        # all_words = vocab.get_itos()
        print()

        # Build model
        model = Transformer(vocab_size=vocab_size, d_model=self.d_model, dropout=self.dropout,
                            num_heads=self.num_heads, max_length=self.max_length, num_layers=self.num_layers,
                            device=self.device).to(self.device)
        

        for epoch in range(num_epochs):
            print(f'Epoch: {epoch+1}/{num_epochs}')

        # Set source and target
        source_mfccs = mfccs
        target_sequence = vocabulary.get_tensor_from_sequence(transcript)
        
        # Train
        out = model(source_mfccs=source_mfccs, target_sequence=target_sequence)

        print('\nModel Output:', out.shape)
        output = out[0][-1]
        guess_index = torch.argmax(output)
        print(guess_index)
        print(vocabulary.get_word_from_index(guess_index))
        print()