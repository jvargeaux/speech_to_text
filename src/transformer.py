import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.functional import softmax, log_softmax
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Optional


class Embeddings(nn.Module):
    '''
    Create vectors to store word embeddings, which are the representation
    of semantic meaning or relationships.

    - d_model: Dimension (size) of the word embedding vectors
    - vocab_size: Size of the known vocabulary

    Example
    --
    ```
        |--- vocab_size = 8 ---|
    How [0, 0, 0, x, 0, 0, 0, 0] |
    Are [0, 0, y, 0, 0, 0, 0, 0] sequence_length = 3
    You [0, 0, 0, 0, 0, z, 0, 0] |
    ```

    (d = d_model)

    After embeddings:
    ```
    |-- d = 4 -|
    [0, 0, 0, 0]  | o
    [0, 0, 0, 0]  | o
    [x, x, x, x]  | <-- "How"
    [y, y, y, y]  | <-- "Are"
    [0, 0, 0, 0]  | o           vocab_size = 8
    [z, z, z, z]  | <-- "You"
    [0, 0, 0, 0]  | o
    [0, 0, 0, 0]  | o
    ```

    (o -> unused word removed)

    Result:
    ```
    |-- d = 4 -|
    [x, x, x, x]  | <-- "How"
    [y, y, y, y]  | <-- "Are"   sequence_length = 3
    [z, z, z, z]  | <-- "You"
    ```
    '''
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        # If we have not pre-trained, we have no word vectors yet
        # Initalize randomized word embedding vectors of size embed_dim
        self.embeddings_lut = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        print('[Embedding]')
        # Average out the magnitude of the vectors by multiplying by the square root of the dimensions
        # Ex: [2, 2, 2] (d = 3)
        # Magnitude: sqrt(2^2 + 2^2 + 2^2) = sqrt(3 * 2^2) = sqrt(3) * 2 = sqrt(d) * 2
        return self.embeddings_lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    '''
    Encode word embedding vectors with positional information
    to use in the multi-head self-attention mechanism.

    Formulae
    --
    PE(pos,2i) = sin(pos/10000^(2i/d))
    PE(pos,2i+1) = cos(pos/10000^(2i/d))

    where:
    - `pos` is the index of the sequence token
    - `i` is the index within the word embedding vector
    - `d` is the dimension (size) of the word embedding vectors

    Example
    --
    ```
    [0, 1, 2, 3]

    2i/d
    Evens: [0, 2] -> [0/4, 2/4] -> [0, 0.5]
    Odds: [1, 3] -> [1/4, 3/4] -> [0.25, 0.75]

    Evens: [1/10000^0, 1/10000^0.5] -> [1/1, 1/100] -> [1, 0.01]
    Odds: [1/10000^0.25, 1/10000&0.75] -> [1/10, 1/1000] -> [0.1, 0.001]

    [1, 0.1, 0.01, 0.001] * pos = [0, 1, 2, 3]
    ```

    References
    --
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''
    def __init__(self, d_model: int, dropout: float, max_length: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1)

        # Rearrangement of denominator
        # e^log(x) = x, e^-log(x) = 1/x
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)  # Evens
        positional_encoding[:, 1::2] = torch.cos(position * div_term)  # Odds
        positional_encoding.unsqueeze(0)

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x: Tensor) -> Tensor:
        '''
        Parameters:
        - x: Tensor, shape `[sequence_length, batch_size, embed_dim]`
        '''
        print('[Positional Encoding]')
        # Add encoding to each element in sequence (token as vectorized work embedding)
        x = x + self.positional_encoding[:x.size(0)]
        return self.dropout(x)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == self.d_model

        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.attention = None

    def _attention(self, query, key, value, mask=None, dropout=None):
        # Compute "Scaled Dot Product Attention"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        prob = softmax(scores, dim=-1)
        if dropout is not None:
            prob = dropout(prob)
        return torch.matmul(prob, value), prob

    def forward(self, query, key, value, mask = None):
        print('[Multihead Attention]', 'Shape of Q/K/V:', value.shape)
        N = query.shape[0]  # Batch size
        print('[Multihead Attention]', 'Num heads:', self.num_heads)
        print('[Multihead Attention]', 'Head dimension:', self.head_dim)
        print('[Multihead Attention]', 'Mask:', mask)

        if mask is not None:
            # mask = mask.unsqueeze(1)
            # energy = energy.masked_fill(mask == 0, float('-1e20'))
            pass
        num_batches = 1
        # num_batches = query.size(0)

        # Linear projections
        value = self.linear(value)
        key = self.linear(key)
        query = self.linear(query)

        # 1) Split embedding into self.num_heads pieces and stack in new dimensionality
        # This compacting is done for better dependency & computation optimization (see paper)
        # Reshape: (sequence_length, embed_dim) -> (batch_size, query_length, num_heads, head_dim)
        # Then: (batch_size, query_length, num_heads, head_dim) ->
        #       (query_length, batch_size, num_heads, head_dim)
        value = value.view(num_batches, -1, self.num_heads, self.head_dim).transpose(1, 0)
        key = key.view(num_batches, -1, self.num_heads, self.head_dim).transpose(1, 0)
        query = query.view(num_batches, -1, self.num_heads, self.head_dim).transpose(1, 0)
        print('[Multihead Attention]', 'Shape of Q/K/V after reshape:', value.shape)

        # 2) Apply attention on all projected vectors
        x, self.attention = self._attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) Concat (flatten last 2 dimensions)
        # Essentially, undo the reshape we did earlier
        out = x.transpose(0, 1).contiguous().view(num_batches, -1, self.num_heads * self.head_dim)
        out = self.linear(out)
        return out


class Encoder(nn.Module):
    '''
    Encoder Architecture
    --
    ```
    Input
    ↓
    Input Embedding
    ↓
    Positional Encoding
    ↓
    |-----------------------------|
    | ↓      ↓                    |
    | ↓   Q, K, V                 |
    | ↓   ↓  ↓  ↓                 |
    | ↓ Multi-Head Attention      |
    | ↓        ↓                  |
    | Concat & Normalize          |  Nx layers
    | ↓        ↓                  |
    | ↓ Feed-Forward (Linear)     |
    | ↓        ↓                  |
    | Concat & Normalize          |
    | ↓                           |
    |-----------------------------|
    ↓
    Output
    ```
    '''
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, dropout: float,
                 max_length: int, num_heads: int, forward_expansion: int = 4, mask: Optional[Tensor] = None):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.attention = MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * forward_expansion),
            nn.ReLU(),
            nn.Linear(d_model * forward_expansion, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[Tensor] = None):
        '''
        Pass the inputs (and mask) through the layers in turn.
        '''
        out = x
        for n in range(self.num_layers):
            print('[Encoder]', 'Layer', n+1)
            attention = self.attention(out, out, out, mask)
            out = self.dropout(self.norm1(attention + out))
            forward = self.feed_forward(out)
            out = self.dropout(self.norm2(forward + out))
        print('[Encoder]', out.shape)
        return out


class Decoder(nn.Module):
    '''
    Decoder Architecture
    --
    ```
    Outputs (shifted)
    ↓
    Output Embedding
    ↓
    Positional Encoding
    ↓
    |--------------------------------|
    | ↓      ↓                       |
    | ↓   Q, K, V                    |
    | ↓   ↓  ↓  ↓                    |
    | ↓ Masked Multi-Head Attention  |
    | ↓        ↓                     |
    | Concat & Normalize             |
    | ↓   ↓  ↓--↓----------- Encoder output
    | ↓   Q, K, V                    |
    | ↓   ↓  ↓  ↓                    |
    | ↓ Multi-Head Attention         |  Nx layers
    | ↓        ↓                     |
    | Concat & Normalize             |
    | ↓        ↓                     |
    | ↓ Feed-Forward (Linear)        |
    | ↓        ↓                     |
    | Concat & Normalize             |
    | ↓                              |
    |--------------------------------|
    ↓
    Linear
    ↓
    Softmax
    ↓
    Output Probabilities
    ```
    '''
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, dropout: float, num_heads: int):
        super().__init__()

    def forward(self, target: Tensor, encoder_source: Tensor):
        print('[Decoder]')
        return encoder_source


class Transformer(nn.Module):
    '''
    Construct a transformer model from hyperparameters.

    According to the original paper "Attention Is All You Need", the transformer
    contains the following modules:

    - Input Embedding (Word Vectorization)
    - Positional Encoding
    - Multi-Head Attention (Scaled Dot-Product Self-Attention)

    The vanilla transformer is comprised of an Encoder and a Decoder, which
    share a common weight matrix.
    '''
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, dropout: float,
                 num_heads: int, max_length: int, device):
        super().__init__()

        self.device = device
        self.word_embeddings = Embeddings(vocab_size=vocab_size, d_model=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_length=max_length)
        self.encoder = Encoder(vocab_size=vocab_size, d_model=d_model, dropout=dropout, num_heads=num_heads,
                               max_length=max_length, num_layers=num_layers)
        self.decoder = Decoder(vocab_size=vocab_size, d_model=d_model, dropout=dropout, num_heads=num_heads,
                               num_layers=num_layers)
        self.linear = nn.Linear(d_model, vocab_size)

        # Torch
        # nn.Embedding()
        # nn.MultiheadAttention()
        # nn.TransformerEncoder()
        # nn.TransformerDecoder()

    def forward(self, source: Tensor, target: Tensor):
        print('[Transformer]', 'Size of word embedding matrix:', self.word_embeddings.embeddings_lut)
        print('[Transformer]', 'Size of positional encoding matrix:', self.positional_encoding.get_buffer('positional_encoding').shape)

        # Create the word embedding vectors from source
        embedded_source = self.word_embeddings(source)
        print('[Transformer]', 'Shape of word embedded source:', embedded_source.shape)

        # Append the positional encodings to the vectorized word embeddings
        positional_encoded_source = self.positional_encoding(embedded_source)
        print('[Transformer]', 'Shape of positional encoded source:', positional_encoded_source.shape)

        # Encoder
        encoder_source = self.encoder(positional_encoded_source)
        print('[Transformer]', 'Shape of encoder output:', encoder_source.shape)

        # Decoder
        out = self.decoder(target, encoder_source)
        print('[Transformer]', 'Shape of decoder output:', encoder_source.shape)

        out = log_softmax(self.linear(out), dim=-1)

        return out


class TorchTransformer(nn.Module):
    '''
    A transformer made using pre-existing PyTorch modules.
    '''
    def __init__(self):
        super().__init__()


def main():
    # Hyperparameters
    d_model = 16  # embed_size
    num_heads = 4
    dropout = 0.1
    max_length = 5000
    num_layers = 2
    device = 'cpu'

    # Test first with text input, then audio input

    # Tokenize input
    tokenizer = get_tokenizer('basic_english')
    sequence1 = 'hello there dude this is phenomenal! I cant believe it.'
    sequence2 = 'Here are some additional features.'
    sample_set = [sequence1, sequence2]

    # Go through all batches of data, and expand the vocabulary
    vocab = build_vocab_from_iterator(map(tokenizer, sample_set), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    vocab_size = len(vocab)
    print()
    print('Model vocab size:', vocab_size)
    print()
    # all_words = vocab.get_itos()

    # Convert tokens -> vocab indices -> tensor, index of 0 = <unk> (unfamiliar word)
    source_sequence = sequence1
    source = torch.tensor(vocab(tokenizer(source_sequence)), dtype=torch.long)
    target = source
    print('Sample input:')
    print(source_sequence)
    print(source)
    print('Sequence length:', len(source))
    print()

    # Test sample audio input
    # import os
    # import h5py
    # inputs = []
    # for root, directories, files in os.walk('mfcc'):
    #     file = files[0]
    #     with h5py.File(f'mfcc/{file}', 'r') as file_data:
    #         inputs.append([file_data['mfccs'][:],
    #                        file_data['mfccs'].attrs['sample_rate'],
    #                        file_data['mfccs'].attrs['transcript'],
    #                        file_data['mfccs'].attrs['speaker_id']])

    # Create tokens from MFCC data
    # sample_input = inputs[0]
    # source = torch.tensor(sample_input[0])  # mfccs
    # target = sample_input[2]  # transcript
    # print('Sequence length (tokens):', source.shape[0])
    # print('MFCC dimension:', source.shape[1])
    # print('Target:', target)
    # print()

    # Build model
    transformer = Transformer(vocab_size=vocab_size, d_model=d_model, dropout=dropout, num_heads=num_heads,
                              max_length=max_length, num_layers=num_layers, device=device).to(device)
    out = transformer(source=source, target=target)

    print()
    print('Model Output:\n', out.shape)
    output = out[0][-1]
    print(output)
    guess_index = torch.argmax(output)
    print(guess_index)
    print(vocab.lookup_token(guess_index))
    print()

    # Tokenize output & begin training


if __name__ == '__main__':
    main()