import math
import numpy as np
import torch
from torch import nn, Tensor


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
        # Add encoding to each element in sequence (token as vectorized work embedding)
        x = x + self.positional_encoding[:x.size(0)]
        return self.dropout(x)


class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()


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
    def __init__(self, vocab_size: int, d_model: int, dropout: float, max_length: int):
        super().__init__()
        self.d_model = d_model
        self.word_embeddings = Embeddings(vocab_size=vocab_size, d_model=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_length=max_length)

        print(self.word_embeddings.embeddings_lut)
        pe = self.positional_encoding.get_buffer('positional_encoding')
        print(pe.size())
        print(pe)


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
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()


class Transformer(nn.Module):
    '''
    According to the original paper "Attention Is All You Need", the transformer
    contains the following modules:

    - Input Embedding (Word Vectorization)
    - Positional Encoding
    - Multi-Head Attention (Scaled Dot-Product Self-Attention)
    
    The vanilla transformer is comprised of an Encoder and a Decoder, which
    share a common weight matrix.
    '''
    def __init__(self, vocab_size: int, d_model: int, dropout: float, max_length: int):
        super().__init__()

        self.encoder = Encoder(vocab_size=vocab_size, d_model=d_model, dropout=dropout, max_length=max_length)
        self.decoder = Decoder(vocab_size=vocab_size, d_model=d_model)

        # Torch
        # nn.Embedding()
        # nn.MultiheadAttention()
        # nn.TransformerEncoder()
        # nn.TransformerDecoder()


class TorchTransformer(nn.Module):
    '''
    A transformer made using pre-existing PyTorch modules.
    '''
    def __init__(self):
        super().__init__()


def main():
    vocab_size = 300
    d_model = 16  # embed_size
    dropout = 0.1
    max_length = 5000
    
    transformer = Transformer(vocab_size=vocab_size, d_model=d_model, dropout=dropout, max_length=max_length)
    

if __name__ == '__main__':
    main()