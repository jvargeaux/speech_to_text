import copy
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.functional import pad, softmax, log_softmax
from typing import Optional
from src.vocabulary import Vocabulary


class Logger():
    def __init__(self, debug=True):
        self.debug = debug

    def __call__(self, *args):
        if self.debug:
            print(*args)

log = Logger(debug=False)


class AudioEmbedder(nn.Module):
    '''
    Convert MFCC data taken from preprocessed audio samples directly into word
    embedding vectors, skipping the symbol tokenization process (for text).

    x: Samples -> MFCCs -> tokens -> word embeddings
    o: Samples -> MFCCs -> word embeddings

    Additionally, the number of chunks of MFCCs don't match the target sequence length.
    So after convolution, we create a linear projection to form matching source and
    target sequence lengths.

    Arguments
    --
    - source: preprocessed MFCC data of shape `(N, c, d_m)`, where `N` is the number of batches,
    `c` is the number of chunks, and `d_m` is the dimension of the MFCC vector
    - target_length (t): length of the target sequence
    - embed_dim (d_v): dimension of the word embedding vector

    (N, c, d_m) -> (N, t, d_v)
    Ex: 187 chunks of 13 mfccs, target length of 20 words, word embedding dimension of 64
    (1, 187, 13) -> (1, 20, 64)

    ```
      | [o] \\
      | [o]  \\
      | [o]    [o] |
    n | [o]    [o] | t
      | [o]    [o] |
      | [o]  //
      | [o] //
    ```
    '''
    def __init__(self, embed_dim: int, device, mfcc_depth: int):
        super().__init__()
        conv1_depth = 16
        conv2_depth = 64
        linear_in_depth = conv2_depth * (mfcc_depth // 4)  # 64*(d_mfcc/4)
        self.device = device
        self.embed_dim = embed_dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv1_depth, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=conv1_depth, out_channels=conv2_depth, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(linear_in_depth, device=self.device)
        self.fc = nn.Linear(in_features=linear_in_depth, out_features=self.embed_dim, device=self.device)

    def forward(self, source: Tensor):
        # Input shape: (N, seq_len, d_mfcc)
        N = source.shape[0]
        out = source.unsqueeze(1)  # Add dim for conv: (N, 1, seq_len, d_mfcc)
        out = self.conv1(out)  # (N, 16, seq_len/2, d_mfcc/2)
        out = self.conv2(out)  # (N, 64, seq_len/4, d_mfcc/4)
        out = out.transpose(-2, -3)  # (N, seq_len/4, 64, d_mfcc/4)
        out = out.reshape(N, -1, out.shape[-1] * out.shape[-2]).contiguous()  # (N, seq_len/4, 64*(d_mfcc/4))
        out = self.norm(out)
        out = self.fc(out)  # (N, seq_len/4, d_model)
        return out


class WordEmbedder(nn.Module):
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
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor):
        # Average out the magnitude of the vectors by multiplying by the square root of the dimensions
        # Ex: [2, 2, 2] (d = 3)
        # Magnitude: sqrt(2^2 + 2^2 + 2^2) = sqrt(3 * 2^2) = sqrt(3) * 2 = sqrt(d) * 2
        return self.norm(self.embeddings_lut(x) * math.sqrt(self.d_model))


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

    [1, 0.1, 0.01, 0.001] * pos: [0, 1, 2, 3] = [0, 0.1, 0.02, 0.003]
    ```

    References
    --
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''
    def __init__(self, d_model: int, dropout: float | None, max_length: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

        positional_encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1)

        # Rearrangement of denominator
        # e^log(x) = x, e^-log(x) = 1/x
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)  # Evens
        positional_encoding[:, 1::2] = torch.cos(position * div_term)  # Odds

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x: Tensor) -> Tensor:
        '''
        Parameters:
        - x: Tensor, shape `[sequence_length, embed_dim]`
        '''
        x = x + self.positional_encoding[:x.size(1)]  # cut to sequence_length (1), not batch_size (0)
        return self.dropout(x) if self.dropout is not None else x


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float | None, device):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == self.d_model

        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        self.attention = None

    def _attention(self, query: Tensor, key: Tensor, value: Tensor, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            trimmed_mask = mask[:,:,:scores.shape[-2],:scores.shape[-1]]
            scores = scores.masked_fill(trimmed_mask == 0, -1e9)
        prob = softmax(scores, dim=-1)
        if dropout is not None:
            prob = dropout(prob)
        return torch.matmul(prob, value), prob

    def forward(self, query, key, value, mask = None):
        N = query.shape[0]  # batch size

        if mask is not None:
            mask = mask.unsqueeze(1)

        # Linear projections
        value = self.linear(value)
        key = self.linear(key)
        query = self.linear(query)

        # 1) Split embedding into self.num_heads pieces and stack in new dimensionality
        # This compacting is done for better dependency & computation optimization (see paper)
        # Reshape: (batch_size, sequence_length, embed_dim) -> (batch_size, query_length, num_heads, head_dim)
        # Then: (batch_size, query_length, num_heads, head_dim) ->
        #       (batch_size, num_heads, query_length, head_dim)
        value = value.reshape(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        query = query.reshape(N, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 2) Apply attention on all projected vectors
        x, self.attention = self._attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) Concat (flatten last 2 dimensions)
        # Essentially, undo the reshape we did earlier
        # Final shape: (batch_size, )
        out = x.transpose(0, 1).contiguous().view(N, -1, self.num_heads * self.head_dim)
        out = self.linear(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float | None, num_heads: int, device, forward_expansion: int = 4):
        super().__init__()
        self.d_model = d_model
        self.attention = MultiheadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, device=device)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * forward_expansion),
            nn.ReLU(),
            nn.Linear(d_model * forward_expansion, d_model)
        )
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        out = x
        attention = self.attention(out, out, out, mask)
        out = self.dropout(self.norm1(attention + out)) if self.dropout is not None else self.norm1(attention + out)
        forward = self.feed_forward(out)
        out = self.dropout(self.norm2(forward + out)) if self.dropout is not None else self.norm2(forward + out)
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
    def __init__(self, d_model: int, num_layers: int, dropout: float | None, num_heads: int, device, forward_expansion: int = 4,
                 mask: Optional[Tensor] = None):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.encoder_block = EncoderBlock(d_model=d_model, dropout=dropout, num_heads=num_heads,
                                          device=device, forward_expansion=forward_expansion)
        self.encoder_layers = nn.ModuleList([copy.deepcopy(self.encoder_block) for _ in range(num_layers)])

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        '''
        Pass the inputs (and mask) through the layers in turn.
        '''
        out = x
        for layer in self.encoder_layers:
            out = layer(out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float | None, num_heads: int, device, forward_expansion: int = 4):
        super().__init__()
        self.d_model = d_model
        self.attention = MultiheadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, device=device)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * forward_expansion),
            nn.ReLU(),
            nn.Linear(d_model * forward_expansion, d_model)
        )
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x: Tensor, encoder_out: Tensor, source_mask: Optional[Tensor] = None,
                target_mask: Optional[Tensor] = None):
        out = x
        self_attention = self.attention(out, out, out, target_mask)
        out = self.dropout(self.norm1(self_attention + out)) if self.dropout is not None else self.norm1(self_attention + out)
        cross_attention = self.attention(out, encoder_out, encoder_out, source_mask)
        out = self.dropout(self.norm2(cross_attention + out)) if self.dropout is not None else self.norm2(cross_attention + out)
        forward = self.feed_forward(out)
        out = self.dropout(self.norm3(forward + out)) if self.dropout is not None else self.norm3(forward + out)
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
    def __init__(self, d_model: int, num_layers: int, dropout: float | None, num_heads: int, device, forward_expansion: int = 4,
                 mask: Optional[Tensor] = None):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.decoder_block = DecoderBlock(d_model=d_model, dropout=dropout, num_heads=num_heads,
                                          device=device, forward_expansion=forward_expansion)
        self.decoder_layers = nn.ModuleList([copy.deepcopy(self.decoder_block) for _ in range(num_layers)])

    def forward(self, x: Tensor, encoder_out: Tensor, source_mask: Optional[Tensor] = None,
                target_mask: Optional[Tensor] = None):
        '''
        Pass the inputs (and mask) through the layers in turn.
        '''
        out = x
        for layer in self.decoder_layers:
            out = layer(out, encoder_out, source_mask, target_mask)
        return out


class TargetMask(nn.Module):
    def __init__(self, batch_size: int, max_length: int, device):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.mask = nn.Parameter(torch.tril(torch.ones((batch_size, max_length, max_length), requires_grad=False, device=device)))

    def forward(self, target: Tensor, pad_token_tensor: Tensor) -> Tensor:
        N, target_length = target.shape
        mask = self.mask[:self.batch_size, :target_length, :target_length]
        non_padded = (target != pad_token_tensor)  # (N, seq_len)
        non_padded = non_padded.unsqueeze(-1).expand(N, target_length, target_length)  # (N, seq_len, seq_len)

        return torch.logical_and(non_padded, mask)


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
    def __init__(self, vocabulary: Vocabulary, d_model: int, batch_size: int, num_layers: int, dropout: float,
                 num_heads: int, max_length: int, mfcc_depth: int, device, debug = False):
        super().__init__()
        self.vocabulary = vocabulary
        vocab_size = vocabulary.vocab_size
        self.d_model = d_model
        self.device = device
        self.debug = debug
        self.audio_embedder = AudioEmbedder(embed_dim=d_model, device=device, mfcc_depth=mfcc_depth)
        self.word_embeddings = WordEmbedder(vocab_size=vocab_size, d_model=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_length=max_length)
        self.encoder = Encoder(d_model=d_model, dropout=dropout, num_heads=num_heads, num_layers=num_layers, device=device)
        self.decoder = Decoder(d_model=d_model, dropout=dropout, num_heads=num_heads, num_layers=num_layers, device=device)
        self.linear = nn.Linear(d_model, vocab_size)
        self.target_mask = TargetMask(batch_size=batch_size, max_length=max_length, device=device)
        self.pad_token_tensor = self.vocabulary.get_tensor_from_sequence(self.vocabulary.pad_token)

    def source_mask(self, source: Tensor):
        return (source != 0)

    def forward(self, encoder_source: Tensor, decoder_source: Tensor):
        '''
        Source (mfccs) shape: (batch_size, num_chunks, mfcc_length)
        Target (indices) shape: (batch_size, sequence_length)
        '''
        # Encoder
        embedded_source = self.audio_embedder(source=encoder_source)
        # source_mask = self.source_mask(source=embedded_source)
        source_mask = None
        pos_encoded_source = self.positional_encoding(embedded_source)
        encoder_out = self.encoder(x=pos_encoded_source, mask=source_mask)

        # Decoder
        target_mask = self.target_mask(target=decoder_source, pad_token_tensor=self.pad_token_tensor)
        embedded_target = self.word_embeddings(decoder_source)
        pos_encoded_target = self.positional_encoding(embedded_target)
        decoder_out = self.decoder(x=pos_encoded_target, encoder_out=encoder_out, source_mask=source_mask, target_mask=target_mask)

        out = self.linear(decoder_out)
        out = log_softmax(self.linear(decoder_out), dim=-1)

        return (out, embedded_source, pos_encoded_source, encoder_out,
                embedded_target, pos_encoded_target, target_mask, decoder_out)