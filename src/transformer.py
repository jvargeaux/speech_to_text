import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.functional import pad, softmax, log_softmax
from typing import Optional

from src.metrics import Metrics


class Logger():
    def __init__(self, debug=True):
        self.debug = debug

    def __call__(self, *args):
        if self.debug:
            print(*args)

log = Logger(debug=False)
metrics = Metrics(debug=False)


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
    - source: preprocessed MFCC data of shape `(n, d_m)`, where `n` is the number
    of chunks, and `d_m` is the dimension of the MFCC vector
    - target_length (t): length of the target sequence
    - embed_dim (d_v): dimension of the word embedding vector

    (n, d_m) -> (t, d_v)
    Ex: 187 chunks of 13 mfccs, target length of 20 words, word embedding dimension of 64
    (187, 13) -> (20, 64)

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
    def __init__(self, embed_dim: int, device):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
        )

    def make_square(self, source: Tensor):
        # Input shape: (n, mfcc_length)
        dim_ratio = round(math.sqrt(source.shape[0] // source.shape[1]))
        remainder = source.shape[0] % dim_ratio
        pad_amount = dim_ratio - remainder if remainder != 0 else 0
        padded = pad(input=source, pad=(0, 0, 0, pad_amount))
        reshaped = padded.view(-1, source.shape[1] * dim_ratio).contiguous()
        # log('Source shape:', source.shape)
        # log('Padded:', padded.shape)
        return reshaped

    def forward(self, source: Tensor, target_length: int):
        log('[Audio Embedder] Source shape:', source.shape)
        log('[Audio Embedder] Target sequence length:', target_length)
        log('[Audio Embedder] Embed dimension:', self.embed_dim)

        # metrics.add_heatmap(data=source)

        square = self.make_square(source)

        # metrics.add_heatmap(data=square)

        # log('Square:', square.shape)
        out = square.unsqueeze(0)  # Add 1 dimension for conv input channel
        out = self.conv(out)  # Shape: (conv_channels, dim_1, dim_2)
        normalize = nn.LayerNorm(out.shape[-1], device=self.device)
        out = normalize(out)
        # log('Conv & Norm:', out.shape)
        out = out.view(-1, out.shape[-1]).contiguous()  # (conv_channels*dim_1, dim_2)

        # metrics.add_heatmap(data=out)
        
        # log('Flatten:', out.shape)
        linear1 = nn.Linear(in_features=out.shape[1], out_features=target_length, device=self.device)  # dim_2 -> target_length
        out = linear1(out)  # (conv_channels*dim_1, target_length)

        # metrics.add_heatmap(data=out)

        # log('Linear1:', out.shape)
        out = out.transpose(0, 1).contiguous()  # (target_length, conv_channels*dim_1)
        # log('Flip:', out.shape)
        linear2 = nn.Linear(in_features=out.shape[1], out_features=self.embed_dim, device=self.device)  # conv_channels*dim_1 -> embed_dim
        out = linear2(out)  # (target_length, embed_dim)
        # log('Linear2:', out.shape)

        # metrics.add_heatmap(data=out)

        # metrics.show_heatmaps()

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

    def forward(self, x):
        log('[Embedding]')
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
        # positional_encoding = positional_encoding.unsqueeze(0)  # Add 1 dimension for batches

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x: Tensor) -> Tensor:
        '''
        Parameters:
        - x: Tensor, shape `[sequence_length, embed_dim]`
        '''
        log('[Positional Encoding]', x.shape)
        # Add encoding to each element in sequence (token as vectorized work embedding)
        x = x + self.positional_encoding[:x.size(0)]
        return self.dropout(x)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, device):
        super().__init__()
        self.device = device
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
        log('[Multihead Attention] Scores (attention) shape:', scores.shape)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        prob = softmax(scores, dim=-1)
        if dropout is not None:
            prob = dropout(prob)
        return torch.matmul(prob, value), prob

    def forward(self, query, key, value, mask = None):
        N = query.shape[0]  # batch size
        log('[Multihead Attention] Shape of Q/K/V:', value.shape)
        log('[Multihead Attention] Num heads:', self.num_heads)
        log('[Multihead Attention] Head dimension:', self.head_dim)
        log('[Multihead Attention] Batch size:', N)

        if mask is not None:
            mask = mask.unsqueeze(1)
            log('[Multihead Attention] Mask shape:', mask.shape)

        # Linear projections
        value = self.linear(value)
        key = self.linear(key)
        query = self.linear(query)

        # 1) Split embedding into self.num_heads pieces and stack in new dimensionality
        # This compacting is done for better dependency & computation optimization (see paper)
        # Reshape: (sequence_length, embed_dim) -> (batch_size, query_length, num_heads, head_dim)
        # Then: (batch_size, query_length, num_heads, head_dim) ->
        #       (query_length, batch_size, num_heads, head_dim)
        value = value.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        query = query.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        log('[Multihead Attention] Shape of sliced Q/K/V:', value.shape)

        # 2) Apply attention on all projected vectors
        x, self.attention = self._attention(query, key, value, mask=mask, dropout=self.dropout)
        log('[Multihead Attention] Shape of attention output:', x.shape, self.attention.shape)

        # 3) Concat (flatten last 2 dimensions)
        # Essentially, undo the reshape we did earlier
        # Final shape: (batch_size, )
        out = x.transpose(0, 1).contiguous().view(N, -1, self.num_heads * self.head_dim)
        out = self.linear(out)
        log('[Multihead Attention] Shape of concat (flattened) attention:', out.shape)
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
    def __init__(self, d_model: int, num_layers: int, dropout: float, num_heads: int, device, forward_expansion: int = 4,
                 mask: Optional[Tensor] = None):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.attention = MultiheadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, device=device)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * forward_expansion),
            nn.ReLU(),
            nn.Linear(d_model * forward_expansion, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        '''
        Pass the inputs (and mask) through the layers in turn.
        '''
        out = x
        for n in range(self.num_layers):
            log('[Encoder]', 'Layer', n+1)
            attention = self.attention(out, out, out, mask)
            out = self.dropout(self.norm1(attention + out))
            forward = self.feed_forward(out)
            out = self.dropout(self.norm2(forward + out))
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
    def __init__(self, d_model: int, num_layers: int, dropout: float, num_heads: int, device, forward_expansion: int = 4,
                 mask: Optional[Tensor] = None):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.attention = MultiheadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, device=device)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * forward_expansion),
            nn.ReLU(),
            nn.Linear(d_model * forward_expansion, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, encoder_out: Tensor, source_mask: Optional[Tensor] = None,
                target_mask: Optional[Tensor] = None):
        '''
        Pass the inputs (and mask) through the layers in turn.
        '''
        out = x
        for n in range(self.num_layers):
            log('[Decoder]', 'Layer', n+1)
            attention = self.attention(out, out, out, target_mask)
            out = self.dropout(self.norm1(attention + out))
            attention = self.attention(out, encoder_out, encoder_out, source_mask)
            out = self.dropout(self.norm2(attention + out))
            forward = self.feed_forward(out)
            out = self.dropout(self.norm3(forward + out))
        return out


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
                 num_heads: int, max_length: int, device, debug = False):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.debug = debug
        self.audio_embedder = AudioEmbedder(embed_dim=d_model, device=device)
        self.word_embeddings = WordEmbedder(vocab_size=vocab_size, d_model=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_length=max_length)
        self.encoder = Encoder(d_model=d_model, dropout=dropout, num_heads=num_heads, num_layers=num_layers, device=device)
        self.decoder = Decoder(d_model=d_model, dropout=dropout, num_heads=num_heads, num_layers=num_layers, device=device)
        self.linear = nn.Linear(d_model, vocab_size)

        if debug:
            global log
            global metrics
            log = Logger(debug=debug)
            metrics = Metrics(debug=debug)

    def make_mask(self, source: Tensor):
        # Encoder output shape: (batch_size, sequence_length, d_model)
        log('[Mask] Encoder output (source) shape:', source.shape)
        N, target_length, _ = source.shape

        attention_shape = (1, target_length, target_length)  # (1, sequence_length, sequence_length)
        mask = torch.tril(torch.ones(attention_shape)).to(self.device)
        log('[Mask] Attention mask shape:', mask.shape)
        # Final mask shape at multihead attention will be: (batch_size, 1, sequence_length, sequence_length)

        # Calculate tokens?

        # pad = 0
        # target = source[:, :-1]
        # target_y = source[:, 1:]
        # # New target (target_y) shape: (batch_size, sequence_length - 1, d_model)
        # print('[Mask] Target (Target_y) shape:', target.shape)

        # target_mask = (target != pad)
        # # target_mask = (target != pad).unsqueeze(-2)
        # print('[Mask] Target Mask shape:', target_mask.shape)
        # print('[Mask] Target Mask:', target_mask)
        # target_mask = target_mask & mask.type_as(target_mask.data)
        # print('[Mask] Target Mask:', target_mask)

        # num_tokens = (target_y != pad).data.sum()
        # print('[Mask] Num tokens:', num_tokens)

        return mask

    def forward(self, source_mfccs: Tensor, target_sequence: Tensor):
        '''
        Source shape: (batch_size, num_chunks, mfcc_length)
        Target shape: (batch_size, sequence_length)
        '''
        metrics = Metrics(debug=self.debug)

        # NOTE: Need to implement batching for audio embedder, word embedder, and positional encoding

        metrics.add_heatmap(data=source_mfccs, ylabel='Source MFCCs')
        source = self.audio_embedder(source_mfccs, len(target_sequence))
        log('[Transformer] Source shape:', source.shape)
        log('[Transformer] Audio Embedding Output:', source.shape)
        log('[Transformer] Size of word embedding matrix:', self.word_embeddings.embeddings_lut)
        log('[Transformer] Size of positional encoding matrix:', self.positional_encoding.get_buffer('positional_encoding').shape)

        # Encoder
        metrics.add_heatmap(data=source, ylabel='Source audio embedding')
        source = self.positional_encoding(source)
        metrics.add_heatmap(data=source, ylabel='Source pos encoding')
        log('[Transformer] Source shape after positional encoding:', source.shape)
        source = source.unsqueeze(0)  # Add batches: (sequence_length, d_model) -> (batch_size, sequence_length, d_model)
        log('[Transformer] Shape of encoder input:', source.shape)
        encoder_out = self.encoder(x=source, mask=None)
        log('[Transformer] Shape of encoder output:', encoder_out.shape)
        metrics.add_heatmap(data=encoder_out[0], ylabel='Encoder out')

        # target_mask = self.make_mask(source=encoder_out)
        # log('[Transformer] Target mask shape:', target_mask.shape)

        # Decoder
        target = self.word_embeddings(target_sequence)
        log('[Transformer] Target shape after word embeddeding:', target.shape)
        metrics.add_heatmap(data=target, ylabel='Target word embedding')
        target = self.positional_encoding(target)
        log('[Transformer] Target shape after positional encoding:', target.shape)
        metrics.add_heatmap(data=target, ylabel='Target pos encoding')

        target_mask = self.make_mask(source=target.unsqueeze(0))
        log('[Transformer] Target mask shape:', target_mask.shape)
        metrics.add_heatmap(data=target_mask[0], ylabel='Target mask')

        target = target.unsqueeze(0)  # Add batches: (sequence_length, d_model) -> (batch_size, sequence_length, d_model)
        log('[Transformer] Shape of decoder input:', target.shape)
        decoder_out = self.decoder(x=target, encoder_out=encoder_out, source_mask=None, target_mask=target_mask)
        log('[Transformer] Shape of decoder output:', decoder_out.shape)
        metrics.add_heatmap(data=decoder_out[0], ylabel='Decoder out')

        # out = log_softmax(self.linear(decoder_out), dim=-1)
        out = self.linear(decoder_out)  # Trying no softmax
        metrics.add_heatmap(data=out[0], ylabel='Out')
        metrics.show_heatmaps()
        return out