import torch
from torch import Tensor
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab


class Vocabulary():
    def __init__(self, batch=None, vocab: Vocab=None, max_size: int | None=None, device=None):
        # Go through all batches of data, and expand the vocabulary
        self.device = device
        self.tokenizer = get_tokenizer('basic_english')
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.vocab = None
        self.vocab_size = 0
        self.max_size = max_size
        if batch is not None:
            self.init_vocab(batch)
        if vocab is not None:
            self.load_vocab(vocab)

    def init_vocab(self, batch):
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, batch),
                                               specials=[self.unk_token, self.sos_token, self.eos_token, self.pad_token],
                                               max_tokens=self.max_size)
        self.vocab.set_default_index(self.vocab[self.unk_token])
        self.vocab_size = len(self.vocab)

    def load_vocab(self, vocab: Vocab):
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

    def tokenize_sequence(self, sequence: str):
        return [self.sos_token] + self.tokenizer(sequence) + [self.eos_token]

    def build_tokenized_target(self, source_sequence: str):
        tokenized = self.tokenize_sequence(source_sequence)
        return torch.tensor(self.vocab(tokenized), dtype=torch.long, device=self.device)

    def get_tensor_from_sequence(self, source_sequence: str):
        return torch.tensor(self.vocab(self.tokenizer(source_sequence)), dtype=torch.long, device=self.device)

    def get_sequence_from_tensor(self, indices: Tensor):
        return self.vocab.lookup_tokens(indices=list(indices))
        # return self.vocab.lookup_token(index)