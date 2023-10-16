import torch
from torch import Tensor
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab


class Vocabulary():
    def __init__(self, batch = None, device = None, vocab: Vocab = None):
        # Go through all batches of data, and expand the vocabulary
        self.device = device
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = None
        self.vocab_size = 0
        if batch is not None:
            self.init_vocab(batch)
        if vocab is not None:
            self.load_vocab(vocab)

    def init_vocab(self, batch):
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, batch), specials=['<unk>'])
        # Index of 0 = <unk> (unfamiliar word)
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.vocab_size = len(self.vocab)

    def load_vocab(self, vocab: Vocab):
        self.vocab = vocab
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.vocab_size = len(self.vocab)

    def get_tensor_from_sequence(self, source_sequence: str):
        # Convert tokens -> vocab indices -> tensor
        return torch.tensor(self.vocab(self.tokenizer(source_sequence)), dtype=torch.long).to(self.device)

    def get_sequence_from_tensor(self, indices: Tensor):
        return self.vocab.lookup_tokens(indices=list(indices))
        # return self.vocab.lookup_token(index)