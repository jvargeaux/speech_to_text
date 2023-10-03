import argparse
import os
import torch
from torch import nn
from src.trainer import Trainer


# print(os.getenv('PYTORCH_ENABLE_MPS_FALLBACK'))

# Set to avoid this error:
# NotImplementedError: The operator 'aten::index_fill_.int_Scalar' is not currently
# implemented for the MPS device. If you want this op to be added in priority during
# the prototype phase of this feature, please comment on
# https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set
# the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback
# for this op. WARNING: this will be slower than running natively on MPS.

def main(debug = False):
    print()
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.backends.mps.is_available():  # some functions not impemented in mps
    #     device = 'mps'
    device = torch.device(device)
    print('Device:', device)

    # Model Hyperparameters
    d_model = 512  # embed_size
    num_heads = 8
    dropout = 0.1
    max_length = 5000
    num_layers = 6 if not debug else 1

    # Training
    num_epochs = 8
    batch_size = 1
    optimizer = torch.optim.Adam
    learning_rate = 0.001

    trainer = Trainer(d_model=d_model,
                      num_layers=num_layers,
                      dropout=dropout,
                      num_heads=num_heads,
                      max_length=max_length,
                      device=device,
                      debug=debug)
    trainer.train(num_epochs=num_epochs,
                  batch_size=batch_size,
                  optimizer=optimizer,
                  learning_rate=learning_rate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='S2T Trainer',
        description='Train the S2T transformer neural network',
        epilog='Epilogue sample text')
    parser.add_argument('--debug', action='store_true', help='Run through only one training example for debugging')
    args = parser.parse_args()
    main(debug=args.debug)