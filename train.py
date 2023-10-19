import argparse
import torch
from torch import nn
from src.trainer import Trainer
from config import config


def main():
    parser = argparse.ArgumentParser(
        prog='S2T Trainer',
        description='Train the S2T transformer neural network',
        epilog='Epilogue sample text')

    # Hyperparameters from args/config
    parser.add_argument('--d_model', '-dm', type=int, nargs='?', default=config['d_model'], help='Size of embedding vector')
    parser.add_argument('--num_heads', '-nh', type=int, nargs='?', default=config['num_heads'], help='Number of attention heads')
    parser.add_argument('--dropout', '-d', type=float, nargs='?', default=config['dropout'], help='Dropout probability')
    parser.add_argument('--max_length', '-m', type=int, nargs='?', default=config['max_length'], help='Max sequence length')
    parser.add_argument('--num_layers', '-nl', type=int, nargs='?', default=config['num_layers'], help='Number of encoder/decoder layers')
    parser.add_argument('--num_epochs', '-ne', type=int, nargs='?', default=config['num_epochs'], help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, nargs='?', default=config['batch_size'], help='Size of each batch')
    parser.add_argument('--learning_rate', '-lr', type=float, nargs='?', default=config['learning_rate'], help='Base learning rate')
    parser.add_argument('--lr_gamma', '-lg', type=float, nargs='?', default=config['lr_gamma'], help='Gamma for learning rate scheduler')
    parser.add_argument('--num_warmup_steps', '-nw', type=int, nargs='?', default=config['num_warmup_steps'], help='Number of warmup steps in LR scheduler')

    parser.add_argument('--debug', action='store_true',
                        help='Run through only one training example for debugging')
    parser.add_argument('--num_debug_layers', '-nd', type=int, nargs='?', default=config['num_debug_layers'],
                        help='Number of encoder/decoder layers in debug mode')
    parser.add_argument('--subset_size', '-s', type=int, nargs='?', default=config['subset_size'],
                        help='Use a smaller subset with this number of files. None = use all')

    args = parser.parse_args()

    # Set device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.backends.mps.is_available():  # some functions not impemented in mps
    #     device = 'mps'
    device = torch.device(device)

    # Display config
    if args.debug:
        args.num_layers = args.num_debug_layers
        print()
        print('DEBUG ENABLED.')
        if args.subset_size is not None:
            print('Using smaller debug subset.')
    print()
    print('Device:', device)
    print()
    print('-- Hyperparameters --')
    print('Embed dimension (d model):', args.d_model)
    print('Num attention heads:', args.num_heads)
    print('Num encoder/decoder layers:', args.num_layers)
    print('Max sequence length:', args.max_length)
    print('Dropout probability:', args.dropout)
    print('Batch size:', args.batch_size)
    print('Learning rate:', args.learning_rate)
    print('LR gamma:', args.lr_gamma)
    print('Num warmup steps:', args.num_warmup_steps)
    print()

    # Begin training
    optimizer = torch.optim.Adam
    trainer = Trainer(d_model=args.d_model,
                      num_layers=args.num_layers,
                      dropout=args.dropout,
                      num_heads=args.num_heads,
                      max_length=args.max_length,
                      device=device,
                      debug=args.debug)
    trainer.train(num_epochs=args.num_epochs,
                  batch_size=args.batch_size,
                  optimizer=optimizer,
                  learning_rate=args.learning_rate,
                  lr_gamma=args.lr_gamma,
                  num_warmup_steps=args.num_warmup_steps,
                  num_files=args.subset_size)

if __name__ == '__main__':
    main()