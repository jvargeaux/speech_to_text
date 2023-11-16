import argparse
from pathlib import Path
import torch
from torch import nn
from src.trainer import Trainer
from config import Config

config = Config()


def main():
    parser = argparse.ArgumentParser(
        prog='S2T Trainer',
        description='Train the S2T transformer neural network',
        epilog='Epilogue sample text')

    # Hyperparameters from args/config
    parser.add_argument('--d_model', '-dm', type=int, nargs='?',
                        default=config.D_MODEL, help='Size of embedding vector')
    parser.add_argument('--num_heads', '-nh', type=int, nargs='?',
                        default=config.NUM_HEADS, help='Number of attention heads')
    parser.add_argument('--dropout', '-d', type=float, nargs='?',
                        default=config.DROPOUT, help='Dropout probability')
    parser.add_argument('--max_length', '-m', type=int, nargs='?',
                        default=config.MAX_LENGTH, help='Max sequence length of positional encoding matrix')
    parser.add_argument('--num_layers', '-nl', type=int, nargs='?',
                        default=config.NUM_LAYERS, help='Number of encoder/decoder layers')
    parser.add_argument('--mfcc_depth', '-md', type=int, nargs='?',
                        default=config.MFCC_DEPTH, help='Size of preprocessed mfcc vector')
    parser.add_argument('--batch_size', '-b', type=int, nargs='?',
                        default=config.BATCH_SIZE, help='Size of each batch')
    parser.add_argument('--num_epochs', '-ne', type=int, nargs='?',
                        default=config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', '-lr', type=float, nargs='?',
                        default=config.LR, help='Base learning rate')
    parser.add_argument('--lr_gamma', '-lg', type=float, nargs='?',
                        default=config.LR_GAMMA, help='Gamma for learning rate scheduler')
    parser.add_argument('--num_warmup_steps', '-nw', type=int, nargs='?',
                        default=config.NUM_WARMUP_STEPS, help='Number of warmup steps in LR scheduler')
    parser.add_argument('--cooldown', '-cd', type=int, nargs='?',
                        default=config.COOLDOWN, help='Number of seconds to sleep after each epoch')
    parser.add_argument('--output_lines_per_epoch', '-le', type=int, nargs='?',
                        default=config.OUTPUT_LINES_PER_EPOCH, help='Number of lines of output per epoch')
    parser.add_argument('--checkpoint_after_epoch', '-se', type=int, nargs='?',
                        default=config.CHECKPOINT_AFTER_EPOCH, help='Save model checkpoint & sample prediction after x number of epochs')
    parser.add_argument('--checkpoint_path', '-c', type=Path, nargs='?',
                        default=config.CHECKPOINT_PATH, help='Save model checkpoint & sample prediction after x number of epochs')
    parser.add_argument('--reset_lr', '-r', action='store_true', default=config.RESET_LR,
                        help='Should reset optimizer when loading checkpoint')
    parser.add_argument('--split', '-s', type=str,
                        default=config.SPLIT, help='Name of dataset split to use')
    parser.add_argument('--subset', '-sub', type=int, nargs='?',
                        default=config.SUBSET, help='Use a smaller subset with x number of files. None = use all')
    parser.add_argument('--debug', action='store_true', help='Run through only one training example for debugging')
    args = parser.parse_args()

    # Set device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.backends.mps.is_available():  # some functions not impemented in mps
    #     device = 'mps'
    device = torch.device(device)

    # Display config
    print()
    print()
    print('-----  config  -----')
    print('Device:', device)
    print('Embed dimension (d_model):', args.d_model)
    print('Num attention heads:', args.num_heads)
    print('Num encoder/decoder layers:', args.num_layers)
    print('MFCC depth:', args.mfcc_depth)
    print('Batch size:', args.batch_size)
    print('Max sequence length:', args.max_length)
    print('Dropout probability:', args.dropout)
    print('Split:', args.split)
    print('Subset:', args.subset)
    print('Learning rate:', args.lr)
    print('LR gamma:', args.lr_gamma)
    print('Num warmup steps:', args.num_warmup_steps)
    print('Cooldown:', args.cooldown)
    if args.checkpoint_path is not None:
        print('Using checkpoint model:', args.checkpoint_path)
        print('Reset LR:', args.reset_lr)
    print()
    print()

    # Begin training
    trainer = Trainer(d_model=args.d_model,
                      num_layers=args.num_layers,
                      dropout=args.dropout,
                      num_heads=args.num_heads,
                      max_length=args.max_length,
                      batch_size=args.batch_size,
                      mfcc_depth=args.mfcc_depth,
                      device=device,
                      debug=args.debug,
                      num_epochs=args.num_epochs,
                      lr=args.lr,
                      lr_gamma=args.lr_gamma,
                      num_warmup_steps=args.num_warmup_steps,
                      cooldown=args.cooldown,
                      output_lines_per_epoch=args.output_lines_per_epoch,
                      checkpoint_after_epoch=args.checkpoint_after_epoch,
                      checkpoint_path=args.checkpoint_path,
                      reset_lr=args.reset_lr,
                      split=args.split,
                      subset=args.subset)
    trainer.train()

if __name__ == '__main__':
    main()