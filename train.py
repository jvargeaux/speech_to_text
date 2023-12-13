import argparse
from pathlib import Path
import shutil
import torch
from src.trainer import Trainer
from config import Config


def clear_runs() -> bool:
    print()
    confirm = ''
    while not confirm:
        confirm = input('Remove all runs. Are you sure? (Y/n): ').lower().strip()
    if confirm != 'y':
        print('Understood. Aborting...')
        return False

    runs_path = Path('runs')
    try:
        shutil.rmtree(runs_path)
    except OSError as error:
        print(error)

    return True


def main():
    parser = argparse.ArgumentParser(
        prog='S2T Trainer',
        description='Train the S2T transformer neural network',
        epilog='Epilogue sample text')

    # Hyperparameters from args/Config
    parser.add_argument('--d_model', '-dm', type=int, nargs='?',
                        default=Config.D_MODEL, help='Size of embedding vector')
    parser.add_argument('--num_heads', '-nh', type=int, nargs='?',
                        default=Config.NUM_HEADS, help='Number of attention heads')
    parser.add_argument('--dropout', '-d', type=float, nargs='?',
                        default=Config.DROPOUT, help='Dropout probability')
    parser.add_argument('--max_source_length', '-ms', type=int, nargs='?',
                        default=Config.MAX_SOURCE_LENGTH, help='Max sequence length of source mfcc data')
    parser.add_argument('--max_target_length', '-mt', type=int, nargs='?',
                        default=Config.MAX_TARGET_LENGTH, help='Max sequence length of target word tokens')
    parser.add_argument('--max_vocab_size', '-mv', type=int, nargs='?',
                        default=Config.MAX_VOCAB_SIZE, help='Max size of vocabulary')
    parser.add_argument('--num_layers', '-nl', type=int, nargs='?',
                        default=Config.NUM_LAYERS, help='Number of encoder/decoder layers')
    parser.add_argument('--mfcc_depth', '-md', type=int, nargs='?',
                        default=Config.MFCC_DEPTH, help='Size of preprocessed mfcc vector')
    parser.add_argument('--batch_size', '-b', type=int, nargs='?',
                        default=Config.BATCH_SIZE, help='Size of each batch')
    parser.add_argument('--num_epochs', '-ne', type=int, nargs='?',
                        default=Config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', '-lr', type=float, nargs='?',
                        default=Config.LR, help='Base learning rate')
    parser.add_argument('--lr_gamma', '-lg', type=float, nargs='?',
                        default=Config.LR_GAMMA, help='Gamma for learning rate scheduler')
    parser.add_argument('--lr_min', '-lm', type=float, nargs='?',
                        default=Config.LR_MIN, help='Min LR for learning rate scheduler')
    parser.add_argument('--weight_decay', '-wd', type=float, nargs='?',
                        default=Config.WEIGHT_DECAY, help='Weight decay (L2 penalty) for optimizer')
    parser.add_argument('--num_warmup_steps', '-nw', type=int, nargs='?',
                        default=Config.NUM_WARMUP_STEPS, help='Number of warmup steps in LR scheduler')
    parser.add_argument('--cooldown', '-cd', type=int, nargs='?',
                        default=Config.COOLDOWN, help='Number of seconds to sleep after each epoch')
    parser.add_argument('--output_lines_per_epoch', '-le', type=int, nargs='?',
                        default=Config.OUTPUT_LINES_PER_EPOCH, help='Number of lines of output per epoch')
    parser.add_argument('--checkpoint_after_epoch', '-se', type=int, nargs='?',
                        default=Config.CHECKPOINT_AFTER_EPOCH, help='Save model checkpoint & sample prediction after x number of epochs')
    parser.add_argument('--tests_per_epoch', '-te', type=int, nargs='?',
                        default=Config.TESTS_PER_EPOCH, help='Number of tests (validations) to perform every epoch')
    parser.add_argument('--checkpoint_path', '-cp', type=Path, nargs='?',
                        default=Config.CHECKPOINT_PATH, help='Save model checkpoint & sample prediction after x number of epochs')
    parser.add_argument('--reset_lr', action='store_true', default=Config.RESET_LR,
                        help='Should reset optimizer when loading checkpoint')
    parser.add_argument('--split_train', '-st', type=str,
                        default=Config.SPLIT_TRAIN, help='Name of dataset split for training')
    parser.add_argument('--split_test', '-sv', type=str,
                        default=Config.SPLIT_TEST, help='Name of dataset split for testing (validation)')
    parser.add_argument('--subset', '-sub', type=int, nargs='?',
                        default=Config.SUBSET, help='Use a smaller subset with x number of files. None = use all')
    parser.add_argument('--debug', action='store_true', help='Run through only one training example for debugging')
    parser.add_argument('--clear_runs', '-c', action='store_true', help='Remove the runs directory to start fresh')
    args = parser.parse_args()

    if args.clear_runs:
        ok_to_continue = clear_runs()
        if not ok_to_continue:
            return

    # Set device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.backends.mps.is_available():  # some functions not impemented in mps
    #     device = 'mps'
    device = torch.device(device)

    # Display Config
    print()
    print()
    print('-----  Config  -----')
    print('Device:', device)
    print('Embed dimension (d_model):', args.d_model)
    print('Num attention heads:', args.num_heads)
    print('Num encoder/decoder layers:', args.num_layers)
    print('MFCC depth:', args.mfcc_depth)
    print('Batch size:', args.batch_size)
    print('Max source length:', args.max_source_length)
    print('Max target length:', args.max_target_length)
    print('Max vocab size:', args.max_vocab_size)
    print('Dropout probability:', args.dropout)
    print('Split (train):', args.split_train)
    print('Split (test):', args.split_test)
    print('Subset:', args.subset)
    print('LR:', args.lr)
    print('LR gamma:', args.lr_gamma)
    print('LR min:', args.lr_min)
    print('Weight decay:', args.weight_decay)
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
                      max_source_length=args.max_source_length,
                      max_target_length=args.max_target_length,
                      max_vocab_size=args.max_vocab_size,
                      batch_size=args.batch_size,
                      mfcc_depth=args.mfcc_depth,
                      device=device,
                      debug=args.debug,
                      num_epochs=args.num_epochs,
                      lr=args.lr,
                      lr_gamma=args.lr_gamma,
                      lr_min=args.lr_min,
                      weight_decay=args.weight_decay,
                      num_warmup_steps=args.num_warmup_steps,
                      cooldown=args.cooldown,
                      output_lines_per_epoch=args.output_lines_per_epoch,
                      checkpoint_after_epoch=args.checkpoint_after_epoch,
                      tests_per_epoch=args.tests_per_epoch,
                      checkpoint_path=args.checkpoint_path,
                      reset_lr=args.reset_lr,
                      split_train=args.split_train,
                      split_test=args.split_test,
                      subset=args.subset)
    trainer.train()

if __name__ == '__main__':
    main()