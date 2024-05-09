import argparse
from pathlib import Path
import shutil

from omegaconf import OmegaConf
import torch

from src.trainer import Trainer
from util import pretty_print


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
    config = OmegaConf.load('config.yaml')
    config_override = None
    if Path('config_override.yaml').exists():
        config_override = OmegaConf.load('config_override.yaml')
        config = OmegaConf.merge(config, config_override)

    parser = argparse.ArgumentParser(
        prog='S2T Trainer',
        description='Train the S2T transformer neural network',
        epilog='Epilogue sample text')

    parser.add_argument('--mfcc_depth', '-md', type=int, nargs='?',
                        default=config.audio.mfcc_depth, help='Size of preprocessed mfcc vector')
    parser.add_argument('--d_model', '-dm', type=int, nargs='?',
                        default=config.model.d_model, help='Size of embedding vector')
    parser.add_argument('--num_heads', '-nh', type=int, nargs='?',
                        default=config.model.num_heads, help='Number of attention heads')
    parser.add_argument('--dropout', '-d', type=float, nargs='?',
                        default=config.model.dropout, help='Dropout probability')
    parser.add_argument('--max_source_length', '-ms', type=int, nargs='?',
                        default=config.model.max_source_length, help='Max sequence length of source mfcc data')
    parser.add_argument('--max_target_length', '-mt', type=int, nargs='?',
                        default=config.model.max_target_length, help='Max sequence length of target word tokens')
    parser.add_argument('--max_vocab_size', '-mv', type=int, nargs='?',
                        default=config.model.max_vocab_size, help='Max size of vocabulary')
    parser.add_argument('--num_layers', '-nl', type=int, nargs='?',
                        default=config.model.num_layers, help='Number of encoder/decoder layers')
    parser.add_argument('--batch_size', '-b', type=int, nargs='?',
                        default=config.model.batch_size, help='Size of each batch')
    parser.add_argument('--num_epochs', '-ne', type=int, nargs='?',
                        default=config.training.num_epochs, help='Number of epochs')
    parser.add_argument('--lr', '-lr', type=float, nargs='?',
                        default=config.training.lr, help='Base learning rate')
    parser.add_argument('--lr_gamma', '-lg', type=float, nargs='?',
                        default=config.training.lr_gamma, help='Gamma for learning rate scheduler')
    parser.add_argument('--lr_min', '-lm', type=float, nargs='?',
                        default=config.training.lr_min, help='Min LR for learning rate scheduler')
    parser.add_argument('--weight_decay', '-wd', type=float, nargs='?',
                        default=config.training.weight_decay, help='Weight decay (L2 penalty) for optimizer')
    parser.add_argument('--num_warmup_steps', '-nw', type=int, nargs='?',
                        default=config.training.num_warmup_steps, help='Number of warmup steps in LR scheduler')
    parser.add_argument('--cooldown', '-cd', type=int, nargs='?',
                        default=config.training.cooldown, help='Number of seconds to sleep after each epoch')
    parser.add_argument('--checkpoint_path', '-cp', type=Path, nargs='?',
                        default=config.training.checkpoint_path, help='Save model checkpoint & sample prediction after x number of epochs')
    parser.add_argument('--reset_lr', action='store_true', default=config.training.reset_lr,
                        help='Should reset optimizer when loading checkpoint')
    parser.add_argument('--splits_train', '-st', type=list[str],
                        default=config.training.splits_train, help='Name of dataset splits for training')
    parser.add_argument('--splits_test', '-sv', type=list[str],
                        default=config.training.splits_test, help='Name of dataset splits for testing (validation)')
    parser.add_argument('--subset', '-sub', type=int, nargs='?',
                        default=config.training.subset, help='Use a smaller subset with x number of files. None = use all')
    parser.add_argument('--output_lines_per_epoch', '-le', type=int, nargs='?',
                        default=config.output.output_lines_per_epoch, help='Number of lines of output per epoch')
    parser.add_argument('--checkpoint_after_epoch', '-se', type=int, nargs='?',
                        default=config.output.checkpoint_after_epoch, help='Save model checkpoint & sample prediction after x number of epochs')
    parser.add_argument('--tests_per_epoch', '-te', type=int, nargs='?',
                        default=config.output.tests_per_epoch, help='Number of tests (validations) to perform every epoch')
    parser.add_argument('--debug', action='store_true', help='Run through only one training example for debugging')
    parser.add_argument('--clear_runs', '-c', action='store_true', help='Remove the runs directory to start fresh')
    args = parser.parse_args()

    if args.clear_runs:
        ok_to_continue = clear_runs()
        if not ok_to_continue:
            return

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.backends.mps.is_available():  # some functions not impemented in mps
    #     device = 'mps'
    device = torch.device(device)

    print()
    print()
    if config_override is not None:
        print('Config override detected.')
        print()
    print('-----  Config  -----')
    print('device:', device)
    pretty_print(OmegaConf.to_object(config))
    print()
    print()

    trainer = Trainer(config=config,
                      d_model=args.d_model,
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
                      splits_train=args.splits_train,
                      splits_test=args.splits_test,
                      subset=args.subset)
    trainer.train()

if __name__ == '__main__':
    main()
