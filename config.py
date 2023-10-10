config = {
    'd_model': 512,
    'num_heads': 8,
    'dropout': 0.1,
    'max_length': 5000,
    'num_layers': 6,

    'num_epochs': 10,
    'batch_size': 1,
    'learning_rate': 1e-4,  # 0.0001
    'lr_gamma': 0.998,

    # Debug
    'num_debug_layers': 1,
    'num_debug_files': 50  # None = all
}