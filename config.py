config = {
    'd_model': 512,
    'num_heads': 8,
    'dropout': 0.1,
    'max_length': 5000,
    'num_layers': 6,

    'num_epochs': 200,
    'batch_size': 1,
    'learning_rate': 1e-3,  # 0.0001
    'lr_gamma': 0.999,

    # Debug
    'num_debug_layers': 1,
    'subset_size': 10  # None = all
}