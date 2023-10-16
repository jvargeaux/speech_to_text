config = {
    'd_model': 512,
    'num_heads': 8,
    'dropout': 0.1,
    'max_length': 5000,
    'num_layers': 6,

    'num_epochs': 20,
    'batch_size': 1,
    'learning_rate': 1e-4,  # 0.0001
    'lr_gamma': 0.9998,

    # Debug
    'num_debug_layers': 1,
    'subset_size': 400  # None = all
}