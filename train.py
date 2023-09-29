from src.trainer import Trainer


def main():
    # Hyperparameters
    d_model = 32  # embed_size
    num_heads = 4
    dropout = 0.1
    max_length = 5000
    num_layers = 2
    device = 'cpu'

    trainer = Trainer(d_model=d_model,
                      num_layers=num_layers,
                      dropout=dropout,
                      num_heads=num_heads,
                      max_length=max_length,
                      device=device)
    trainer.train(num_epochs=2)

if __name__ == '__main__':
    main()