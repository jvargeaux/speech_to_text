class Config:
    # Preprocessing
    MODEL_SAMPLE_RATE = 16000
    HOP_LENGTH = 512  # number of samples to shift
    N_FFT = 2048  # number of samples per fft (window size)
    N_MFCC = 13  # standard minimum

    # Model
    D_MODEL = 512
    NUM_HEADS = 4
    DROPOUT = None
    MAX_LENGTH = 5000
    NUM_LAYERS = 3

    # Training
    NUM_EPOCHS = 100
    BATCH_SIZE = 8
    LR = 0.0001
    LR_GAMMA = 0.999
    NUM_WARMUP_STEPS = 0
    COOLDOWN = 10  # every epoch, None = no cooldown
    SUBSET = None  # None = all
    CHECKPOINT_PATH = None  # None = don't use
    RESET_OPTIMIZER = False

    # Output
    OUTPUT_LINES_PER_EPOCH = 20
    CHECKPOINT_AFTER_EPOCH = 5