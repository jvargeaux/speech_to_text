class Config:
    # Preprocessing
    MODEL_SAMPLE_RATE = 16000
    HOP_LENGTH = 512  # number of samples to shift
    N_FFT = 1024  # number of samples per fft (window size)
    MFCC_DEPTH = 40

    # Model
    D_MODEL = 512
    NUM_HEADS = 4
    DROPOUT = None
    MAX_LENGTH = 5000
    NUM_LAYERS = 4
    BATCH_SIZE = 8

    # Training
    NUM_EPOCHS = 40
    LR = 1e-3
    LR_GAMMA = 0.9997
    NUM_WARMUP_STEPS = 0
    COOLDOWN = 30  # every epoch, None = no cooldown
    SUBSET = None  # None = all
    CHECKPOINT_PATH = None  # None = don't use
    RESET_LR = False

    # Output
    OUTPUT_LINES_PER_EPOCH = 20
    CHECKPOINT_AFTER_EPOCH = 2