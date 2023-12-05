from splits import SPLITS

class Config:
    # Preprocessing
    MODEL_SAMPLE_RATE = 16000
    HOP_LENGTH = 320  # number of samples to shift  |  16000 / 320 = 50 per second
    N_FFT = 640  # number of samples per fft (window size)
    MFCC_DEPTH = 64

    # Model
    D_MODEL = 512
    NUM_HEADS = 8
    DROPOUT = 0.1
    MAX_SOURCE_LENGTH = 2000  # 50/second => 30s total | max: 1748
    MAX_TARGET_LENGTH = 600   #                        | max: 576
    MAX_VOCAB_SIZE = 18000
    NUM_LAYERS = 4
    BATCH_SIZE = 64

    # Training
    NUM_EPOCHS = 40
    LR = 8e-4
    LR_GAMMA = 0.9998
    LR_MIN = 1e-7
    WEIGHT_DECAY = 1e-3
    NUM_WARMUP_STEPS = 200
    COOLDOWN = 0  # seconds every step, None = no cooldown
    SPLIT_TRAIN = SPLITS.TRAIN_CLEAN_100.value
    SPLIT_TEST = SPLITS.TEST_CLEAN.value
    SUBSET = None  # None = all
    CHECKPOINT_PATH = None  # None = don't use
    RESET_LR = True

    # Output
    OUTPUT_LINES_PER_EPOCH = 40
    CHECKPOINT_AFTER_EPOCH = 1