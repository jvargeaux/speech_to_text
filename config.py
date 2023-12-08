from splits import SPLITS

class Config:
    # Preprocessing
    MODEL_SAMPLE_RATE = 16000
    HOP_LENGTH = 160  # number of samples to shift  |  16000 / 160 = 100 per second
    N_FFT = 320  # number of samples per fft (window size)
    MFCC_DEPTH = 80

    # Model
    D_MODEL = 512
    NUM_HEADS = 8
    DROPOUT = 0.1
    MAX_SOURCE_LENGTH = 1800  # 50/second => 30s total | max: 1748
    MAX_TARGET_LENGTH = 200   #                        | max: 86
    MAX_VOCAB_SIZE = 40000
    NUM_LAYERS = 6
    BATCH_SIZE = 16

    # Training
    NUM_EPOCHS = 40
    LR = 3e-4
    LR_GAMMA = 0.9998
    LR_MIN = 1e-7
    WEIGHT_DECAY = 1e-4
    NUM_WARMUP_STEPS = 200
    COOLDOWN = 0  # seconds every step, None = no cooldown
    SPLIT_TRAIN = SPLITS.TRAIN_CLEAN_100.value
    SPLIT_TEST = SPLITS.TEST_CLEAN.value
    SUBSET = None  # None = all
    CHECKPOINT_PATH = None  # None = don't use
    RESET_LR = True

    # Output
    OUTPUT_LINES_PER_EPOCH = 100
    CHECKPOINT_AFTER_EPOCH = 1