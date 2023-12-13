from splits import SPLITS

class Config:
    # Preprocessing
    MODEL_SAMPLE_RATE = 16000
    HOP_LENGTH = 160  # number of samples to shift  |  16000 / 160 = 100 per second
    N_FFT = 640  # number of samples per fft (window size)
    MFCC_DEPTH = 80

    # Model
    D_MODEL = 512
    NUM_HEADS = 8
    DROPOUT = 0.3
    MAX_SOURCE_LENGTH = 4000  # 50/second => 30s total | max: 3496
    MAX_TARGET_LENGTH = 200   #                        | max: 86
    MAX_VOCAB_SIZE = 30000
    NUM_LAYERS = 3
    BATCH_SIZE = 4

    # Training
    NUM_EPOCHS = 20
    LR = 3e-5
    LR_GAMMA = 0.9998
    LR_MIN = 1e-7
    WEIGHT_DECAY = 1e-2
    NUM_WARMUP_STEPS = 150
    COOLDOWN = 0  # seconds every step, None = no cooldown
    SPLIT_TRAIN = SPLITS.DEV_CLEAN.value
    SPLIT_TEST = SPLITS.TEST_CLEAN.value
    SUBSET = 1000  # None = all
    CHECKPOINT_PATH = None  # None = don't use
    RESET_LR = True

    # Output
    OUTPUT_LINES_PER_EPOCH = 50
    CHECKPOINT_AFTER_EPOCH = 1
    TESTS_PER_EPOCH = 10