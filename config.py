from splits import SPLITS

class Config(object):
    def __init__(self):
        # Preprocessing
        self.MODEL_SAMPLE_RATE = 16000
        self.HOP_LENGTH = 512  # number of samples to shift
        self.N_FFT = 1024  # number of samples per fft (window size)
        self.MFCC_DEPTH = 48

        # Model
        self.D_MODEL = 512
        self.NUM_HEADS = 4
        self.DROPOUT = 0.1
        self.MAX_LENGTH = 5000
        self.NUM_LAYERS = 2
        self.BATCH_SIZE = 8

        # Training
        self.NUM_EPOCHS = 10
        self.LR = 0.0001
        self.LR_GAMMA = 0.9998
        self.NUM_WARMUP_STEPS = 1000
        self.COOLDOWN = 0  # seconds every step, None = no cooldown
        self.SPLIT = SPLITS.TRAIN_CLEAN_100.value
        self.SUBSET = None  # None = all
        self.CHECKPOINT_PATH = None  # None = don't use
        self.RESET_LR = False

        # Output
        self.OUTPUT_LINES_PER_EPOCH = 200
        self.CHECKPOINT_AFTER_EPOCH = 1