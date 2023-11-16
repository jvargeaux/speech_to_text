from enum import Enum

class SPLITS(Enum):
    DEV_CLEAN = 'dev-clean'
    DEV_OTHER = 'dev-other'
    TRAIN_CLEAN_100 = 'train-clean-100'
    TRAIN_CLEAN_360 = 'train-clean-360'
    TRAIN_OTHER_500 = 'train-other-500'
    TEST_CLEAN = 'test-clean'
    TEST_OTHER = 'test-other'