# import necessary packages
from os import path

# define the base path to datasets
BASE_PATH = path.sep.join(["/home", "djav", "CV", "datasets"])

# path to the INITIAL dataset
ORIG_INPUT_DATASET = path.sep.join([BASE_PATH, "breast_cancer", "orig"])

# path to the builded (train / val / test split) dataset
DATASET_PATH = path.sep.join([BASE_PATH, "breast_cancer", "idc"])

# derive the path to training, validation and testing directories
TRAIN_PATH = path.sep.join([DATASET_PATH, "training"])
VAL_PATH = path.sep.join([DATASET_PATH, "validation"])
TEST_PATH = path.sep.join([DATASET_PATH, "testing"])

# define the amount of data used for splits
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# initialize our number of epochs, initial learning rate, and batch
# size
NUM_EPOCHS = 40
INIT_LR = 1e-2
BS = 32
