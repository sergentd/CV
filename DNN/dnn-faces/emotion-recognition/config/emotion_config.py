# import the necessary packages
from os import path

# define the base path to the emotion dataset
BASE_PATH = "/media/djav/djavpass/datasets/fer2013"

CHECKPOINTS_DIR = "checkpoints"
OUTPUT_DIR = "output"

# use the base path to define the path to the input emotion file
INPUT_PATH = path.sep.join([BASE_PATH, "fer2013", "fer2013.csv"])

# define the number of classes (6 if ignoring disgusting class)
NUM_CLASSES = 6

# define the path to the output training,
# validation and testing HDF5 files
TRAIN_HDF5 = path.sep.join([BASE_PATH, "hdf5", "train.hdf5"])
VAL_HDF5 = path.sep.join([BASE_PATH, "hdf5", "val.hdf5"])
TEST_HDF5 = path.sep.join([BASE_PATH, "hdf5", "test.hdf5"])

# define the batch size
BATCH_SIZE = 32

# define the starting epoch
START_EPOCH = 0

# define the path to output logs to be stored
OUTPUT_PATH = path.sep.join([BASE_PATH, "output"])

BUILD_ENABLED = True
BUILD_SCRIPT  = path.sep.join([SCRIPTS_DIR, "build_dataset.py"])
BUILD_DESC    = "build the dataset"
BUILD_ARGS    = [["input-path", INPUT_PATH],
                ["train-hdf5", TRAIN_HDF5],
                ["val-hdf5", VAL_HDF5],
                ["test-hdf5", TEST_HDF5],
                ["num-classes", NUM_CLASSES]]

TRAIN_ENABLED = True
TRAIN_SCRIPT  = path.sep.join([SCRIPTS_DIR, "train_recognizer.py"])
TRAIN_DESC    = "train the model to recognize emotions"
TRAIN_ARGS    = [["checkpoints", CHECKPOINTS_DIR],
                ["start-epoch", START_EPOCH],
                ["output", OUTPUT_DIR],
                ["train-hdf5", TRAIN_HDF5],
                ["val-hdf5", VAL_HDF5],
                ["test-hdf5", TEST_HDF5],
                ["num-classes", NUM_CLASSES],
                ["batch-size", BATCH_SIZE]]
