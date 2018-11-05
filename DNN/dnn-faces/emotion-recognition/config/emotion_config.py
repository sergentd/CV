# import the necessary packages
from os import path

# define the base path to the emotion dataset
BASE_PATH = "/media/djav/djavpass/datasets/fer2013"

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
BATCH_SIZE = 64

# define the path to output logs to be stored
OUTPUT_PATH = path.sep.join([BASE_PATH, "output"])
