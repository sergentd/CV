# import necessary packages
import os

# define the paths to the image directory
BASE_PATH = "/media/djav/djavpass/datasets/kaggle_dogs_vs_cats"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "train"])
# using training data as testing
NUM_CLASSES = 2
NUM_VAL_IMAGES  = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# define the path to the output training, validation and testing HDF5 files
TRAIN_HDF5 = os.path.sep.join([BASE_PATH, "hdf5", "train.hdf5"])
VAL_HDF5   = os.path.sep.join([BASE_PATH, "hdf5", "val.hdf5"])
TEST_HDF5  = os.path.sep.join([BASE_PATH, "hdf5", "test.hdf5"])

# path to the output model file
MODEL_PATH = os.path.sep.join(["output", "alexnet_dogs_vs_cats.model"])

# define the path to the dataset mean
DATASET_MEAN = os.path.sep.join(["output", "dogs_vs_cats_mean.json"])

# define the path to the output directory used for
# storing plots, classification reports etc.
OUTPUT_PATH = "output"

# size of batches
BATCH_SIZE = 2
