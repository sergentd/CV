# import necessary packages
from os import path

# define the type of dataset we are training, can be either "age" or "gender"
DATASET_TYPE = "age"

# define the base path for dataset and output
BASE_PATH  = "/media/djav/Data/datasets/adience"
OUTPUT_DIR = "output"
MX_OUTPUT  = BASE_PATH
LISTS_DIR  = "list"
RECS_DIR   = "rec"

# define the mean pixel intensity files
MEAN_NAME    = "age_adience_mean.json"
DATASET_MEAN_PATH = path.sep.join([OUTPUT_DIR, MEAN_NAME])

# define the age filenames
TRAIN_LST_NAME = "age_train.lst"
VAL_LST_NAME   = "age_val.lst"
TEST_LST_NAME  = "age_test.lst"
TRAIN_REC_NAME = "age_train.rec"
VAL_REC_NAME   = "age_val.rec"
TEST_REC_NAME  = "age_test.rec"

# derive the paths to images and labels from BASE_PATH
IMAGES_NAME = "aligned"
LABELS_NAME = "folds"
IMAGES_PATH = path.sep.join([BASE_PATH, IMAGES_NAME])
LABELS_PATH = path.sep.join([BASE_PATH, LABELS_NAME])

# define the percentage of validation and testing image
# relative to the training set
NUM_VAL_IMAGES  = 0.15
NUM_TEST_IMAGES = 0.15

# define the batch size and the number of GPU to use
BATCH_SIZE  = 32
NUM_DEVICES = 1

# define the number of labels for the "age" dataset, along with
# the path to the label encoder
NUM_CLASSES = 8
LABEL_ENCODER_PATH = path.sep.join([OUTPUT_DIR, "age_le.pickle"])

# define the paths to the output training, validation and testing lists
TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, LISTS_DIR, TRAIN_LST_NAME])
VAL_MX_LIST   = path.sep.join([MX_OUTPUT, LISTS_DIR, VAL_LST_NAME])
TEST_MX_LIST  = path.sep.join([MX_OUTPUT, LISTS_DIR, TEST_LST_NAME])

# define the paths to the mx records
TRAIN_MX_REC = path.sep.join([MX_OUTPUT, RECS_DIR, TRAIN_REC_NAME])
VAL_MX_REC   = path.sep.join([MX_OUTPUT, RECS_DIR, VAL_REC_NAME])
TEST_MX_REC  = path.sep.join([MX_OUTPUT, RECS_DIR, TEST_REC_NAME])
