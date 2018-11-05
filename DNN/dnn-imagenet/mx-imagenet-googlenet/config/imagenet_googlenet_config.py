# import necessary packages
from os import path

# define the datasets path
DATASETS_PATH = "/media/djav/djavpass/datasets"

# define the base path to where the ImageNet dataset
# devkit are stored on disk
BASE_PATH = path.sep.join([DATASETS_PATH, "imagenet/ILSVRC2015"])

# based on BASE_PATH, derive the images base path, images set path
# and devkit path
IMAGES_PATH = path.sep.join([BASE_PATH, "Data/CLS-LOC"])
IMAGE_SETS_PATH = path.sep.join([BASE_PATH, "ImageSets/CLS-LOC"])
DEVKIT_PATH = path.sep.join([BASE_PATH, "devkit/data"])

# define the path that maps the 1000 possibles wordnet ids
# to the class label integers
WORD_IDS = path.sep.join([DEVKIT_PATH, "map_clsloc.txt"])

# define the path to the training file that maps the partial
# image filename to integer class labels
TRAIN_LIST = path.sep.join([IMAGE_SETS_PATH, "train_cls.txt"])

# define the paths to validation filenames along with the
# file that contains the ground-truth validation labels
VAL_LIST = path.sep.join([IMAGE_SETS_PATH, "val.txt"])
VAL_LABELS = path.sep.join([DEVKIT_PATH,
    "ILSVRC2015_clsloc_validation_ground_truth.txt"])

# define the path to the validation files that are blacklisted
VAL_BLACKLIST = path.sep.join([DEVKIT_PATH,
    "ILSVRC2015_clsloc_validation_blacklist.txt"])

# define the number of images needed from training set
# to construct testing set
NUM_CLASSES = 1000
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# define the path to output training, validation and testing lists
MX_OUTPUT = path.sep.join([DATASETS_PATH, "imagenet"])
TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, "lists/train.lst"])
VAL_MX_LIST = path.sep.join([MX_OUTPUT, "lists/val.lst"])
TEST_MX_LIST = path.sep.join([MX_OUTPUT, "lists/test.lst"])

# define path to the output training, validation and testing
# image records
TRAIN_MX_REC = path.sep.join([MX_OUTPUT, "rec/train.rec"])
VAL_MX_REC = path.sep.join([MX_OUTPUT, "rec/val.rec"])
TEST_MX_REC = path.sep.join([MX_OUTPUT, "rec/test.rec"])

# define the path to the dataset mean
DATASET_MEAN = "output/imagenet_mean.json"

# define the batch size and number of devices
BATCH_SIZE = 32
NUM_DEVICES = 1
