# import necessary packages
from os import path

DATASET_DIR = "dataset"
SCRIPTS_DIR = "scripts"
HELPERS_DIR = "helpers"
MODELS_DIR  = "models"
OUTPUT_DIR  = "output"
CONFIG_DIR  = "conf"

PREFIX = "vggnet"
EPOCH_TUNE = -1
EPOCH_EVAL = 50

VGG_DIR  = "vgg16"
VGG_NAME = "vgg16"
VGG_PATH = path.sep.join([MODELS_DIR, VGG_DIR, VGG_NAME])

CONFIG_PATH = path.sep.join([CONFIG_DIR, "config.json"])

BUILD_ENABLED = True
BUILD_SCRIPT  = path.sep.join([SCRIPTS_DIR, "build_dataset.py"])
BUILD_DESC    = "build the cars dataset"
BUILD_ARGS    = [["conf", CONFIG_PATH]]

TUNE_ENABLED = True
TUNE_SCRIPT  = path.sep.join([SCRIPTS_DIR, "fine_tune.py"])
TUNE_DESC    = "fine tune the VGG16 network on cars dataset"
TUNE_ARGS    = [["conf", CONFIG_PATH],
               ["prefix", PREFIX],
               ["vgg", VGG_PATH],
               ["start-epoch", EPOCH_TUNE]]

EVAL_ENABLED = True
EVAL_SCRIPT  = path.sep.join([SCRIPTS_DIR, "test_cars.py"])
EVAL_DESC    = "test the model"
EVAL_ARGS    = [["conf", CONFIG_PATH],
                ["prefix", PREFIX
                ["epoch", EPOCH_EVAL]]]
