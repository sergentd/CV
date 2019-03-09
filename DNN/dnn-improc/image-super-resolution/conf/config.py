# import necessary packages
from os import path

DATASETS_DIR = "dataset"
SCRIPTS_DIR = "scripts"
HELPERS_DIR = "helpers"
MODELS_DIR  = "models"
OUTPUT_DIR  = "output"
IMAGES_DIR  = path.sep.join([OUTPUT_DIR, "images"])
LABELS_DIR  = path.sep.join([OUTPUT_DIR, "labels"])

DATASET_NAME   = "ukbench100"
INPUT_DB_NAME  = "inputs.hdf5"
OUTPUT_DB_NAME = "outputs.hdf5"
MODEL_NAME     = "srcnn.model"
PLOT_NAME      = "plot.png"

DATASET_PATH   = path.sep.join([DATASETS_DIR, DATASET_NAME])
INPUT_DB_PATH  = path.sep.join([OUTPUT_DIR, INPUT_DB_NAME])
OUTPUT_DB_PATH = path.sep.join([OUTPUT_DIR, OUTPUT_DB_NAME])
MODEL_PATH     = path.sep.join([MODELS_DIR, MODEL_NAME])
PLOT_PATH      = path.sep.join([OUTPUT_DIR, PLOT_NAME])

# initialize the batch size and number of epochs for training
BATCH_SIZE = 32
NUM_EPOCHS = 10

# initialize the scale and the input dimensions
SCALE = 2.0
INPUT_DIM = 33

# the label size should ne the output spatial dimensions of
# the srcnn, while the padding ensures we properly crop the labels
# ROI
LABEL_SIZE = 21
PAD = int((INPUT_DIM - LABEL_SIZE) / 2.0)

# the stride controls the step size of the sliding window
STRIDE = 14

BUILD_ENABLED = False
BUILD_SCRIPT  = path.sep.join([SCRIPTS_DIR, "build_dataset.py"])
BUILD_DESC    = "build the dataset"
BUILD_ARGS    = [["input-images", DATASET_PATH],
                ["labels", LABELS_DIR],
                ["output-images", IMAGES_DIR],
                ["input-db", INPUT_DB_PATH],
                ["output-db", OUTPUT_DB_PATH],
                ["scale", SCALE],
                ["input-dim", INPUT_DIM],
                ["stride", STRIDE],
                ["padding", PAD],
                ["label-size", LABEL_SIZE]]

TRAIN_ENABLED = True
TRAIN_SCRIPT  = path.sep.join([SCRIPTS_DIR, "train_model.py"])
TRAIN_DESC    = "train the model to upgrade the resolution of pictures"
TRAIN_ARGS    = [["input-db", INPUT_DB_PATH],
                ["output-db", OUTPUT_DB_PATH],
                ["input-dim", INPUT_DIM],
                ["batch-size", BATCH_SIZE],
                ["epochs", NUM_EPOCHS],
                ["model-path", MODEL_PATH],
                ["plot-path", PLOT_PATH]]
