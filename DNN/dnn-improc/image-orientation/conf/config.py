# import necessar packages
from os import path

OUTPUT_DIR  = "output"
SCRIPTS_DIR = "scripts"

DATASET_DIR = path.sep.join(["/media", "djav", "Data", "datasets"])
BUILD_DATASET_DIR = path.sep.join(["/media", "djav", "Data", "datasets",
    "image-orientation"])

MODEL_NAME = "orientation.model"
MODEL_PATH = path.sep.join([OUTPUT_DIR, MODEL_NAME])

FEATURES_NAME = "features.hdf5"
FEATURES_PATH = path.sep.join([OUTPUT_DIR, FEATURES_NAME])

FEATURES_BATCH_SIZE  = 8
FEATURES_BUFFER_SIZE = 1000

# //////////////// CREATE DATASET PARAMETERS ///////////////
CREATE_ENABLED = True
CREATE_SCRIPT  = path.sep.join([SCRIPTS_DIR, "create_dataset.py"])
CREATE_DESC    = "create a dataset of original image + rotated + angle"
CREATE_ARGS    = [["dataset", DATASET_DIR],
                 ["output", BUILD_DATASET_DIR]]

# ////////////// EXTRACT FEATURES PARAMETERS //////////////
EXTRACT_ENABLED = True
EXTRACT_SCRIPT  = path.sep.join([SCRIPTS_DIR, "extract_features.py"])
EXTRACT_DESC    = "extract features from dataset and save it to HDF5 format"
EXTRACT_ARGS    = [["dataset", BUILD_DATASET_DIR],
                  ["output", FEATURES_PATH],
                  ["batch-size", FEATURES_BATCH_SIZE],
                  ["buffer-size", FEATURES_BUFFER_SIZE]]

# ////////////// TRAIN MODEL PARAMETERS //////////////
TRAIN_ENABLED = True
TRAIN_SCRIPT  = path.sep.join([SCRIPTS_DIR, "train_model.py"])
TRAIN_DESC    = "extract features from dataset and save it to HDF5 format"
TRAIN_ARGS    = [["db", FEATURES_PATH],
                 ["output", MODEL_PATH],
                 ["jobs", -1]]
