# import necessary packages
from os import path

# DATASET_DIR = path.sep.join(["/media", "djav", "Data", "datasets", "openface"])
DATASET_DIR = "dataset"
IMAGES_DIR  = "images"
SCRIPT_DIR  = "scripts"
OUTPUT_DIR  = "output"
MODELS_DIR  = "models"

PROTO_NAME = "deploy.prototxt"
CAFFE_NAME = "res10_300x300_ssd_iter_140000.caffemodel"
EMB_MODEL_NAME  = "openface_nn4.small2.v1.t7"
RECOGNIZER_NAME = "recognizerv2.pickle"
LABEL_ENCO_NAME = "labelencoder.pickle"
FEATURES_NAME   = "features.pickle"

PROTO_PATH = path.sep.join([MODELS_DIR, PROTO_NAME])
CAFFE_PATH = path.sep.join([MODELS_DIR, CAFFE_NAME])
EMB_MODEL_PATH = path.sep.join([MODELS_DIR, EMB_MODEL_NAME])
FEATURES_PATH  = path.sep.join([OUTPUT_DIR, FEATURES_NAME])
RECOGNIZER_PATH = path.sep.join([OUTPUT_DIR, RECOGNIZER_NAME])
LABEL_ENCO_PATH = path.sep.join([OUTPUT_DIR, LABEL_ENCO_NAME])

# ////////////// EXTRACT EMBEDDINGS SCRIPT /////////////
CONFIDENCE = 0.5

EXTRACT_ENABLED = True
EXTRACT_SCRIPT  = path.sep.join([SCRIPT_DIR, "extract_embeddings.py"])
EXTRACT_DESC    = "extract the embeddings from the faces in images"
EXTRACT_ARGS    = [["dataset", DATASET_DIR],
                  ["embeddings", FEATURES_PATH],
                  ["prototxt", PROTO_PATH],
                  ["caffe-model", CAFFE_PATH],
                  ["embedding-model", EMB_MODEL_PATH],
                  ["confidence", CONFIDENCE]]

# //////////// TRAIN MODEL SCRIPT ///////////////////
TRAIN_ENABLED = True
TRAIN_SCRIPT  = path.sep.join([SCRIPT_DIR, "train_model.py"])
TRAIN_DESC    = "train the recognizer model"
TRAIN_ARGS    = [["embeddings", FEATURES_PATH],
                ["recognizer", RECOGNIZER_PATH],
                ["le", LABEL_ENCO_PATH]]
