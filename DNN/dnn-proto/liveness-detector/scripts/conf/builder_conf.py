# import necessary packages
from . import deploy_conf as deploy
from os import path

# initialize the list of directories used in the scripts
DETECTOR_DIR = "detector"
DATASET_DIR = "dataset"
SCRIPTS_DIR = "scripts"
OUTPUT_DIR  = "output"
VIDEOS_DIR  = "videos"
MODELS_DIR  = "models"
REAL_DIR    = "real"
FAKE_DIR    = "fake"

# initialize the initial learning rate along with the batch size and the number
# of epochs. initialize the number of skipped frames between two detections
INIT_LR = 1e-4
BATCH_SIZE = 8
EPOCHS = 50
SKIP_FRAMES = 2

# initialize the confidence minimum when filtering predictions
CONFIDENCE_THRESH = deploy.CONFIDENCE_THRESH

# initialize the Caffe model names
PROTO_NAME  = "deploy.prototxt"
CAFFE_NAME  = "res10_300x300_ssd_iter_140000.caffemodel"

# initialize the Caffe face detector paths
PROTO_PATH = path.sep.join([DETECTOR_DIR, PROTO_NAME])
CAFFE_PATH  = path.sep.join([DETECTOR_DIR, CAFFE_NAME])

# initialize the file names of video where the datas are comming from
REAL_VIDEO_NAME  = "real.mov"
FAKE_VIDEO_NAME  = "fake.mp4"

# initialize the video paths for gathering data
REAL_VIDEO_PATH = path.sep.join([VIDEOS_DIR, REAL_VIDEO_NAME])
FAKE_VIDEO_PATH = path.sep.join([VIDEOS_DIR, FAKE_VIDEO_NAME])

# initialize the filename of the training process outputs
PLOT_NAME = "plot.png"
MODEL_NAME = "livenet.model"
LABEL_ENCODER_NAME = "le.pickle"

# initialize the output paths for the training process
PLOT_PATH = path.sep.join([OUTPUT_DIR, PLOT_NAME])
MODEL_PATH = path.sep.join([MODELS_DIR, MODEL_NAME])
LABEL_ENCODER_PATH = path.sep.join([MODELS_DIR, LABEL_ENCODER_NAME])

# /////////////////// BUILD THE FACES DATASETS ////////////////////////////
BUILD_ENABLED = [True, True]
BUILD_SCRIPT  = path.sep.join([SCRIPTS_DIR, "build_dataset.py"])
BUILD_DESC    = ["build the real faces dataset", "build the fake faces dataset"]
BUILD_ARGS    = [[["input", REAL_VIDEO_PATH], ["dataset-type", REAL_DIR]],
                [["input", FAKE_VIDEO_PATH], ["dataset-type", FAKE_DIR]]]


# ///////////////////////////// TRAIN THE MODEL //////////////////////////
TRAIN_ENABLED = True
TRAIN_SCRIPT  = path.sep.join([SCRIPTS_DIR, "train_model.py"])
TRAIN_DESC    = "train the liveness detector model"
TRAIN_ARGS    = [["dataset", DATASET_DIR], ["model", MODEL_PATH],
                ["le", LABEL_ENCODER_PATH],["plot", PLOT_PATH]]
