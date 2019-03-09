# import necessary packages
from os import path

MODELS_DIR = "models"

POINTS_IN_HULL_NAME     = "pts_in_hull.npy"
COLORIZATION_PROTO_NAME = "colorization_deploy_v2.prototxt"
COLORIZATION_MODEL_NAME = "colorization_release_v2.caffemodel"

POINTS_IN_HULL_PATH     = path.sep.join([MODELS_DIR, POINTS_IN_HULL_NAME])
COLORIZATION_PROTO_PATH = path.sep.join([MODELS_DIR, COLORIZATION_PROTO_NAME])
COLORIZATION_MODEL_PATH = path.sep.join([MODELS_DIR, COLORIZATION_MODEL_NAME])

INPUT_WIDTH = 500
CAM_WARMUP = 2.0
