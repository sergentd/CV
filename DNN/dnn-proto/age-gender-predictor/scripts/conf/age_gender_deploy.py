# import necessary packages
from . import age_conf as ac
from . import gender_conf as gc
from . import builder_conf as bc
from os import path

# define the path to DLIB facial landmark predictor
DLIB_LANDMARK_PATH = "shape_predictor_68_face_landmarks.dat"

# define the path to age checkpoints and label encoder
AGE_NETWORK_PATH = path.sep.join([bc.CHECKPOINTS_DIR, bc.AGE_DATASET])
AGE_PREFIX = bc.AGE_PREFIX
AGE_EPOCH = bc.EVAL_AGE_EPOCH
AGE_LABEL_ENCODER = ac.LABEL_ENCODER_PATH
AGE_MEANS = ac.DATASET_MEAN_PATH

# define the path to gender checkpoints and label encoder
GENDER_NETWORK_PATH = path.sep.join([bc.CHECKPOINTS_DIR, bc.GENDER_DATASET])
GENDER_PREFIX = bc.GENDER_PREFIX
GENDER_EPOCH = bc.EVAL_GENDER_EPOCH
GENDER_LABEL_ENCODER = gc.LABEL_ENCODER_PATH
GENDER_MEANS = gc.DATASET_MEAN_PATH
