# import necessary packages
from os import path
from . import age_conf
from . import gender_conf

# ////////////////////////////////////////////
# names of datasets, directories and prefix
SCRIPTS_DIR = "scripts"
CHECKPOINTS_DIR  = "checkpoints"
AGE_DATASET    = age_conf.DATASET_TYPE
GENDER_DATASET = gender_conf.DATASET_TYPE
AGE_PREFIX     = "agenet"
GENDER_PREFIX  = "gendernet"
EVAL_AGE_EPOCH = 70
EVAL_GENDER_EPOCH = 40

# ////////////////////////////////////////////
# define paths to output checkpoints models
AGE_CHECKPOINTS    = path.sep.join([CHECKPOINTS_DIR, AGE_DATASET])
GENDER_CHECKPOINTS = path.sep.join([CHECKPOINTS_DIR, GENDER_DATASET])

# ////////////////////////////////////////////
# build the datasets
BUILD_ENABLED = [False, False]
BUILD_SCRIPT  = path.sep.join([SCRIPTS_DIR, "build_dataset.py"])
BUILD_DESC    = ["build the age dataset",
                "build the gender dataset"]
BUILD_ARGS    = [[["dataset-type", AGE_DATASET]],
                [["dataset-type", GENDER_DATASET]]]

# ////////////////////////////////////////////
# create the records for mxnet
REC_ENABLED = [False, False, False, False, False, False]
REC_SCRIPT  = path.sep.join([SCRIPTS_DIR, "im2rec.sh"])
REC_DESC    = ["convert 'age' train images to record",
               "convert 'age' eval images to record",
               "convert 'age' test images to record",
               "convert 'gender' train images to record",
               "convert 'gender' eval images to record",
               "convert 'gender' test images to record"]
REC_ARGS    = [(age_conf.TRAIN_MX_LIST, age_conf.TRAIN_MX_REC),
              (age_conf.VAL_MX_LIST,    age_conf.VAL_MX_REC),
              (age_conf.TEST_MX_LIST,   age_conf.TEST_MX_REC),
              (gender_conf.TRAIN_MX_LIST, gender_conf.TRAIN_MX_REC),
              (gender_conf.VAL_MX_LIST,   gender_conf.VAL_MX_REC),
              (gender_conf.TEST_MX_LIST,  gender_conf.TEST_MX_REC)]

# ////////////////////////////////////////////
# train the models (1 for age, 1 for gender)
TRAIN_ENABLED = [False, False]
TRAIN_SCRIPT  = path.sep.join([SCRIPTS_DIR, "train_model.py"])
TRAIN_DESC    = ["train age model",
                "train gender model"]
TRAIN_ARGS    = [
    (("dataset-type", AGE_DATASET), ("checkpoints", AGE_CHECKPOINTS),
    ("prefix", AGE_PREFIX)),

    (("dataset-type", GENDER_DATASET), ("checkpoints", GENDER_CHECKPOINTS),
    ("prefix", GENDER_PREFIX))]

# ////////////////////////////////////////////
# evaluate the models on a specific checkpoint
EVAL_ENABLED = [False, False]
EVAL_SCRIPT  = path.sep.join([SCRIPTS_DIR, "eval_model.py"])
EVAL_DESC    = ["evaluate the age model", "evaluate the gender model"]
EVAL_ARGS    = [
    (("dataset-type", AGE_DATASET), ("checkpoints", AGE_CHECKPOINTS),
    ("prefix", AGE_PREFIX), ("epoch", EVAL_AGE_EPOCH)),

    (("dataset-type", GENDER_DATASET), ("checkpoints", GENDER_CHECKPOINTS),
    ("prefix", GENDER_PREFIX), ("epoch", EVAL_GENDER_EPOCH))]
