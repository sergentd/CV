# import necessary packages
from os import path

DATASET_DIR = "dataset"
MODELS_DIR  = "models"
OUTPUT_DIR  = "output"
SCRIPT_DIR  = "scripts"

MODEL_NAME = "model.model"
MODEL_PATH = path.sep.join([MODELS_DIR, MODEL_NAME])

# TASK
TASK_ENABLED = True
TASK_SCRIPT  = path.sep.join([SCRIPT_DIR, "script.py"])
TASK_DESC    = "task description"
TASK_ARGS    = [["first-arg", "first-value"],
                ["second-arg", "second-value"]]
