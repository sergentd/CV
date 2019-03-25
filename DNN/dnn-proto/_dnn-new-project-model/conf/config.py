# import necessary packages
from os import path

DATASET_DIR = "dataset"
SCRIPTS_DIR = "scripts"
HELPERS_DIR = "helpers"
MODELS_DIR  = "models"
OUTPUT_DIR  = "output"

TASK_ENABLED = True
TASK_SCRIPT  = path.sep.join([SCRIPTS_DIR, "script.py"])
TASK_DESC    = "example of task to add in pypeline"
TASK_ARGS    = [["input", "test-example"],]
