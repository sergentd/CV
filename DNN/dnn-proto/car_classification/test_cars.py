# import necessary packages
from os import path

# define the base path to the cars dataset
BASE_PATH = "/media/djav/djavpass/datasets/cars"

# based on BASE_PATH, construct the full image path
# and meta file path
IMAGES_PATH = path.sep.join([BASE_PATH, "car_ims"])
LABEL_PATH = path.sep.join([BASE_PATH, "complete_dataset.csv"])
