# import necessary packages
from os import path

# set the output paths for original frame and deep-art picture
MODELS = path.sep.join(["models", "instance_norm"])
ORIG_DIR = path.sep.join(["/media", "djav", "Data", "PO", "original"])
DEEP_DIR = path.sep.join(["/media", "djav", "Data", "PO", "deep-art"])

# neural style tranfer script parameters
NEURAL_INPUT_WIDTH = 400
NEURAL_OUTPUT_WIDTH = 600

# live stream script parameters
LIVE_LEGEND = True
LIVE_INPUT_WIDTH = 500
LIVE_OUTPUT_WIDTH = 1000
LIVE_EMAIL = True

# compare images script parameters
COMPARE_LEGEND = True
COMPARE_WIDTH = 600

# montage images script parameters
MONTAGE_LEGEND = True
MONTAGE_SIZE = (140, 140)
MONTAGE_TILES = (8, 6)

# other legend parameters
LEGEND_WIDTH = 500
L_LEGEND_HEIGHT = 120
S_LEGEND_HEIGHT = 50

# auto models rotation parameters
AUTO_ROTATION_MODE = False
AUTO_MAX_FRAME = 32
