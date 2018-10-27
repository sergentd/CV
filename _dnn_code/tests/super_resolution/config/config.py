# import necessary packages
import os

# define the base dataset path
DATASETS = os.path.sep.join(["..", "..", "datasets"])

# define the path to input images to build the training crops
INPUT_IMAGES = os.path.sep.join([DATASETS, "ukbench100"])

# define the output path for temporary files
BASE_OUTPUT = "output"
IMAGES = os.path.sep.join([BASE_OUTPUT, "images"])
LABELS = os.path.sep.join([BASE_OUTPUT, "labels"])

# define the path to HDF5 files
INPUTS_DB = os.path.sep.join([BASE_OUTPUT, "inputs.hdf5"])
OUTPUTS_DB = os.path.sep.join([BASE_OUTPUT, "outputs.hdf5"])

# define the path to the output model file and the plot files
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "srcnn.model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

# initialize the batch size and number of epochs for training
BATCH_SIZE = 128
NUM_EPOCHS = 10

# initialize the scale and the input dimensions
SCALE = 2.0
INPUT_DIM = 33

# the label size should ne the output spatial dimensions of
# the srcnn, while the padding ensures we properly crop the labels
# ROI
LABEL_SIZE = 21
PAD = int((INPUT_DIM - LABEL_SIZE) / 2.0)

# the stride controls the step size of the sliding window
STRIDE = 14
