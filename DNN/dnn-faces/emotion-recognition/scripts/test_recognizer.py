# import necessary packages
from helpers.preprocessing import ImageToArrayPreprocessor
from helpers.io import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
    help="path to model checkpoint to load")
ap.add_argument("-t", "--test-hdf5", type=str, required=True,
    help="path to HDF5 test dataset")
ap.add_argument("-b", "--batch-size", type=int, default=32,
    help="size of batches to process in the network")
ap.add_argument("-c", "--num-classes", type=int, default=2,
    help="number of classes we want to recognize")
args = vars(ap.parse_args())

# initialize the testing data generator and image preprocessor
testAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

# initialize the testing dataset generator
testGen = HDF5DatasetGenerator(args["test_hdf5"], args["batch_size"]
    aug=testAug, preprocessors=[iap], classes=args["num_classes"])

# load the model from disk
print("[INFO] loading {}...".format(args["model"]))
model = load_model(args["model"])

(loss, acc) = model.evaluate_generator(
    testGen.generator(),
    steps=testGen.numImages // args["batch_size"],
    max_queue_size=args["batch_size"]*2)

# close the testing database
testGen.close()
