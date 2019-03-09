# import necessary packages
from helpers.io import HDF5DatasetWriter
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--input-path", type=str, required=True,
    help="path to input data")
ap.add_argument("-t", "--train-hdf5", type=str, required=True,
    help="path to HDF5 test dataset")
ap.add_argument("-s", "--test-hdf5", type=str, required=True,
    help="path to HDF5 test dataset")
ap.add_argument("-v", "--val-hdf5", type=str, required=True,
    help="path to HDF5 val dataset")
ap.add_argument("-c", "--num-classes", type=int, default=2,
    help="number of classes we want to recognize")
args = vars(ap.parse_args())

# open the input file for reading and then initialize
# the list of data and labels for the training,
# validation and testing sets
print("[INFO] loading input data...")
f = open(args["input_path"])

f.__next__() # skipping header
(trainX, trainY) = ([], [])
(valX, valY) = ([], [])
(testX, testY) = ([], [])

# loop over the rows
for row in f:
    # extract the label, image and usage from the row
    (label, image, usage) = row.strip().split(",")
    label = int(label)

    # if we are ignoring "disgust" class, there will be
    # a total of 6 classes instead of 7
    if args["num_classes"] == 6:
        # merge the anger and disgust classes
        if label == 1:
            label = 0

        # if label has a value greater than zero, subtract one from
        # it to make all labels sequentials
        if label > 0:
            label -= 1

        # reshape the flattened pixel list into 48x48 (grayscale) image
        image = np.array(image.split(" "), dtype="uint8")
        image = image.reshape((48, 48))

        # check to see if it is training image
        if usage == "Training":
            trainX.append(image)
            trainY.append(label)

        # if not, check if it is a validation image
        elif usage == "PrivateTest":
            valX.append(image)
            valY.append(label)

        # else its a training image
        else:
            testX.append(image)
            testY.append(label)

# construct a list pairing the training, validation and testign images
# along with their corresponding labels and output HDF5 files
datasets = [
    (trainX, trainY, args["train_hdf5"]),
    (valX, valY, args["val_hdf5"]),
    (testX, testY, args["test_hdf5"])
]

# loop over the datasets tuples
for (images, labels, outputPath) in datasets:
    # create HDF5 writer
    print("[INFO] building {}".format(outputPath))
    writer = HDF5DatasetWriter((len(images), 48, 48), outputPath)

    # loop over the umage and add them to the dataset
    for (image, label) in zip(images, labels):
        writer.add([image], [label])

    # close the dataset writer
    writer.close()

# close the input file
f.close()
