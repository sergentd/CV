# import necessary packages
from config import emotion_config as config
from pyimagesearch.io import HDF5DatasetWriter
import numpy as np

# open the input file for reading and then initialize
# the list of data and labels for the training,
# validation and testing sets
print("[INFO] loading input data...")
f = open(config.INPUT_PATH)

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
    if config.NUM_CLASSES == 6:
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
    (trainX, trainY, config.TRAIN_HDF5),
    (valX, valY, config.VAL_HDF5),
    (testX, testY, config.TEST_HDF5)
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
