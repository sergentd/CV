# import necessary packages
from utils import Conf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import argparse
import progressbar
import pickle
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
    help="path to json configuration file")
args = vars(ap.parse_args())

# read the content of the conf file
conf = Conf(args["conf"])
imagePath = os.path.sep.join([conf["base_path"], conf["images_path"]])
labelPath = os.path.sep.join([conf["base_path"], conf["labels_path"]])

# initialize the list of image paths and labels
print("[INFO] loading images paths and labels...")
rows = open(labelPath).read()
rows = rows.strip().split("\n")[1:]
trainPaths = []
trainLabels = []

# loop over the rows
for row in rows:
    # unpack the row and then update the image paths and labels
    # list
    (filename, make, model) = row.split(",")[:3]
    filename = os.path.basename(filename)
    trainPaths.append(os.path.sep.join([imagePath, filename]))
    trainLabels.append("{}:{}".format(make, model))

# compute the number of images that should be used
# for validation and testing
numVal = int(len(trainPaths) * conf["num_val_images"])
numTest = int(len(trainPaths) * conf["num_test_images"])

# encode the class label from string to vectors
print("[INFO] encoding labels...")
le = LabelEncoder().fit(trainLabels)
trainLabels = le.transform(trainLabels)

# perform sampling from the training set to construct
# a validation set
print("[INFO] constructing validation data...")
split = train_test_split(trainPaths, trainLabels, test_size=numVal,
    stratify=trainLabels)
(trainPaths, valPaths, trainLabels, valLabels) = split

# perform stratified sampling from the training set to construct
# the testing set
print("[INFO] constructing testing data...")
split = train_test_split(trainPaths, trainLabels, test_size=numTest,
    stratify=trainLabels)
(trainPaths, testPaths, trainLabels, testLabels) = split

# construct the output paths for list files
trainList = os.path.sep.join([conf["mx_output"], conf["lists_dir"],
    conf["train_mx_list"]])
valList = os.path.sep.join([conf["mx_output"], conf["lists_dir"],
    conf["val_mx_list"]])
testList = os.path.sep.join([conf["mx_output"], conf["lists_dir"],
    conf["test_mx_list"]])

# construct a list pairing the training, validation and testing
# image paths along with their corresponding labels and output
# list files
datasets = [
    ("train", trainPaths, trainLabels, trainList),
    ("val", valPaths, valLabels, valList),
    ("test", testPaths, testLabels, testList)]

# loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
    # open the output file for writing
    print("[INFO] building {}".format(outputPath))
    f = open(outputPath, "w")

    # initialize the progressbar
    widgets = ["Building List: ", progressbar.Percentage(), " ",
        progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),
        widgets=widgets).start()

    # loop over each of the individual images and labels
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # write the image index, label and output path to file
        row = "\t".join([str(i), str(label), path])
        f.write("{}\n".format(row))
        pbar.update(i)

    # close the output file
    pbar.finish()
    f.close()

# write the label encoder to disk
print("[INFO] serializing label encoder...")
f = open(os.path.sep.join([conf["output_dir"], conf["label_encoder_path"]]),
    "wb")
f.write(pickle.dumps(le))
f.close()
