# import necessary packages
from utils import AgeGenderHelper
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import progressbar
import argparse
import pickle
import json
import cv2

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d,", "--dataset-type", type=str, required=True,
        help="type of dataset to build (can be either 'age' or 'gender')")
    args = vars(ap.parse_args())

    # check to see if we need to load the 'age' parameters
    if args["dataset_type"] == "age":
        from conf import age_conf as config

    # otherwise, check to see if it is the 'gender' parameters
    elif args["dataset_type"] == "gender":
        from conf import gender_conf as config

    # otherwise, there is no configuration file corresponding to the argument
    else:
        print("no configuration file corresponding to the dataset-type provided")
        return

    # initialize the helper class, then build the set of image paths
    # and class labels
    print("[INFO] building paths and labels...")
    agh = AgeGenderHelper(config)
    (trainPaths, trainLabels) = agh.buildPathsAndLabels()

    # compute the number of images that should be used for training, validation
    # and testing purpose
    numVal = int(len(trainPaths) * config.NUM_VAL_IMAGES)
    numTest = int(len(trainPaths) * config.NUM_TEST_IMAGES)

    # encode the class labels
    print("[INFO] encoding labels...")
    le = LabelEncoder().fit(trainLabels)
    trainLabels = le.transform(trainLabels)

    # perform sampling from the training set to construct a validation set
    print("[INFO] constructing validation data...")
    split = train_test_split(trainPaths, trainLabels, test_size=numVal,
        stratify=trainLabels)
    (trainPaths, valPaths, trainLabels, valLabels) = split

    # perform sampling from the training set to construct a testing set
    print("[INFO] constructing testing data...")
    split = train_test_split(trainPaths, trainLabels, test_size=numTest,
        stratify=trainLabels)
    (trainPaths, testPaths, trainLabels, testLabels) = split

    # construct a list pairing the training, validation and testing image paths
    # along with their corresponding labels and output list files
    datasets = [
        ("train", trainPaths, trainLabels, config.TRAIN_MX_LIST),
        ("val", valPaths, valLabels, config.VAL_MX_LIST),
        ("test", testPaths, testLabels, config.TEST_MX_LIST)
    ]

    # initialize the lists of RGB channel averages
    (R, G, B) = ([], [], [])

    # loop over the dataset tuples
    for (dType, paths, labels, outputPath) in datasets:
        # open the output file for writting
        print("[INFO] building {}".format(outputPath))
        f = open(outputPath, "w")

        # initialize the progressbar
        widgets = ["Building List: ", progressbar.Percentage(), " ",
            progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(paths),
            widgets=widgets).start()

        # loop over each of the individual images and labels
        for (i, (path, label)) in enumerate(zip(paths, labels)):
            # if we are building the training dataset, then compute the
            # mean of each channel in the image, then update the respective lists
            if dType == "train":
                image = cv2.imread(path)
                (b, g, r) = cv2.mean(image)[:3]
                R.append(r)
                G.append(g)
                B.append(b)

            # write the image index, label and output path to file
            row = "\t".join([str(i), str(label), path])
            f.write("{}\n".format(row))
            pbar.update(i)

        # close the output file
        pbar.finish()
        f.close()

    # construct a dictionnary of averages and serialize the means
    # to a JSON file
    print("[INFO] serializing means...")
    D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
    f = open(config.DATASET_MEAN_PATH, "w")
    f.write(json.dumps(D))
    f.close()

    # serialize the label encoder
    print("[INFO] serializing label encoder")
    f = open(config.LABEL_ENCODER_PATH, "wb")
    f.write(pickle.dumps(le))
    f.close()

if __name__ == "__main__":
    main()
