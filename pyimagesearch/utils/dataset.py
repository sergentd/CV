# import necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import mahotas
import imutils
import pickle
import glob
import h5py
import cv2
import os

def dump_dataset(data, labels, path, datasetName, writeMethod="w"):
    # open the database, create the dataset, write the data and labels to dataset,
    # and then close the database
    db = h5py.File(path, writeMethod)
    dataset = db.create_dataset(datasetName, (len(data), len(data[0]) + 1), dtype="float")
    dataset[0:len(data)] = np.c_[labels, data]
    db.close()

def load_dataset(path, datasetName):
    # open the database, grab the labels and data then close the dataset
    db = h5py.File(path, "r")
    (labels, data) = (db[datasetName][:, 0], db[datasetName][:, 1:])
    db.close()

    # return a tuple of data and labels
    return (data, labels)

def build_cifar10(inputPaths, outputPath, outputFile):
    # loop over the input CIFAR-10 files
    for path in inputPaths:
        data = pickle.loads(open(path, "rb").read(), encodings="latin1")

        # loop over the data
        for (i, image) in enumerate(data["data"]):
            # unpack the RGB components of the image
            (R, G, B) = (image[:1024], image[1024:2048], image[2048:])
            image = np.dstack([B, G, R]).reshape((32, 32, 3))

            # construct the path to the output image file and write it to disk
            p = "{}/{}".format(outputPath, data["filenames"][i])
            cv2.imwrite(p, image)

            # update the training file with the path and class label
            outputFile.write("{} {}\n".format(p, data["labels"][i]))

def load_digits(path):
    # load the dataset and then split it into data and labels
    data = np.genfromtxt(path, delimiter=",",dtype="uint8")
    target = data[:, 0]
    data = data[:, 1:].reshape(data.shape[0], 28, 28)

    # return a tuple of the data and targets
    return (data, target)

def load_house_attributes(path):
    # initialize the list of column names in the CSV file and then
    # load it using pandas
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)

    # determien the (1) unique zip codes and (2) the number of data
    # points with each zipcode
    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()

    # loop over each of the unique zip codes and their corresponding count
    for (zipcode, count) in zip(zipcodes, counts):
        # the zip code counts for our housing dataset is very unbalanced
        # so let's remove the houses with less than 25 hourses per zip code
        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)

        # return the data frame
        return df

def load_house_images(df, path):
    # initialize the images array
    images = []

    # loop over the indexes of the houses
    for i in df.index.values:
        # find the four images for the house and sort the file paths
        # ensuring the four are always in the same order
        basePath = os.path.sep.join([path, "{}_*".format(i + 1)])
        housePaths = sorted(list(glob.glob(basePath)))

        # initialize the list of input images along with the output image
        # after combining the four input images
        inputImages = []
        outputImage = np.zeros((64, 64, 3), dtype="uint8")

        # loop over the input house paths
        for housePath in housePaths:
            image = cv2.imread(housePath)
            image = cv2.resize(image, (32, 32))
            inputImages.append(image)

        # tiles the four input images in the output image
        outputImage[ 0:32,  0:32] = inputImages[0]
        outputImage[ 0:32, 32:64] = inputImages[1]
        outputImage[32:64, 32:64] = inputImages[2]
        outputImage[32:64,  0:32] = inputImages[3]

        # add the tiled image to the set of images the network will
        # be trained on
        images.append(outputImage)

    # return the set of images
    return np.array(images)

def process_house_attributes(df, train, test):
    # initialize the column names of the continuous data
    continuous = ["bedrooms", "bathrooms", "area"]

    # perform the min-max scaling each continuous feature column to
    # the range [0, 1]
    cs.MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])

    # one-hot encode the zip code categorical data
    zipBinarizer = LabelBinarizer().fit(df["zipcode"])
    trainCategorical = zipBinarizer.transform(train["zipcode"])
    testCategorical = zipBinarizer.transform(test["zipcode"])

    # construct our training and testing data points by concatenating
    # the categorical features with the continuous features
    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical, testContinuous])

    # return the concatenated training and testing data
    return (trainX, testX)

def deskew(image, width):
    # grab the width and height of the image and compute the moments
    # for the image
    (h, w) = image.shape[:2]
    moments = cv2.moments(image)

    # deskew the image by applying an affine transformation
    skew = moments["mu11"] / moments["mu02"]
    M = np.float32([
        [1, skew, -0.5 * w * skew],
        [0, 1, 0]])
    image = cv2.warpAffine(image, M, (w, h),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    # resize the image
    image = imutils.resize(image, width=width)

    # return the deskewed image
    return image

def center_extent(image, size):
    # grab the extent width and height
    (eW, eH) = size

    # handle when the width is greater than the height
    if image.shape[1] > image.shape[0]:
        image = imutils.resize(image, width=eW)

    # otherwise, the height is greater than width
    else:
        image = imutils.resize(image, height=eH)

    # allocate memory for the extent of the image and grab it
    extent = np.zeros((eH, eW), dtype="uint8")
    offsetX = (eW - image.shape[1]) // 2
    offsetY = (eH - image.shape[0]) // 2
    extent[offsetY:offsetY + image.shape[0],
        offsetX:offsetX + image.shape[1]] = image

    # compute the center of mass of the image and then move
    # the center of mass in the center of the image
    (cY, cX) = np.round(mahotas.center_of_mass(extent)).astype("int32")
    (dX, dY) = ((size[0] / 2) - cX, size[1] / 2 - cY)
    M = np.float32([[1, 0, dX], [0, 1, dY]])
    extent = cv2.warpAffine(extent, M, size)

    # return the extent of the image
    return extent
