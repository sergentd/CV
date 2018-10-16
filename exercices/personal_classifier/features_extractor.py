# import necessary packages
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import mahotas
import cv2
import sklearn
import os

def describe(image):
  # extract means and standard deviations from each color channel
  (means, stds) = cv2.MeanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
  colorStats = np.concatenate([means, stds]).flatten()
  
  # extract haralick textures
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  haralick = mahotas.features.haralick(gray).mean(axis=0)
  
  # return concatened features vector
  return np.hstack([colorStats, haralick])

# construct the argument parser and parse the arguments  
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input images")
args = vars(ap.parse_args())

# grab the set of image paths and initialize the list of labels and matrix of
# features
print("[INFO] extracting features...")
imagePaths = sorted(paths.list_images(args["dataset"]))
labels = []
data = []

# loop over all images in the dataset
for path in imagePaths:
  # init the label and the image to add to our data
  label = os.path.dirname(path).split(os.path.sep)[-1]
  image = cv2.imread(path)
  
  # extract features from image and store it
  features = describe(image)
  data.append(features)
  labels.append(label)
  
# split into training, validation and testing sets
((trainX, testY), (trainY, testY)) = train_test_split(np.array(data),
  np.array(labels), test_size=0.25, random_state=42)
((trainX, trainY),(valX, valY)) = train_test_split(trainX, trainY,
  test_size=0.1, random_state=84)
  
# create the model
print("[INFO] compiling model...")
model = RandomForestClassifier(n_estimators=20, random_state=96)

# train the model
print("[INFO] training model...")
model.fit(trainX, trainY, validation_data=(valX, valY))

# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(testX)
print(classification_report(trainY, predictions))
