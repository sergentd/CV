# import necessary packages
from sklearn.preprocessing import LabelBinarizer()
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
  help="path to the models directory")
args = vars(ap.parse_args())

# load the testing data and then scale it to the range
# [0, 1]
(testX, testY) = cifar10.load_data()[1]
testX = testX.astype("float") / 255.0

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck"]
 
# convert the labels from integers to vectors 
lb = LabelBinarizer()
testY = lb.fit_transform(testY)

# construct the path used to load the models
modelPaths = os.path.sep.join(args["models"], "*.model")
modelPaths = list(glob.glob(modelPaths))
models = []

# loop over the model paths, load the model and add it to
# the models list
for (i, modelPath) in enumerate(modelPaths):
  print("[INFO] loading model {}\{}".format(i + 1, len(modelPaths)))
  models.append(load_model(modelPath))
  
# initialize a list of predictions
print("[INFO] evaluating ensemble")
predictions = []

# loop over the models
for model in models:
  # use the current model to make predictions on the testing data,
  # then store these predictions in the aggregate predictions list
  predictions.append(model.predict(testX, batch_size=64))
  
# average the probabilities across all model predictions, then show
# a classification report
predictions = np.average(predictions, axis=0)
print(classification_report(testY.argmax(axis=1),
  predictions.argmax(axis=1), target_names=labelNames))
