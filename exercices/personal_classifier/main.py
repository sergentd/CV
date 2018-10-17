# import necessary packages
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pyimagesearch.descriptors.featuresextractor import FeaturesExtractor
from imutils import paths
import numpy as np
import progressbar
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments  
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
  help="path to input images")
ap.add_argument("-m", "--model", help="path to serialized model")
args = vars(ap.parse_args())

# grab the set of image paths and initialize the list of labels and matrix of
# features
print("[INFO] extracting features...")
imagePaths = sorted(paths.list_images(args["dataset"]))
labels = []
data = []

# initialize the extractor
descriptor = FeaturesExtractor(["HSVStats", "BGRStats"])

# initialize the progressbar (feedback to user on the task progress)
widgets = ["Features extraction: ", progressbar.Percentage(), " ",
  progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
  widgets=widgets).start()
  
# loop over all images in the dataset
for (i,path) in enumerate(imagePaths):
  # init the label and the image to add to our data
  label = os.path.basename(path).split("_")[0]
  image = cv2.imread(path)
  
  # extract features from image and store it
  features = descriptor.describe(image)
  data.append(features)
  labels.append(label)
  
  # update the progressbar
  pbar.update(i)
  
  if i > 10: break

# close the progressbar
pbar.finish()
  
# split into training, validation and testing sets
(trainX, testX, trainY, testY) = train_test_split(np.array(data),
  np.array(labels), test_size=0.25, random_state=42)
  
# create the model
print("[INFO] compiling model...")
model = RandomForestClassifier(n_estimators=20, random_state=42)

# train the model
print("[INFO] training model...")
model.fit(trainX, trainY)

# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(testX)
print(classification_report(testY, predictions))

# save the model to disk
print("[INFO] serializing the model...")
output = args["model"] if args["model"] is not None else "model.pickle"
f = open(output, 'wb')
f.write(pickle.dumps(model))
f.close()
