# import necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from pyimagesearch.nn.conv import LeNet
from pyimagesearch.utils.captchahelper import preprocess
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argumen("-d", "--directory", required=True, help="path to the dataset")
ap.add_argumen("-m", "--model", required=True, help="path to the output model")
args = vars(ap.parse_args())

# initialize the data and labels
data = []
labels = []

# loop over the input images
for imagePath in paths.list_images(args["dataset"]):
  # load the image and preprocess it, then store it in the data list
  image = cv2.imread(imagePath)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = preprocess(image, 28, 28)
  image = img_to_array(image)
  data.append(image)
  
  # extract the class label from the image path and update
  # the labels list
  label = imagePath.split(os.path.sep)[-2]
  labels.append(label)
  
# sclae the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training, validation testing splits
(trainX, testX, trainY, testY) = train_test_split(data,
  labels, test_size=0.25, random_state=42)
(trainX, valX, trainY, valY) = train_test_split(trainX,
  trainY, test_size=0.10, random_state=42)
  
# convert the labels from integers to vectors
lb = LabelBinarizer().fit(trainY)
trainY = lb.transform(trainY)
testY  = lb.transform(testY)
valY   = lb.transform(valY)

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=3, classes=9)
opt = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(valX, valY),
  batch_size=32, epochs=15, verbose=1)
  
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
  predictions.argmax(axis=1), target_names=lb.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["acc"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()