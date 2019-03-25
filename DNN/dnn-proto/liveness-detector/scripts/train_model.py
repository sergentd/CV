# set the matplotlib backend
import matplotlib
matplotlib.use("Agg")

# import necessary packages
from scripts.conf import builder_conf as conf
from utils import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
    help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True,
    help="path to output trained model")
ap.add_argument("-l", "--le", type=str, required=True,
    help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# grab the list of images in the dataset directory, then initialize
# the list of data (i.e images) and class images
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the paths
for imagePath in imagePaths:
    # extract the class label from the filename, load the image,
    # resize it to be a fixed 96x96 pixels ignoring aspect ratio
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))

    # update the data and labels list respectively
    data.append(image)
    labels.append(label)

# convert the data into a numpy array, then preprocess it by
# scaling all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# encode the labels (which are currently strings) as integers
# and one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

# partition the data into training and testing splits using 75%
# of the datas for training and the 25% remaining for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.25, random_state=42)

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=conf.INIT_LR, decay=conf.INIT_LR / conf.EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3,
    classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network for {} epochs...".format(conf.EPOCHS))
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=conf.BATCH_SIZE),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // conf.BATCH_SIZE,
    epochs=conf.EPOCHS)

print("[INFO] evaluate the network...")
predictions = model.predict(testX, batch_size=conf.BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=le.classes_))

print("[INFO] serializing network to '{}'".format(args["model"]))
model.save(args["model"])

# save the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, conf.EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, conf.EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, conf.EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, conf.EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
