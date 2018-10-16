# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import necessary packages
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

# set a high recursion limit for Theano
sys.setrecursionlimit(5000)

# set constantes
NUM_EPOCHS = 100
BATCH_SIZE = 128

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
  help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
  help="path to specific model checkpoint to load")
ap.add_argument("-a", "--start-epoch", type=int,
  help="epoch to restart training at")
args = vars(ap.parse_args())

# load the training data and testing data, converting the images
# integers to float
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testY.astype("float")

# apply mean subtraction
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# split the training set into train and val
i = int(len(trainX) * 0.75)
valX = trainX[i:]
valY = trainY[i:]
trainX = trainX[:i]
trainY = trainY[:i] 

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
  horizontal_flip=True, fill_mode="nearest")

# if there is no specific model checkpoint supplied, then initialize
# the network and compile model
if args["model"] is None:
  print("[INFO] compiling model ...")
  opt = SGD(lr=1e-1)
  model = ResNet.build(32, 32, 3, 10, (9,9,9),
    (64, 64, 128, 256), reg=5e-4)
  model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# otherwise, load the model from disk
else:
  print("[INFO] loading {}...".format(args["model"])
  model = load_model(args["model"])
  
  # update the learning rate
  print("[INFO] old learning rate: {}".format(
    K.get_value(model.optimizer.lr)))
  K.set_value(model.optimizer.lr, 1e-5)
  print("[INFO] new learning rate: {}".format(
    K.get_value(model.optimizer.lr)))

# construct the set of callbacks
callbacks = [
  EpochCheckpoint(args["checkpoints"], every=5,
    startAt=args["start_epoch"]),
  TrainingMonitor("output/resnet.png",
    jsonPath="output/resnet.json",
    startAt=args["start_epoch"])
]

# train the network
print("[INFO] training network...")
model.fit_generator(
  aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
  validation_data = (valX, valY),
  steps_per_epoch=len(trainX) // BATCH_SIZE, epochs=NUM_EPOCHS,
  callbacks=callbacks)
)
