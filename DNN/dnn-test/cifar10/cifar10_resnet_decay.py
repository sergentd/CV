# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import necessary packages
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import sys
import os

# set a high recursion limit for Theano
sys.setrecursionlimit(5000)

# define the total number of epochs to train for along
# with the initial learning rate
NUM_EPOCHS = 100
BATCH_SIZE = 128
INIT_LR = 1e-1

def poly_decay(epoch):
  # initialize the maximum number of epochs, base learning
  # rate and power of polynomial
  maxEpochs = NUM_EPOCHS
  baseLR = INIT_LR
  power = 1.0
  
  # compute the new learning rate based on polynomial decay
  alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
  
  # return the new learning rate
  return alpha

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
  help="path to output model")
ap.add_argument("-o", "--output", required=True,
  help="path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())
 
# load the training and testing data, converting the images from
# integers to floats
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX,testY)) = cifar10.load()
trainX = trainX.astype("float")
testX = testX.astype("float")

# apply mean subtraction on all the datas
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert label from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
  height_shift_range=0.1, horizontal_flip=True,
  fill_mode="nearest")
  
# construct the set of callbacks
figPath = os.path.sep.join([args["output"],
  "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"],
  "{}.json".format(os.getpid())])
callbacks = [
  TrainingMonitor(figPath, jsonPath=jsonPath),
  LearningRateScheduler(poly_decay)
]

# init the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = ResNet.build(32, 32, 3, 10, (9, 9, 9),
  (64, 64, 128, 256), reg=5e-4)

# train the network
print("[INFO] training network...")
model.fit_generator(
  aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
  validation_data=(testX, testY),
  steps_per_epoch=len(trainX) // BATCH_SIZE, epochs=NUM_EPOCHS,
  callbacks=callbacks, verbose=1
)
  
# save the network to disk
print("[INFO] serializing network...")
model.save(args["model"])