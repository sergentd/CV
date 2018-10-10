# import necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argpase.ArgumentParser()
ap.add_argument("-d","--dataset", required=True, help="path to the input dataset")
ap.add_argument("-m","--model", required=True, help="path to the output model")
args = vars(ap.parse_args())

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
  height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
  horizontal_flip=True, fill_mode="nearest")
  
# grab the list of images that we'll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# initialize the image preprocessors
aap = AspectAwarePreprocessor(224,224)
iap = ImageToArrayPreprocessor()

# load the dataset from disk, then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels,
  test_size=0.25, random_state=42)
  
# convert the labels into vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY  = LabelBinarizer().transform(testY)

# load the model and ensure the head FC layers are left off
baseModel = VGG16(weights="imagenet", include_top=False,
  input_tensor=Input(shape=(224,224,3)))
  
# initialize the new head of the network, a set of FC layers
# followed by a softmax classifier
headModel = FCHeadNet.build(baseModel, len(classNames), 256)

# place the FC model on top of the base model
# -- this will become the actual model to train
model = Model(inputs=baseModel.inputs, outputs=headModel)

# freeze the layers so they are not trained
for layer in baseModel.layers:
  layer.trainable = False
  
# compile the model (need to be done AFTER the layer freeze)
print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
  metrics=["accuracy"])
  
# train the head of the network for a few epochs (all other
# layers are frozen) -- this will allow the new FC layers to
# start to become initialized with actual learned values
# versus pure random
print("[INFO] training head...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
  validation_data=(testX, testY), epochs=25,
  steps_per_epochs=len(trainX) // 32, verbose=1)

# evaluate the network after the initialization
print("[INFO] evaluating after initialization...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
  predictions.argmax(axis=1), target_names=classNames))
  
# now the head is init, let's unfreeze some of the layers
for layer in baseModel.layers[15:]:
  layer.trainable = True

# for the changes to the model to take effect, we need to recompile
# the model, this time using and SGD optimizer
print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
  metrics=["accuracy"])

# train the model again, this time fine-tuning both the
# final set of CONV layers and the FC layers
print("[INFO] fine-tuning model...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
  validation_data=(testX, testY), epochs=100,
  steps_per_epochs=len(trainX) // 32, verbose=1)

# evaluate the network on the fine-tuned model
print("[INFO] evaluating fine-tuned model...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
  predictions.argmax(axis=1), target_names=classNames))
  
# save the model to disk
print("[INFO] saving model...")
model.save(args["model"])