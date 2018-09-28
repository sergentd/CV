# import necessary packages
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
    help="Path to weight directory")
args = vars(ap.parse_args())

# load the training set and scale it to the
# range [0, 1]
print("[INFO] loading CIFAR-10 dataset...")
((trainX, trainY),(testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integer to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transfrom(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# construct the callback to save the model
fname = os.path.sep.join([args["weights"],
  "weights-{epochs:03d}-{val_loss:.4f}.hdf5"])
improve = ModelCheckpoint(fname, monitor="val_loss", mode="min",
    save_best_only=True, verbose=-1)
best = ModelCheckpoint(args["weights"], monitor="val_loss", mode="min",
    save_best_only=True, verbose=-1)
callbacks=[improve, best]

# train the model
model.fit(trainX, trainY, validation_data=(testX, testY),
    batch_size=64, epochs=40, callbacks=callbacks, verbose=2)

# evaluate the model
print("[INFO] evaluating the model...TODO !")
