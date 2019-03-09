# set the matplotlib backend so figure can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import necessary packages
from helpers import HDF5DatasetGenerator
from helpers import SRCNN
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import argparse
import numpy as np

def super_res_generator(inputDataGen, targetDataGen):
    # start an infinite loop for the training data
    while True:
        # grab the next input images and target outputs,
        # discarding the class labels (which are irrelevant)
        inputData = next(inputDataGen)[0]
        targetData = next(targetDataGen)[0]

        yield (inputData, targetData)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input-db", type=str, required=True,
    help="path to input database")
ap.add_argument("-o", "--output-db", type=str, required=True,
    help="path to output database")
ap.add_argument("-d", "--input-dim", type=float, default=33,
    help="scale factor for images")
ap.add_argument("-b", "--batch-size", type=int, default=128,
    help="batch size for training network")
ap.add_argument("-e", "--epochs", type=int, default=10,
    help="number of epochs to run")
ap.add_argument("-m", "--model-path", type=str, default="models/srcnn.model",
    help="path to output model")
ap.add_argument("-p", "--plot-path", type=str, default="output/plot.png",
    help="path to output plot")
args = vars(ap.parse_args())

inputs = HDF5DatasetGenerator(args["input_db"], args["batch_size"])
targets = HDF5DatasetGenerator(args["output_db"], args["batch_size"])

# initialize the model and optimizer
print("[INFO] compiling model...")
opt = Adam(lr=1e-3, decay = 1e-3 / args["epochs"])
model = SRCNN.build(width=args["input_dim"], height=args["input_dim"],
    depth=3)
model.compile(loss="mse", optimizer=opt)

# train the model using the generators
H = model.fit_generator(
    super_res_generator(inputs.generator(), targets.generator()),
    steps_per_epoch=inputs.numImages // args["batch_size"],
    epochs=args["epochs"], verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(args["model_path"], overwrite=True)

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), H.history["loss"],
    label="loss")
plt.title("Loss on super resolution training")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig(args["plot_path"])

# close the HDF5 datasets
inputs.close()
targets.close()
