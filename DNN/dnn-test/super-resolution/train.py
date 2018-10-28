# set the matplotlib backend so figure can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import necessary packages
from conf import config
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import SRCNN
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

def super_res_generator(inputDataGen, targetDataGen):
    # start an infinite loop for the training data
    while True:
        # grab the next input images and target outputs,
        # discarding the class labels (which are irrelevant)
        inputData = next(inputDataGen)[0]
        targetData = next(targetDataGen)[0]

        yield (inputData, targetData)


inputs = HDF5DatasetGenerator(config.INPUTS_DB, config.BATCH_SIZE)
targets = HDF5DatasetGenerator(config.OUTPUTS_DB, config.BATCH_SIZE)

# initialize the model and optimizer
print("[INFO] compiling model...")
opt = Adam(lr=1e-3, decay = 1e-3 / config.NUM_EPOCHS)
model = SRCNN.build(width=config.INPUT_DIM, height=config.INPUT_DIM,
    depth=3)
model.compile(loss="mse", optimizer=opt)

# train the model using the generators
H = model.fit_generator(
    super_res_generator(inputs.generator(), targers.generator()),
    steps_per_epoch=input.numImages // config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["loss"],
    label="loss")
plt.title("Loss on super resolution training")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig(config.PLOT_PATH)

# close the HDF5 datasets
inputs.close()
targets.close()
