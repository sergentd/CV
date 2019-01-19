# import necessary packages
from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import CropPreprocessor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.utils.ranked import rank5_accuracy
from keras.models import load_model
import numpy as np
import progressbar
import json

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

# load the pretrained network
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# initialize the testing dataset generator and then make a
# prediction on the testing data
print("[INFO] predicting on test data (no crops)...")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
    preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
predictions = model.predict_generator(testGen.generator(),
    steps=testGen.numImages // config.BATCH_SIZE, max_queue_size=10)

# compute the rank1 accuracy
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()

# re-initialize the testing set generator
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
    preprocessors=[mp], classes=config.NUM_CLASSES)
predictions = []

# initialize the progressbar
widgets = ["Evaluating: ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=testGen.numImages // config.BATCH_SIZE,
    widgets=widgets).start()

# loop over a single pass of the test data
for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):
    # loop over each of the individual images
    for image in images:
        # apply the crop preprocessor to generate crops
        # and then convert them from images to arrays
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops], dtype="uint8")

        # make predictions on the crops and then average them
        # together to obtain the final prediction
        pred = model.predict(crops)
        predictions.append(pred.mean(axis=0))

    # update the progressbar
    pbar.update(i)

# compute the rank-1 accuracy
pbar.finish()
print("[INFO] predicting on test data (with crops...)")
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()
