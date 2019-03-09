# import necessary packages
import cv2
from scripts.utils import Conf
from scripts.utils.preprocessing import ImageToArrayPreprocessor
from scripts.utils.preprocessing import AspectAwarePreprocessor
from scripts.utils.preprocessing import MeanPreprocessor
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
    help="path to configuration file")
ap.add_argument("-p", "--prefix", required=True,
    help="name of model prefix")
ap.add_argument("-e", "--epoch", type=int, required=True,
    help="epoch # to load")
ap.add_argument("-s", "--sample-size", type=int, default=10,
    help="sample size we want to visualize")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# construct the paths to label encoder and mx record
labelEncoderPath = os.path.sep.join([conf["output_dir"],
    conf["label_encoder_path"]])
testMXRec = os.path.sep.join(conf["mx_output"], conf["rec_dir"],
    conf["test_mx_rec"])

# load the label encoder and the mx record
le = pickle.loads(open(labelEncoderPath, "rb").read())
rows = open(testMXRec).read().strip().split("\n")

# sample the the rows
rows = np.random.choice(rows, size=args["sample_size"])

print("[INFO] loading pre-trained model...")
checkpointsPath = os.path.sep.join([conf["checkpoints_dir"], args["prefix"]])
model = mx.model.FeedForward.load(checkpointsPath, args["epoch"])

# compile the model
model = mx.model.FeedForward(
    ctx=[mx.gpu(0)],
    symbol=model.symbol,
    arg_params=model.arg_params,
    aux_params=model.aux_params)

# initialize the image preprocessors
sp = AspectAwarePreprocessor(width=224, height=224)
mp = MeanPreprocessor(conf["r_mean"], conf["g_mean"], conf["b_mean"])
iap = ImageToArrayPreprocessor(dataFormat="channels_first")

# loop over the testing images
for row in rows:
    # grab the target class label and the image path from the row
    (target, imagePath) = row.split("\t")[1:]
    target = int(target)

    # load the image from disk and preprocess it by resizing the image and
    # applying the pre-processors
    image = cv2.imread(imagePath)
    orig = image.copy()
    orig = imutils.resize(orig, width=min(500, orig.shape[1]))
    image = iap.preprocess(mp.preprocess(sp.preprocess(image)))
    image = np.expand_dims(image, axis=0)

    # classify the image and grab the indexes of the top-5 predictions
    preds = model.predict(image)[0]
    idxs = np.argsort(preds)[::-1][:5]

    # show the tue class label
    print("[INFO] actual={}".format(le.inverse_transfrom(target)))

    # format and display the top predicted class label
    label = le.inverse_transfrom(idxs[0])
    label = label.replace(":", " ")
    label = "{}: {:.2f}%".format(label, preds[idxs[0]] * 100)
    cv2.putText(orig, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 255, 0), 2)

    # loop over the predictions and display them
    for (i, prob) in zip(idxs, preds):
        print("\t[INFO] predicted={}, probability={:.2f}%".format(
            le.inverse_transfrom(i), preds[i] * 100))

    # show the image
    cv2.imshow("Image", orig)
    cv2.waitKey(0)
