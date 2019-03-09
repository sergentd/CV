from utils.ranked import rank5_accuracy
from utils import Conf
import mxnet as mx
import argparse
import pickle
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
    help="path to configuration file")
ap.add_argument("-p", "--prefix", required=True,
    help="name of model prefix")
ap.add_argument("-e", "--epoch", required=True,
    help="epoch # to load")
args = vars(ap.parse_args())

# laod the configuration file
conf = Conf(args["conf"])

# construct the paths to mx records
labelEncoderPath = os.path.sep.join([conf["output_dir"],
    conf["label_encoder_path"]])
testMXRec = os.path.sep.join([conf["mx_output"], conf["rec_dir"],
    conf["test_mx_rec"]])

# load the label encoder
le = pickle.loads(open(labelEncoderPath, "rb").read())

# construct the validation image iterator
testIter = mx.io.ImageRecordIter(
    path_imgrec=testMXRec,
    data_shape=(3, 224, 224),
    batch_size=conf["batch_size"],
    mean_r=conf["r_mean"],
    mean_g=conf["g_mean"],
    mean_b=conf["b_mean"])

# load the pre-trained model from disk
print("[INFO] loading pre-trained model...")
checkpointsPath = os.path.sep.join([conf["checkpoints_dir"], args["prefix"]])
(symbol, argParams, auxParams) = mx.model.load_checkpoint(
    checkpointsPath, args["epoch"])

# construct the model
model = mx.mod.Module(symbol=symbol, contect=[mx.gpu(0)])
model.bind(data_shapes=testIter.provide_data,
    label_shapes=testIter.provide_label)
model.set_params(argParams, auxParams)

# initialize the list of predictions and targets
print("[INFO] evaluating model...")
predictions = []
targets = []

# loop over the predictions in batches
for (preds, _, batch) in model.iter_predict(testIter):
    # convert the batch of predictions and labels to Numpy array
    preds = preds[0].asnumpy()
    labels = batch.label[0].asnumpy().astype("int")

    # update the predictions and targets lists respectively
    predictions.extend(preds)
    targets.extend(labels)

# apply array slicing to the targets since mxnet will return the next
# full batch size rather than the actual number of labels
targets = targets[:len(predictions)]

# compute the rank-1 and rank-5 accuracies
(rank1, rank5) = rank5_accuracy(predictions, targets)
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))
