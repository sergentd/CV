# import necessary packages
from utils import AgeGenderHelper
from utils.mxcallbacks.mxmetrics import _compute_one_off
import mxnet as mx
import argparse
import pickle
import json
import os

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset-type", required=True,
        help="configuration file to load (can be either 'age' or 'gender')")
    ap.add_argument("-c", "--checkpoints", required=True,
        help="path to checkpoints model")
    ap.add_argument("-p", "--prefix", required=True,
        help="name of model prefix")
    ap.add_argument("-e", "--epoch", type=int, required=True,
        help="epoch number to load")
    args = vars(ap.parse_args())

    # check to see if we need to load the 'age' parameters
    if args["dataset_type"] == "age":
        from conf import age_conf as config

    # otherwise, check to see if it is the 'gender' parameters
    elif args["dataset_type"] == "gender":
        from conf import gender_conf as config

    # otherwise, there is no configuration file corresponding to the argument
    else:
        print("no configuration file corresponding to the dataset-type provided")
        return

    # load the mean pixel intensities
    means = json.loads(open(config.DATASET_MEAN_PATH).read())

    # construct the testing image iterator
    testIter = mx.io.ImageRecordIter(
        path_imgrec=config.TEST_MX_REC,
        data_shape=(3, 227, 227),
        batch_size=config.BATCH_SIZE,
        mean_r=means["R"],
        mean_g=means["G"],
        mean_b=means["B"])

    # load the checkpoint from disk
    print("[INFO] loading model...")
    checkpointsPath = os.path.sep.join([args["checkpoints"], args["prefix"]])
    model = mx.model.FeedForward.load(checkpointsPath, args["epoch"])

    # compile the model
    model = mx.model.FeedForward(
        ctx=[mx.gpu(0)],
        symbol=model.symbol,
        arg_params=model.arg_params,
        aux_params=model.aux_params)

    # make predictions on the testing data
    print("[INFO] predicting on '{}' test data...".format(
        config.DATASET_TYPE))
    metrics = [mx.metric.Accuracy()]
    acc = model.score(testIter, eval_metric=metrics)

    # display the rank-1 accuracy
    print("[INFO] rank-1: {:.2f}%".format(acc[0] * 100))

    # check to see if the one-off accuracy should be used
    if config.DATASET_TYPE == "age":
        # re-compile the model so that we can compute our custom one-off
        # evaluation metric
        arg = model.arg_params
        aux = model.aux_params
        model = mx.mod.Module(symbol=model.symbol, context=[mx.gpu(0)])
        model.bind(data_shapes=testIter.provide_data,
            label_shapes=testIter.provide_label)
        model.set_params(arg, aux)

        # load the label encoder, then build the one-off mappings for
        # computing accuracy
        le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
        agh = AgeGenderHelper(config)
        oneOff = agh.buildOneOffMappings(le)

        # compute and display the one-off accuracy
        acc = _compute_one_off(model, testIter, oneOff)
        print("[INFO] one-off: {:.2f}%".format(acc * 100))

if __name__ == "__main__":
    main()
