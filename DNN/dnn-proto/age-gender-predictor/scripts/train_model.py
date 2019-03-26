# import the necessary packages
from utils import AgeGenderHelper
from utils.mxnn import MxAgeGenderNet
from utils.mxcallbacks import one_off_callback
import mxnet as mx
import argparse
import logging
import pickle
import json
import os

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset-type", type=str, required=True,
        help="configuration file to load (can be either 'age' or 'gender')")
    ap.add_argument("-c", "--checkpoints", required=True,
        help="path to output checkpoints directory")
    ap.add_argument("-p", "--prefix", required=True,
        help="name of model prefix")
    ap.add_argument("-s", "--start-epoch", type=int, default=0,
        help="epoch number to restart training at")
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

    logging.basicConfig(level=logging.DEBUG,
        filename="log/{}_training_{}.log".format(args["dataset_type"],
        args["start_epoch"]), filemode="w")

    # determine the batch and load the mean pixel values
    batchSize = config.BATCH_SIZE * config.NUM_DEVICES
    means = json.loads(open(config.DATASET_MEAN_PATH).read())

    # construct the training image iterator
    trainIter = mx.io.ImageRecordIter(
        path_imgrec=config.TRAIN_MX_REC,
        data_shape=(3, 227, 227),
        batch_size=batchSize,
        rand_crop=True,
        rand_mirror=True,
        rotate=7,
        mean_r=means["R"],
        mean_g=means["G"],
        mean_b=means["B"],
        preprocess_thread=config.NUM_DEVICES * 2)

    # construct the validation image iterator
    valIter = mx.io.ImageRecordIter(
        path_imgrec=config.VAL_MX_REC,
        data_shape=(3, 227, 227),
        batch_size=batchSize,
        mean_r=means["R"],
        mean_g=means["G"],
        mean_b=means["B"])

    # initialize the optimizer
    opt = mx.optimizer.Adam(learning_rate=1e-5, wd=0.0005,
        rescale_grad=1.0 / batchSize)

    # construct the checkpoints path, initialize the model argument
    # and auxiliary parameters
    checkpointsPath = os.path.sep.join([args["checkpoints"], args["prefix"]])
    if not os.path.exists(checkpointsPath):
        os.makedirs(checkpointsPath)
    argParams = None
    auxParams = None

    # if there is no specific model starting epoch supplied, then
    # initialize the model
    if args["start_epoch"] <= 0:
        # build the MxAgeGenderNet architecture
        print("[INFO] building network...")
        model = MxAgeGenderNet.build(config.NUM_CLASSES)

    # otherwise, a specific checkpoint was supplied
    else:
        # load the checkpoint from disk
        print("[INFO] loading epoch {}...".format(args["start_epoch"]))
        (model, argParams, auxParams) = mx.model.load_checkpoint(
            checkpointsPath, args["start_epoch"])

    # compile the model
    model = mx.model.FeedForward(
        ctx=[mx.gpu(0)],
        symbol=model,
        initializer=mx.initializer.Xavier(),
        arg_params=argParams,
        aux_params=auxParams,
        optimizer=opt,
        num_epoch=100,
        begin_epoch=args["start_epoch"])

    # initialize the callbacks and evaluation metrics
    batchEndCBs = [mx.callback.Speedometer(batchSize, 10)]
    epochEndCBs = [mx.callback.do_checkpoint(checkpointsPath)]
    metrics = [mx.metric.Accuracy(), mx.metric.CrossEntropy()]

    # check to see if the one-off accuracy should be used
    if config.DATASET_TYPE == "age":
        # load the label encoder, then build the one-off mappings
        # for computing accuracy
        le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
        agh = AgeGenderHelper(config)
        oneOff = agh.buildOneOffMappings(le)
        epochEndCBs.append(one_off_callback(trainIter, valIter, oneOff, mx.gpu(0)))

    # train the network
    print("[INFO] training network...")
    model.fit(
        X=trainIter,
        eval_data=valIter,
        eval_metric=metrics,
        batch_end_callback=batchEndCBs,
        epoch_end_callback=epochEndCBs)

if __name__ == "__main__":
    main()
