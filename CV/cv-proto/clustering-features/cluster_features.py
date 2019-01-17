# import necessary packages
from __future__ import print_function
from pyimagesearch.ir import Vocabulary
import argparse
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True, help="path to input features database")
ap.add_argument("-c", "--codebook", required=True, help="path to output codebook")
ap.add_argument("-k", "--clusters", type=int, default=64, help="# of clusters to generate")
ap.add_argument("-p", "--percentage", type=float, default=0.25,
    help="percentage of total features to use when clustering")
args = vars(ap.parse_args())

# create the visual words vocabulary
voc = Vocabulary(args["features_db"])
codebook = voc.fit(args["clusters"], args["percentage"])

# dump the cluster to file
print("[INFO] storing cluster centers...")
f = open(args["codebook"], "wb")
f.write(pickle.dumps(codebook))
f.close()