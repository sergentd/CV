# import necessary packages
from pyimagesearch.datasets import SimpleDatasetRenamer
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to the images input directory")
ap.add_argument("-p", "--prefix", type=str, help="prefix to apply on all images of the set")
ap.add_argument("-s", "--suffix", type=str, help="suffix to apply on all images of the set")
args = vars(ap.parse_args())

# load and init the renamer with prefix and suffix
prefix = args["prefix"] if args["prefix"] is not None else "IMG-"
suffix = args["suffix"] if args["suffix"] is not None else None
sdr = SimpleDatasetRenamer(args["directory"], prefix, suffix. move=False, remove=False)

# proceed to the rename process
sdr.rename()