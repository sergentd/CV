# import necessary packages
from pyimagesearch.datasets import SimpleDatasetRenamer
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to the images input directory")
ap.add_argument("-p", "--prefix", type=str, help="prefix to apply on all images of the set")
ap.add_argument("-s", "--suffix", type=str, help="suffix to apply on all images of the set")
ap.add_argument("-m", "--move", default=False, help="copy images in current directory")
ap.add_argument("-r", "--remove", default=False, help="remove original image")
args = vars(ap.parse_args())

# load and init the renamer with prefix and suffix
sdr = SimpleDatasetRenamer(
  path=args["directory"],
  prefix=args["prefix"],
  suffix=args["suffix"],
  move=args["move"],
  remove=args["remove"]
)

# proceed to the rename process
sdr.rename()