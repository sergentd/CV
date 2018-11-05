# import necessary packages
from pyimagesearch.datasets import SimpleDatasetRenamer
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True, help="path to the images input directory")
ap.add_argument("-p", "--prefix", type=str, help="prefix to apply on all images of the set")
ap.add_argument("-s", "--suffix", type=str, help="suffix to apply on all images of the set")
ap.add_argument("-k", "--keep-idx", default=False, help="keep the original idx part of filename")
ap.add_argument("-m", "--move", default=None, help="copy images in current directory")
ap.add_argument("-r", "--remove", default=False, help="remove original image")
ap.add_argument("-q", "--sequential", default=True, help="sequential or random")
ap.add_argument("-l", "--length", default=6, help="length of the filename idx")
ap.add_argument("-e", "--ext", default=None, help="file extension (default: for same as original)")
ap.add_argument("-i", "--index", default=0, help="starting index for renaming sequentially")
args = vars(ap.parse_args())

# load and init the renamer with all parameters
sdr = SimpleDatasetRenamer(
  path=args["directory"],
  prefix=args["prefix"],
  suffix=args["suffix"],
  move=args["move"],
  remove=args["remove"],
  sequential=args["sequential"],
  length=args["length"],
  ext=args["ext"],
  index=args["index"],
)

# proceed to the rename process
sdr.rename()