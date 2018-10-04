# import necessary packages
from keras.application import VGG16
from pyimagesearch.utils import ModelInspector
import argparse

# construct the argument parser and parse the arguments
ap = argpase.ArgumentParser()
ap.add_argument("-i","--include-top", type=int, default=1,
  help="whether or not to include top of CNN")
args = vars(ap.parse_args())

# load the VGG16 network
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=args["include_top"] > 0)

# load the model inspector
mi = ModelInspector(model)

# inspect all layers of the model
mi.inspect_layers()