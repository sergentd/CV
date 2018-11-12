# import necessary packages
from pyimagesearch.object_detection.helpers import pyramid
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-s", "--scale", default=1.5,
    help="scale factor size")
args = vars(ap.parse_args())

# load the input image
image = cv2.imread(args["image"])

# loop over the layers in image pyramid and display them
for (i, layer) in enumerate(pyramid(image, int(args["scale"]))):
    cv2.imshow("Layer {}".format(i + 1), layer)
    cv2.waitKey(0)
