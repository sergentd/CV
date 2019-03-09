# USAGE
# python bw2color.py --input image/bw_image.jpg
# python bw2color.py --input image/bw_image.jpg --output image/colorized_bwimage.jpg

# import necessary packages
from pyimagesearch.improc import Colorizer
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
    help="path to input black and white image")
ap.add_argument("-o", "--output", type=str
    help="path to output colorized image")
args = vars(ap.parse_args())

# colorize the image
colorizer = Colorizer(conf.COLORIZATION_PROTO_PATH,
    conf.COLORIZATION_MODEL_PATH, conf.POINTS_IN_HULL_PATH)
(orig, gray, colorized) = colorizer.predict(args["image"])

if args.get("output", False):
    cv2.imwrite(args["output"], colorized)

# show the images
cv2.imshow("Colorized", colorized)
cv2.imshow("Grayscale", gray)
cv2.imshow("Original", orig)
cv2.waitKey(0)
