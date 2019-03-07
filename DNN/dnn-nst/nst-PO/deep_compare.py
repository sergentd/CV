# import necessary packages
from conf import config as conf
from imutils import build_montages
import imutils
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="image name *without* path")
args = vars(ap.parse_args())

# load the two images
orig = cv2.imread(os.path.sep.join([conf.ORIG_DIR, args["image"]]))
deep = cv2.imread(os.path.sep.join([conf.DEEP_DIR, args["image"]]))

# resize the images
orig = imutils.resize(orig, width=conf.COMPARE_WIDTH)
deep = imutils.resize(deep, width=conf.COMPARE_WIDTH)

# build the montage with the two input images
images = [orig, deep]
montage = build_montages(images, (orig.shape[1], orig.shape[0]), (2, 1))[0]

# check to see if we want to apply a legend on the frame
if conf.COMPARE_LEGEND:
    montage[0:conf.S_LEGEND_HEIGHT, 0:conf.LEGEND_WIDTH] = 0
    cv2.putText(montage, "press any key to continue...", (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

# show the images side by side for a clear comparison
cv2.imshow("Orig vs. Deep", montage)
cv2.waitKey(0)
