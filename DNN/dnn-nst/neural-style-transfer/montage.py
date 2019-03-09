# import necessary packages
from conf import config as conf
from imutils import build_montages
from imutils import paths
import numpy as np
import imutils
import argparse
import cv2
import os

# grab the reference to the list of images we want to include in the montage
origPaths = list(paths.list_images(conf.ORIG_DIR))
deepPaths = list(paths.list_images(conf.DEEP_DIR))

# grab randomly a list of files to include in the montage and initialize the
# list of images
(lines, columns) = conf.MONTAGE_TILES
idxs = np.arange(0, len(origPaths)-1)
idxs = np.random.choice(idxs, size=min(int(lines*columns/2),
    len(origPaths)), replace=False)
images = []

# loop over the selected indexes and load the images from disk, then add it
# to the list of images to build
for i in idxs:
    orig = origPaths[i]
    deep = os.path.sep.join([conf.DEEP_DIR, os.path.basename(orig)])
    images.append(cv2.imread(orig))
    images.append(cv2.imread(deep))

# create the montage
montage = build_montages(images, conf.MONTAGE_SIZE, conf.MONTAGE_TILES)[0]

# check to see if we need to display the legend on the image
if conf.MONTAGE_LEGEND:
    cv2.putText(montage, "press any key to continue...", (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

# show the images side by side for a clear comparison
cv2.imshow("Orig vs. Deep", montage)
cv2.waitKey(0)
