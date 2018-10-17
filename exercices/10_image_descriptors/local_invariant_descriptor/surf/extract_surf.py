# USAGE
# python extract_surf.py --image jp_01.png

# import the necessary packages
from __future__ import print_function
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the input image, convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# handle if we are using OpenCV 2.4
if imutils.is_cv2():
	# initialize the keypoint detector and local invariant descriptor
	detector = cv2.FeatureDetector_create("SURF")
	extractor = cv2.DescriptorExtractor_create("SURF")

	# detect keypoints, and then extract local invariant descriptors
	kps = detector.detect(gray)
	(kps, descs) = extractor.compute(gray, kps)

# otherwise, we are using OpenCV 3+
else:
	# initialize the keypoint detector and local invariant descriptor
	detector = cv2.xfeatures2d.SURF_create()

	# detect keypoints and extract local invariant descriptors
	(kps, descs) = detector.detectAndCompute(gray, None)

# show the shape of the keypoints and local invariant descriptors array
print("[INFO] # of keypoints detected: {}".format(len(kps)))
print("[INFO] feature vector shape: {}".format(descs.shape))