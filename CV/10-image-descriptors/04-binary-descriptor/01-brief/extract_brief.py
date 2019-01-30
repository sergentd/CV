# USAGE
# python extract_brief.py --image jp_01.png

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

# initialize the keypoint detector and local invariant descriptor
# for OpenCV 2.4
if imutils.is_cv2():
	detector = cv2.FeatureDetector_create("FAST")
	extractor = cv2.DescriptorExtractor_create("BRIEF")

# otherwise, initialize the keypoint detector and local invariant descriptor
# for OpenCV 3+
else:
	detector = cv2.FastFeatureDetector_create()
	extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# detect keypoints, and then extract local invariant descriptors
kps = detector.detect(gray)
(kps, descs) = extractor.compute(gray, kps)

# show the shape of the keypoints and local invariant descriptors array
print("[INFO] # of keypoints detected: {}".format(len(kps)))
print("[INFO] feature vector shape: {}".format(descs.shape))