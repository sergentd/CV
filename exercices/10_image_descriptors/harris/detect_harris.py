# USAGE
# python detect_harris.py

# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
import imutils
import argparse

def harris(gray, blockSize=2, apetureSize=3, k=0.1, T=0.02):
	# convert our input image to a floating point data type and then
	# compute the Harris corner matrix
	gray = np.float32(gray)
	H = cv2.cornerHarris(gray, blockSize, apetureSize, k)

	# for every (x, y)-coordinate where the Harris value is above the
	# threshold, create a keypoint (the Harris detector returns
	# keypoint size a 3-pixel radius)
	kps = np.argwhere(H > T * H.max())
	kps = [cv2.KeyPoint(pt[1], pt[0], 3) for pt in kps]

	# return the Harris keypoints
	return kps

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# handle if we are detecting Harris keypoints in the image with OpenCV 2.4
if imutils.is_cv2():
	detector = cv2.FeatureDetector_create("HARRIS")
	kps = detector.detect(gray)

# otherwise we are detecting Harris keypoints with OpenCV 3+ using the function above
else:
	kps = harris(gray)

print("# of keypoints: {}".format(len(kps)))

# loop over the keypoints and draw them
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 2)

# show the image
cv2.imshow("Images", np.hstack([orig, image]))
cv2.waitKey(0)
