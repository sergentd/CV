# USAGE
# python detect_dense_adv.py

# import the necessary packages
from __future__ import print_function
import numpy as np
import argparse
import cv2
import imutils

def dense(image, step, radius):
	# initialize our list of keypoints
	kps = []

	# loop over the height and with of the image, taking a `step`
	# in each direction
	for x in range(0, image.shape[1], step):
		for y in range(0, image.shape[0], step):
			# create a keypoint and add it to the keypoints list
			kps.append(cv2.KeyPoint(x, y, radius * 2))

	# return the dense keypoints
	return kps

#  construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-s", "--step", type=int, default=28, help="step (in pixels) of the dense detector")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# initialize our keypoint list and radii tuple
kps = []
radii = (4, 8, 12)

# detect Dense keypoints in the image for OpenCV 2.4
if imutils.is_cv2():
	detector = cv2.FeatureDetector_create("Dense")
	detector.setInt("initXyStep", args["step"])
	rawKps = detector.detect(gray)

# otherwise detect Dense keypoints in the image for OpenCV 3+
else:
	rawKps = dense(gray, args["step"], 1)

# loop over the raw keypoints
for rawKp in rawKps:
	# loop over the various radii we are going to use
	for r in radii:
		# construct a keypoint manually and then update the keypoitns list
		kp = cv2.KeyPoint(x=rawKp.pt[0], y=rawKp.pt[1], _size=r * 2)
		kps.append(kp)

# show some information regarding the number of keypoints detected
print("# dense keypoints: {}".format(len(rawKps)))
print("# dense + multi radii keypoints: {}".format(len(kps)))

# loop over the keypoints and draw them
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 1)

# show the image
cv2.imshow("Images", np.hstack([orig, image]))
cv2.waitKey(0)