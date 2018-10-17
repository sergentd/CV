# USAGE
# python detect_fast.py

# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
import imutils

# load the image and convert it to grayscale
image = cv2.imread("next.png")
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# handle if we are detecting FAST keypoints in the image for OpenCV 2.4
if imutils.is_cv2():
	detector = cv2.FeatureDetector_create("FAST")
	kps = detector.detect(gray)

# otherwise, we are detecting FAST keypoints for OpenCV 3+
else:
	detector = cv2.FastFeatureDetector_create()
	kps = detector.detect(gray, None)

print("# of keypoints: {}".format(len(kps)))

# loop over the keypoints and draw them
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 2)

# show the image
cv2.imshow("Images", np.hstack([orig, image]))
cv2.waitKey(0)