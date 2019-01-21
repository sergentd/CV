# USAGE
# python detect_gftt.py

# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
import imutils
import argparse

def gftt(gray, maxCorners=0, qualityLevel=0.01, minDistance=1,
  mask=None, blockSize=3, useHarrisDetector=False, k=0.04):
  # compute GFTT keypoints using the supplied parameters (OpenCV 3 only)
  kps = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel,
    minDistance, mask=mask, blockSize=blockSize,
    useHarrisDetector=useHarrisDetector, k=k)

  # create and return `KeyPoint` objects
  return [cv2.KeyPoint(pt[0][0], pt[0][1], 3) for pt in kps]

# construct and parse the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())  
  
# load the image and convert it to grayscale
image = cv2.imread(args["image"])
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# handle if we're detecting and drawing GFTT keypoints using OpenCV 2.4
if imutils.is_cv2():
  detector = cv2.FeatureDetector_create("GFTT")
  kps = detector.detect(gray)

# handle if we're detecting and drawing GFTT keypoints using OpenCV 3+
else:
  kps = gftt(gray)

# loop over the keypoints and draw them
for kp in kps:
  r = int(0.5 * kp.size)
  (x, y) = np.int0(kp.pt)
  cv2.circle(image, (x, y), r, (0, 255, 255), 2)

print("# of keypoints: {}".format(len(kps)))

# show the image
cv2.imshow("Images", np.hstack([orig, image]))
cv2.waitKey(0)