import cv2
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

crop1 = image[85:250, 85:220]
crop2 = image[173:235, 13:81]
crop3 = image[90:450, 0:290]
crop4 = image[124:212, 225:380]

cv2.imshow("crop1", crop1)
cv2.imshow("crop2", crop2)
cv2.imshow("crop3", crop3)
cv2.imshow("crop4", crop4)

cv2.waitKey(0)