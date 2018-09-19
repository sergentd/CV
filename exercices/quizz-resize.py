import imutils
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True)
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

resized = imutils.resize(image, width=100, inter=cv2.INTER_NEAREST)

(b,g,r) = resized[:2]

print("Blue = {}, Green = {}, Red = {}".format(b,g,r))