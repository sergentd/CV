import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

flipped = cv2.flip(image, 1)

(b,g,r) = flipped[254,337]

print("Values : Blue = {}, Green={}, red={}".format(b,g,r))