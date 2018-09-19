import cv2
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-a", "--angle", default=0, help="number of degrees we need to rotate")
ap.add_argument("-x", "--xpos", default=0, help="x position of pixel")
ap.add_argument("-y", "--ypos", default=0, help="y position of pixel")
args = vars(ap.parse_args())

image = imread(args["image"])
angle = args["angle"]

rotated = imutils.rotate(image, angle)

(b,g,r) = rotated[args["ypos"], args["xpos"]]

print("Values : Blue = {}, Green={}, red){}".format(b,g,r))

cv2.imshow(rotated)
cv2.waitKey(0)
