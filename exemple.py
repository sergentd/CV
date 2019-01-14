# import necessary packages
import cv2
import argparse

# create the argument parser and parse the arguments.
# allow the user to indicate paths in command line 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="the input image on which we will draw")
ap.add_argument("-o", "--output", required=True, help="image name and location for saving")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["input"])

# write some text at the center of the image
(h, w) = image.shape[:2]
cv2.putText(image, "Hello world", (w//2 - 30, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# cv2.putText(image, label, (coordX, coordY), font, fontScale, (B, G, R), thickness)
# Warning : color space is BGR

# display the image
cv2.imshow("Exemple", image)
cv2.waitKey(0) # <---- else the script will end imediately and close the window

# save the image
cv2.imwrite(args["output"], image)
