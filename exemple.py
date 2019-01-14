# import necessary packages
import cv2
import argparse

# create the argument parser and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="the input image on which we will draw")
ap.add_argument("-o", "--output", required=True, help="image name and location for saving")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

# write some text at the center of the image
(h, w) = image.shape[:2]
cv2.putText(image, "Hello world", (h/2 - 10, w/2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)

# display the image
cv2.imshow("Exemple", image)
cv2.waitKey(0)

# save the image
cv2.imwrite(args["output"])