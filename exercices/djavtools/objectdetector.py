# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import mahotas
import cv2
import imutils
import argparse

def describe_shapes(image):
	# initialize the list of shape features
	shapeFeatures = []

	# convert the image to grayscale, blur it, and threshold it
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (13, 13), 0)
	thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)

	# perform a series of dilations and erosions to close holes
	# in the shapes
	thresh = cv2.dilate(thresh, None, iterations=8)
	thresh = cv2.erode(thresh, None, iterations=4)
	cv2.imshow("THRESH", thresh)

	# detect contours in the edge map
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	cv2.waitKey(0)

	# loop over the contours
	for c in cnts:
		# create an empty mask for the contour and draw it
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)

		# extract the bounding box ROI from the mask
		(x, y, w, h) = cv2.boundingRect(c)
		roi = mask[y:y + h, x:x + w]

		# compute Zernike Moments for the ROI and update the list
		# of shape features
		features = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(c)[1], degree=8)
		shapeFeatures.append(features)

	# return a tuple of the contours and shapes
	return (cnts, shapeFeatures)

def detect_image(image, host):
	# describe the image
	(_, imageFeatures) = describe_shapes(image)
	print("[INFO] image loaded and features extracted...")
	
	# load the shapes image, then describe each of the images in the image
	(cnts, shapeFeatures) = describe_shapes(host)
	print("[INFO] host loaded and features extracted...")

	# compute the Euclidean distances between the video game features
	# and all other shapes in the second image, then find index of the
	# smallest distance
	D = dist.cdist(shapeFeatures, imageFeatures)
	i = np.argmin(np.unique(D))

	print("I value : {}".format(i))
	print("cnts counter : {}".format(len(cnts)))

	# loop over the contours in the shapes image
	for (j, c) in enumerate(cnts):
		print("[INFO] starting to draw...")
		# if the index of the current contour does not equal the index
		# contour of the contour with the smallest distance, then draw
		# it on the output image
		if i != j:
			box = cv2.minAreaRect(c)
			box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
			cv2.drawContours(host, [box], -1, (0, 0, 255), 2)

	# draw the bounding box around the detected shape
	box = cv2.minAreaRect(cnts[i])
	box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
	cv2.drawContours(host, [box], -1, (0, 255, 0), 2)
	(x, y, w, h) = cv2.boundingRect(cnts[i])
	cv2.putText(host, "FOUND!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
		(0, 255, 0), 3)

	# show the output images
	cv2.imshow("Input Image", image)
	cv2.imshow("Detected Shapes", host)
	cv2.waitKey(0)
	print("[INFO] ending program")


if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="Path to input image")
	ap.add_argument("-o", "--host", required=True, help="Path to host image")
	args = vars(ap.parse_args())
	
	# load the images for the searched image and
	# the host image
	image = cv2.imread(args["image"])
	host = cv2.imread(args["host"])
	
	# check the images size and resize it if they are too big
	hI = image.shape[0]
	hH = host.shape[0]

	if hI > 600:
		image = imutils.resize(image, height=600)
	if hH > 600:
		host = imutils.resize(host, height=600)

	# actually try to detect the image in the host
	detect_image(image, host)
