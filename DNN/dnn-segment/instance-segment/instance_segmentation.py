# USAGE
# python instance_segmentation.py --mask-rcnn mask-rcnn-coco --kernel 41

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mask-rcnn", required=True,
	help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="minimum threshold for pixel-wise mask segmentation")
ap.add_argument("-k", "--kernel", type=int, default=41,
	help="size of gaussian blur kernel")
args = vars(ap.parse_args())

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# construct the kernel for the Gaussian blur and initialize whether
# or not we are in "privacy mode"
K = (args["kernel"], args["kernel"])
privacy = False

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	(H, W) = frame.shape[:2]

	# construct a blob from the input image and then perform a
	# forward pass of the Mask R-CNN, giving us (1) the bounding
	# box coordinates of the objects in the image along with (2)
	# the pixel-wise segmentation for each specific object
	blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
	net.setInput(blob)
	(boxes, masks) = net.forward(["detection_out_final",
		"detection_masks"])

	# sort the indexes of the bounding boxes in by their corresponding
	# prediction probability (in descending order)
	idxs = np.argsort(boxes[0, 0, :, 2])[::-1]

	# initialize the mask, ROI, and coordinates of the person for the
	# current frame
	mask = None
	roi = None
	coords = None

	# loop over the indexes
	for i in idxs:
		# extract the class ID of the detection along with the
		# confidence (i.e., probability) associated with the
		# prediction
		classID = int(boxes[0, 0, i, 1])
		confidence = boxes[0, 0, i, 2]

		# if the detection is not the 'person' class, ignore it
		if LABELS[classID] != "person":
			continue

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the image and then compute the width and the
			# height of the bounding box
			box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			coords = (startX, startY, endX, endY)
			boxW = endX - startX
			boxH = endY - startY

			# extract the pixel-wise segmentation for the object,
			# resize the mask such that it's the same dimensions of
			# the bounding box, and then finally threshold to create
			# a *binary* mask
			mask = masks[i, classID]
			mask = cv2.resize(mask, (boxW, boxH),
				interpolation=cv2.INTER_NEAREST)
			mask = (mask > args["threshold"])

			# extract the ROI and break from the loop (since we make
			# the assumption there is only *one* person in the frame
			# who is also the person with the highest prediction
			# confidence)
			roi = frame[startY:endY, startX:endX][mask]
			break

	# initialize our output frame
	output = frame.copy()

	# if the mask is not None *and* we are in privacy mode, then we
	# know we can apply the mask and ROI to the output image
	if mask is not None and privacy:
		# blur the output frame
		output = cv2.GaussianBlur(output, K, 0)

		# add the ROI to the output frame for only the masked region
		(startX, startY, endX, endY) = coords
		output[startY:endY, startX:endX][mask] = roi

	# show the output frame
	cv2.imshow("Video Call", output)
	key = cv2.waitKey(1) & 0xFF

	# if the `p` key was pressed, toggle privacy mode
	if key == ord("p"):
		privacy = not privacy

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()