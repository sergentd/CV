# import necessary packages
import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detection")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels the model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the path to the YOLO weights and model configuration
weightPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join(args["yolo"], "yolo3.cfg")

# load the YOLO object detector trained on COCO dataset
print("[INFO] loading YOLO model...")
net = cv2.dnn.readNetFromDarknet(configPath, weightPath)

# load the input image and grab the spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# determine the output layer names needed for YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the image and perform a forward in the network
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416),
    swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show the timing information on YOLO
print("[INFO] YOLOR took {:.6f} seconds".format(end - start))

# initialize the lists of detected bounding boxes, confidences,
# and class IDs
boxes = []
confidences = []
classIDs = []

# loop over each of the layer output
for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
        # extract the class ID and confidence
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # filter out weak prediction by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative
            # to size of the image
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)-coordinates to derive the top
            # and left corner
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # update the lists
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# apply non-maxima suppression to suppress weak overlapping
# bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidence, args["confidence"],
    args["threshold"])

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes
    for id in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[id][0], boxes[id][1])
        (w, h) = (boxes[id][3], boxes[id][4])

        # draw a bounding box rectangle and label the image
        color = [int(c) for c in COLORS[classIDs[id]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[id]], confidences[id])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
