# import the necessary packages
import numpy as np
import argparse
import cv2

# construct and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-p", "--prototxt", required=True, help="Path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="Path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="Minimum probability to filter weak detections")

# init the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# init the colors for boxes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load the serialized model from disk
print("[INFO] : loading model...")
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

# load the input image and construt an input blob for the image by resizing to a fixed 300x300 pixels
# and then normalizing it (note : normalization is done by the authors of the MobileNet SSD implementation)
image = cv2.imread(args['image'])
(h,w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 0.007843, (300,300), 127.5)

# pass the blob throught the network and obtain the detections and predictions
print("[INFO] : computing object detections")
net.setInput(blob)
detections = net.forward)()

# loop over the detections
for i in np.arange(0, detections.shape[2]):
  # extract the confidencec associated with the predictions
  confidence = detections[0, 0, i, 2]
  
  # filter out weak detections by ensuring the confidence is greater than the minimum
  if confidence > args['confidence']:
    # extract the index of class label from detections and then compute
	# the (x, y)-coorfinates of the bounding box for the object
	idx = int(detections[0, 0, i, 1])
	box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
	(startX, startY, endX, endY) = box.astype("int")
	
	# display the prediciton
	label = "{} : {:.2f}%".format(CLASSES[idx], confidence * 100)
	print("[INFO] : {}".format(label))
	cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
	y = startY - 15 if startY - 15 > 15 else startY + 15
	cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)