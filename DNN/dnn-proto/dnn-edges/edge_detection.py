# import necessary packages
import argparse
import cv2
import os

class CropLayer(object):
    def __init__(self, params, blob):
        # initialize the starting and ending (x, y)-coordinates of the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (h, w) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + w
        self.endY = self.startY + h

        # return the shape of the volume (the crop will occur during
        # the forward pass)
        return [[batchSize, numChannels, h, w]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
            self.startX:self.endX]]

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", type=str, required=True,
    help="path to OpenCV's edges detector pre-trained model")
ap.add_argument("-i", "--image", type=str, required=True,
    help="path to input image to apply edges detection")
args = vars(ap.parse_args())

# load the serialized edge detector from disk
print("[INFO] loading edge detector...")
protoPath = os.path.sep.join([args["edge_detector"],
    "deploy.prototxt"])
modelPath = os.path.sep.join([args["edge_detector"],
    "hed_pretrained_bsds.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# registrer the new layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)

# load the input image and grabs its dimensions
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]

# convert the image grayscale, blur it, and perform canny edge detections
print("[INFO] performing Canny Edge detection...")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5))
canny = cv2.Canny(blurred, 30, 150)

# construct a blob out of the input image for the HED
# edges dection
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(h, w),
    mean=(104.006, 116.668, 122.678), swapRB=False, crop=False)

# set the blob as the input to the network and perform a forward pass
# to compute the edges
print("[INFO] performing holistically-nested edge detection")
net.setInput(blob)
hed = net.forward()
hed = cv2.resize(hed[0, 0], (w, h))
hed = (255 * hed).astype("uint8")

# show the output edge detection results for canny and
# holistically-nested edge detection
cv2.imshow("Input", image)
cv2.imshow("Canny", canny)
cv2.imshow("HED", hed)
cv2.waitKey(0)
