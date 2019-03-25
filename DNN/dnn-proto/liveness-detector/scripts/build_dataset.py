# import necessary packages
from conf import builder_conf as conf
import numpy as np
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
    help="path to input video")
ap.add_argument("-d", "--dataset-type", type=str, required=True,
    help="type of dataset to build (can be either 'real' or 'fake')")
args = vars(ap.parse_args())

# load the serialized face detector from disk
print("[INFO] loading face-detector...")
net = cv2.dnn.readNetFromCaffe(conf.PROTO_PATH, conf.CAFFE_PATH)

# open a pointer to the video file stream and initialize
# the total number of frames read and saved thus far
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0

# loop over frames from the video file stream
while True:
    # grab the frame from the video file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached
    # the end of the file
    if not grabbed:
        break

    # increment the total number of frames
    read += 1

    # check to see if we should process this frame
    if read % conf.SKIP_FRAMES != 0:
        continue

    # grab the frame dimensions and construct a blob from the
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the
    # detections predictions
    net.setInput(blob)
    detections = net.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only one face,
        # so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > conf.CONFIDENCE_THRESH:
            # compute the (x, y)-coordinates of the bounding box
            # of the face and then extract the ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]

            # derive the paths
            base = os.path.sep.join([conf.DATASET_DIR, args["dataset_type"]])
            p = os.path.sep.join([base, "{}.png".format(saved)])

            # check to see if we need to create the output directory
            if not os.path.exists(base):
                os.mkdirs(base)

            # write image to disk
            cv2.imwrite(p, face)
            saved += 1
            print("[INFO] saved {} to disk".format(p))

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()
