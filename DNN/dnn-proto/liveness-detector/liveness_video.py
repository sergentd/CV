# import necessary packages
from scripts.conf import builder_conf as conf
from scripts.conf import deploy_conf as deploy
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
    help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
    help="path to label encoder")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
net = cv2.dnn.readNetFromCaffe(conf.PROTO_PATH, conf.CAFFE_PATH)

# load the liveness detector model and label encoder from disk
print("[INFO] loading model...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream and allow camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 600px
    frame = vs.read()
    frame = imutils.resize(frame, width=deploy.LIVE_WIDTH)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob throuh prediction
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e probability) associated predictions
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > deploy.CONFIDENCE_THRESH:
            # compute the (x, y)-coordinates of the bounding box of
            # the face and extract the ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the detected bounding box doesnt fall outside
            # the dimension of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # extract the face ROI and then preprocess it in the
            # exact same manner as our training data
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # pass the face ROI through the trained model to determine
            # if the face is real or fake
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]

            # draw the label and bounding box on the frame
            label = "{}: {:.4f}".format(label, preds[j])
            cv2.putText(frame, label, (startX, startY-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # show the output frame and wait for a key press
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key was pressed (for quit), break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
