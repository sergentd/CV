# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
    help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
    help="path to the pre-trained emotion detector CNN")
ap.add_argument("-v", "--video", help="path to optional video file")
args = vars(args.parse_args())

# load the face detector cascade, emotion detection CNN and then
# define the list of emotion labels
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised", "neutral"]

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(1)

# otherwise load the video
else:
    camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a
    # frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # initialize the canvas for the visualisation, then clone
    # the frame so we can draw on it
    canvas = np.zeros((220, 300, 3), dtype="uint8")
    clone = frame.copy()

    # detect faces in the input frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # ensure at least one face was found before continuing
    if len(rects) > 0:
        # determine the largest face area
        rect = sorted(rects, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = rect

        # etract the face ROI from the image, then pre-process
        # it for network
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make the prediction on the ROI, then lookup the class label
        preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        # loop over the labels and probabilities and draw them
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            # draw the label and probability bar on the canvas
            w = int(prob * 300)
            cv2.rectangle(canvas, (5, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 0, 255) -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # draw the label on the face
        cv2.putText(clone, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(clone, (fX, fY), (fX+fW, fY+fH),
            (0, 0, 255), 2)

    # show the face and probabilities
    cv2.imshow("Face", clone)
    cv2.imshow("Probabilities", canvas)

    # if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
