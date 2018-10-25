# import necessary packages
from imutils.video import FPS
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# contruct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-l", "--label", required=True, help="class label we are interested to detect and track")
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
    
# load the serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream
print("[INFO] starting video stream...")
# check to see if we are using camera or file
if not args.get("video", False):
  print("[INFO] starting video stream...")
  vs = VideoStream(src=0).start()
  time.sleep(1.0)
  
# grab the reference of the video file
else:
  vs = cv2.VideoCapture(args["video"])
  
# initialize dlib correlation tracker, output video writer
# and predicted class label
tracker = None
writer = None
label = ""

# start the frames per second estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
  # grab the reference for the frame and handle the VideoCapture and
  # VideoStream possibilities
  frame = vs.read()
  frame = frame[1] if not args.get("video", False) else frame
  
  if frame is None:
    break
    
  # resize the frame for faster processing and then convert the
  # frame from BGR to RGB (dlib need RGB)
  frame = imutils.resize(frame, width=600)
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
  # if we are supposed to be writing a video to disk,
  # initialize the writer
  if args["output"] is not None and writer is None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    
  # if the correlation object tracker is None we need to apply an object detector
  # to seed the tracker with someting to actually track
  if tracker is None:
    # grab the frame dimensions and convert the frame to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (h, w), 127.5)
  
    net.setInput(blob)
    detections = net.forward()
  
    # ensure at leat one detection is made
    if len(detections) > 0:
      # find the index of the detection with the largest probability
      i = np.argmax(detections[0, 0, :, 2])
    
      # grab the probability associated with the object along
      # with its class label
      conf = detections[0, 0, i, 2]
      label = CLASSES[int(detections[0, 0, i, 1])]
    
      # filter weak detections by requiring a minimum confidence
      if conf > args["confidence"] and label == args["label"]:
        # compute the (x, y)-coordinates of the bounding box
        # for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
      
        # construct a dlib rectangle object from the bounding box
        # coordinates and then start the dlib correlation tracker
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(startX, startY, endX, endY)
        tracker.start_track(rgb, rect)
        
        # draw the bounding box and text for the object
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
          (0, 255, 0), 2)
   
  # otherwise the detection has already been done
  else:
    # update the tracker and grab the position of the tracked object
    tracker.update(rgb)
    pos = tracker.get_position()

    # unpack the position object
    startX = int(pos.left())    
    startY = int(pos.top())    
    endX = int(pos.right())    
    endY = int(pos.bottom())
    
    # draw the bounding box from the correlation object tracker
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
       (0, 255, 0), 2)
       

  # check to see if we should write the frame to disk
  if writer is not None:
    writer.write(frame)
    
  # show the output frame
  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) & 0xFF
  
  # if the 'q' key was pressed
  if key == ord("q"):
    break
  
  fps.update()
  
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
  writer.release()
  
# cleanup
cv2.destroyAllWindows()

# if we are using a webcam, release the pointer
if not args.get("video", False):
  vs.stop()

# otherwise, release the file pointer
else:
  vs.release()