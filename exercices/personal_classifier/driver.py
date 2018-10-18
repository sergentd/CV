# import necessary packages
from pyimagesearch.datasets.facedatasetcreator import FaceDatasetCreator
from imutils.video import VideoStream
import argparse
import imutils
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
  help = "path to where the face cascade resides")
ap.add_argument("-o", "--output", required=True,
  help="path to output directory")
args = vars(ap.parse_args())

# load OpenCV's Haar cascade for face detection from disk
creator = FaceDatasetCreator(args["cascade"])

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0

# loop over the frames from the video stream
while True:
  # grab the frame from the threaded video stream, clone it, (just
  # in case we want to write it to disk), and then resize the frame
  # so we can apply face detection faster
  frame = vs.read()
  orig = frame.copy()
  
  # detect the faces in the frame and draw their boxes
  # around
  frame = creator.detect(frame)
    
  # show the output frame
  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) & 0xFF
 
  # if the `k` key was pressed, write the *original* frame to disk
  # so we can later process it and use it for face recognition
  if key == ord("k"):
    creator.write(orig, args["output"])
    total += 1
 
  # if the `q` key was pressed, break from the loop
  elif key == ord("q"):
    break
 
# print the total faces saved and do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()