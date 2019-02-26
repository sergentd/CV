# USAGE
# python bw2color_video.py --prototxt model/colorization_deploy_v2.prototxt --model model/colorization_release_v2.caffemodel --points model/pts_in_hull.npy
# python bw2color_video.py --prototxt model/colorization_deploy_v2.prototxt --model model/colorization_release_v2.caffemodel --points model/pts_in_hull.npy --input video/jurassic_park_intro.mp4

# import the necessary packages
from pyimagesearch.improc import Colorizer
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
    help="path to optional input video (webcam will be used otherwise)")
ap.add_argument("-p", "--prototxt", type=str, required=True,
    help="path to Caffe prototxt file")
ap.add_argument("-m", "--model", type=str, required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--clusters", type=str, required=True,
    help="path to cluster center points")
ap.add_argument("-w", "--width", type=int, default=500,
    help="input width dimension of frame")
args = vars(ap.parse_args())

# initialize a boolean used to indicate if either a webcam or input
# video is being used
webcam = not args.get("input", False)

# if a video path was not supplied, grab a reference to the webcam
if webcam:
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

# load our serialized black and white colorizer model and cluster
# center points from disk
print("[INFO] loading colorizer...")
colorizer = Colorizer(args["prototxt"], args["model"], args["clusters"], False)

# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
    frame = vs.read()
    frame = frame if webcam else frame[1]

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
    if not webcam and frame is None:
        break

	# predict the colorized image
    (orig, gray, colorized) = colorizer.predict(frame, 500, False, True)

	# show the original and final colorized frames
    cv2.imshow("Original", orig)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Colorized", colorized)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# if we are using a webcam, stop the camera video stream
if webcam:
    vs.stop()

# otherwise, release the video file pointer
else:
    vs.release()

# close any open windows
cv2.destroyAllWindows()
