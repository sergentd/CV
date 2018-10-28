# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png

# import the necessary packages
from pyimagesearch.face import FaceEncoder
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
# data = pickle.loads(open(args["encodings"], "rb").read())
encoder = FaceEncoder().load_encodings(args["encodings"])

# load the input image and convert it from BGR to RGB
image = cv2.imread(args["image"])

# try to recognize faces in the image
print("[INFO] recognizing faces...")
encoder.recognize(image)
encoder.draw(image)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
