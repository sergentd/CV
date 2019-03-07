# USAGE
# python neural_style_transfer.py --image images/baden_baden.jpg --model models/instance_norm/starry_night.t7

# import the necessary packages
from conf import config as conf
import argparse
import imutils
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="neural style transfer model")
ap.add_argument("-i", "--image", required=True,
    help="input image to apply neural style transfer to")
ap.add_argument("-o", "--output-path", type=str, default=None,
    help="path to output image where the image will be saved")
ap.add_argument("-w", "--width", type=int, default=600,
    help="size of the input image for neural style tranfer")
ap.add_argument("-s", "--output-width", type=int, default=600,
    help="size of the image after the neural style transfer")
args = vars(ap.parse_args())

# load the neural style transfer model from disk
print("[INFO] loading style transfer model...")
net = cv2.dnn.readNetFromTorch(args["model"])

# load the input image, resize it to have a width of 600 pixels, and
# then grab the image dimensions
image = cv2.imread(args["image"])
image = imutils.resize(image, width=args["width"])
(h, w) = image.shape[:2]

# construct a blob from the image, set the input, and then perform a
# forward pass of the network
blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
	(103.939, 116.779, 123.680), swapRB=False, crop=False)
net.setInput(blob)
start = time.time()
output = net.forward()
end = time.time()

# reshape the output tensor, add back in the mean subtraction, and
# then swap the channel ordering and resize the image
output = output.reshape((3, output.shape[2], output.shape[3]))
output[0] += 103.939
output[1] += 116.779
output[2] += 123.680
output = output.transpose(1, 2, 0)
output = imutils.resize(output, width=args["output_width"])

# show information on how long inference took
print("[INFO] neural style transfer took {:.4f} seconds".format(end - start))

# if an output path has been given, write the image
if args["output_path"] is not None:
    cv2.imwrite(args["output_path"], output)

# put the legend on the image
output[0:conf.S_LEGEND_HEIGHT, 0:conf.LEGEND_WIDTH] = 0
cv2.putText(output, "press any key to continue...", (30, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

# show the image
output /= 255.0
cv2.imshow("Deep-art output", output)
cv2.waitKey(0)
