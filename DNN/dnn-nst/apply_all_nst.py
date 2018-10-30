# USAGE
# python apply_all_nst.py --image images/baden_baden.jpg --models models/ --output-path arts/

# import the necessary packages
from imutils import paths
import argparse
import imutils
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
	help="neural style transfer model")
ap.add_argument("-i", "--image", required=True,
	help="input image to apply neural style transfer to")
ap.add_argument("-o", "--output-path", type=str, default=None)
args = vars(ap.parse_args())

# grab the path to all neural transfer (.t7 ext) in directory
modelPaths = paths.list_files(args["models"], validExts=(".t7",))
modelPaths = sorted(list(modelPaths))

# generate unique IDs for each model, and create a list of it
models = list(zip(range(0, len(modelPaths)), (modelPaths)))

# load the input image, resize it to have a width of 600 pixels, and
# then grab the image dimensions
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# loop over the models
for (i, modelPath) in models:
  # load the neural style transfer model from disk
  print("[INFO] loading style transfer model...")
  net = cv2.dnn.readNetFromTorch(modelPath)

  # construct a blob from the image, set the input, and then perform a
  # forward pass of the network
  blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
	(103.939, 116.779, 123.680), swapRB=False, crop=False)
  net.setInput(blob)
  start = time.time()
  output = net.forward()
  end = time.time()

  # reshape the output tensor, add back in the mean subtraction, and
  # then swap the channel ordering
  output = output.reshape((3, output.shape[2], output.shape[3]))
  output[0] += 103.939
  output[1] += 116.779
  output[2] += 123.680
  output /= 255.0
  output = output.transpose(1, 2, 0)

  #  show information on how long inference took
  print("[INFO] neural style transfer took {:.4f} seconds".format(
	end - start))

  # if we need to write the result on disk
  if args["output_path"] is not None:
    # grab the path to the output filename
    outPath = "{}{}-{}-{}.jpg".format(args["output_path"],
               args["image"].split(os.path.sep)[-1], i,
               modelPath.split(os.path.sep)[-1])
    out = output*255
    cv2.imwrite(outPath, out)
    print("[INFO] writing image to disk as {}".format(outPath))


  # show the images
  cv2.imshow("Input", image)
  cv2.imshow("Output", output)

  cv2.waitKey(0)

# cleanup
cv2.destroyAllWindows()
