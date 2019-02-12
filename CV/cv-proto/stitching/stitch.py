# import necessary packages
from pyimagesearch.datasets import SimpleDatasetRenamer
from pyimagesearch.improc import Stitcher
from imutils import paths
import imutils
import argparse
import cv2

# create the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
    help="path to the input images to stitch")
ap.add_argument("-o", "--output", default=None,
    help="path to the output stitched image")
ap.add_argument("-n", "--name", default=None,
    help="name for output file")
ap.add_argument("-c", "--crop", type=int, default=1,
    help="for deactivate cropping, set this to 0-")
ap.add_argument("-v", "--verbose", type=int, default=-1,
    help="for verbose mode, set this to 0+")
ap.add_argument("-e", "--extension", default="jpg",
    help="extension for the output file")
args = vars(ap.parse_args())

# load the images from disk
imagePaths = paths.list_images(args["images"])
images = [cv2.imread(path) for path in imagePaths]

# create the name for the stitched image
if args["name"] is not None:
    name = args["name"]
else:
    name = SimpleDatasetRenamer.id_generator(sequential=False)

# create the stitcher and stitch the images together
stitcher = Stitcher(crop=args["crop"] > 0, savePath=args["output"], name=name,
    saveExt=args["extension"], verbose=args["verbose"] >= 0)
(status, stitched) = stitcher.stitch(images)

# check to see if the stitching was successfull
if status == 0:
    stitched = imutils.resize(stitched, width=min(stitched.shape[1], 1024))
    cv2.imshow("stitched images", stitched)
    cv2.waitKey(0)
else:
    print("Error while stitching. Activate verbose mode for more details")

# cleanup
cv2.destroyAllWindows()
