# import necessary packages
from pyimagesearch.descriptors import DetectAndDescribe
from pyimagesearch.indexer import FeatureIndexer
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import argparse
import imutils
import cv2
import os

# constrct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="Path to input dataset (images to be indexed)")
ap.add_argument("-f", "--features-db", required=True,
    help="Path and filename of the features database")
ap.add_argument("-a", "--approx-images", type=int, default=500,
    help="Approximate number of images in the dataset")
ap.add_argument("-b", "--buffer-size", type=int, default=50000,
    help="Maximum buffer size for number of features to be stored in memory")
args = vars(ap.parse_args())

# initialize the keypoint detector
detector = FeatureDetector_create("SURF")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)

# initialize the features indexer
fi = FeatureIndexer(args["features_db"], estNumImages=args["approx_images"],
    maxBufferSize=args["max_buffer_size"], verbose=True)

# loop over the images in the dataset
for (i, imagePath) in enumerate(paths.list_images(args["dataset"])):
    # check to see if progress should be displayed
    if i > 0 and i % 10 == 0:
        fi._debug("processed {} images".format(i), msgType="[PROGRESS]")

    # extract the image filename (i.e the unique image ID) from the image
    # path, then load the image itself and convert it to grayscale
    filename = os.path.basename(imagePath)
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=320)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # describe the image
    (kps, descs) = dad.describe(image)

    # if keypoints or descriptors are none, ignore the image
    if kps is None or descs is None:
        continue

    # index the features
    fi.add(filename, kps, descs)

# finish the indexing process
fi.finish()
