from pyimagesearch.descriptors.featuresextractor import FeaturesExtractor
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image")

fe = FeaturesExtractor(["color", "haralick", "hog"])
im = cv2.imread(args["image"])
d = fe.describe(im)
print(d)