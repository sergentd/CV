# import necessary packages
from skimage import feature
import imutils
import numpy as np
import mahotas
import cv2

class FeaturesExtractor:
    def __init__(self, features=["color"]):
      # initialize the set of features to be applied
      self.features = features
      
    def describe(self, image):
      # initialize our descriptors
      colorStats = ()
      haralick = ()
      hog = ()
      
      # extract means and standard deviations from each color channel
      # if needed -- total : 6 float values
      if "color" in self.features:
        (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        colorStats = np.concatenate([means, stds]).flatten()
        print("[INFO] colors: {}".format(colorStats))
      
      # extract haralick textures if needed
      # total : 13 float values
      if "haralick" in self.features:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        print("[INFO] haralick: {} \n len: {}".format(
		  haralick, len(haralick)))
    
      if "hog" in self.features:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = imutils.auto_canny(gray)
        resized = cv2.resize(image, (256,256))
        hog = feature.hog(resized, pixels_per_cell=(128,128),
          cells_per_block=(2,2), transform_sqrt=True, block_norm="L1")
        print("[INFO] hog: {} \n len: {}".format(hog, len(hog)))
      
      # return concatened features vector
      return np.hstack([colorStats, haralick, hog])
