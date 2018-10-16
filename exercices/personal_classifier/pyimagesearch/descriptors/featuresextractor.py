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
      
      # extract haralick textures if needed
      # total : 13 float values
      if "haralick" in self.features:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haralick = mahotas.features.haralick(gray).mean(axis=0)
    
	  # extract hog features if needed
      # total : 36 float values
      if "hog" in self.features:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = imutils.auto_canny(gray)
        resized = cv2.resize(image, (256,256))
        hog = feature.hog(resized, pixels_per_cell=(128,128),
          cells_per_block=(2,2), transform_sqrt=True, block_norm="L1")
      
      # return concatened features vector
      return np.hstack([colorStats, haralick, hog])
