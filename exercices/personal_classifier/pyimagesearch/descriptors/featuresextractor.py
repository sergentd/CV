# import necessary packages
from skimage import feature
import imutils
import numpy as np
import mahotas
import cv2

class FeaturesExtractor:
    def __init__(self, features=["color"]):
      # initialize the set of features extractor to be applied
      self.features = features
      
    def describe(self, image):
      # extract means and standard deviations from each color channel
      # if needed -- total : 6 float values
      colorStats = color_stats(image) if "color" in self.features else ()
      
      # extract haralick textures if needed
      # total : 13 float values
      haralick = haralick_texture(image) if "haralick" in self.features else ()
    
	  # extract hog features if needed
      # total : 36 float values
      hog = hist_oriented_grad(image) if "hog" in self.features else ()
      
      # return concatened features vector
      return np.hstack([colorStats, haralick, hog])
	  
def color_stats(image):
  # compute and return the means and standard 
  # deviation for each channel
  (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
  return np.concatenate([means, stds]).flatten()
	  
def haralick_texture(image):
  # convert the image to gray if the image is BGR
  if len(image.shape) == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	  
  # return the haralick texture encoding
  return mahotas.features.haralick(image).mean(axis=0)
	  
def hist_oriented_grad(image):
  # convert the image to gray if the image is BGR
  if len(image.shape) == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	 
  # edge detection using automatized canny parameters
  edged = imutils.auto_canny(image)
  
  # resize the image ignoring aspect ratio
  # to have similar descriptor for any input image
  resized = cv2.resize(edged, (256,256))
      
  # return the HOG
  return feature.hog(resized, pixels_per_cell=(128,128),
    cells_per_block=(2,2), transform_sqrt=True, block_norm="L1")
	  
