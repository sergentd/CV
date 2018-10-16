# import necessary packages
import numpy as np
import mahotas
import cv2

class FeaturesExtractor:
	def __init__(self, features=["color"])
	  # initialize the set of features to be applied
	  self.features = features
	  
	def describe(image):
	  colorStats = Tuple()
	  haralick = Tuple()
	  hog = Tuple()
	  # extract means and standard deviations from each color channel
	  # if needed -- total : 6 float values
	  if "color" in features:
	    (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
	    colorStats = np.concatenate([means, stds]).flatten()
	  
	  # extract haralick textures if needed
	  # total : 
	  if "haralick" in features:
	    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	    haralick = mahotas.features.haralick(gray).mean(axis=0)
	    print(haralick)
	
	  if "hog" in features:
	    pass
	  
	  # return concatened features vector
	  return np.hstack([colorStats, haralick, hog])
