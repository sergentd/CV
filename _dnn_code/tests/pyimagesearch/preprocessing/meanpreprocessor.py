# import necessary packages
import cv2

class MeanPreprocessor:
  def __init__(self, rMean, gMean, bMean):
    # store the averages accross a training set
    self.rMean = rMean
	self.gMean = gMean
	self.bMean = bMean

  def preprocess(self, image):
    # split the image into its respective channels
	(B, G, R) = cv2.split(image.astype("float32"))
	
	# subtract the means for each channels
	R -= self.rMean
	G -= self.gMean
	B -= self.bMean
	
	# merge the channels back together and return the image
	return cv2.merge([B, G, R])