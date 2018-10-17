# import necessary packages
from skimage import feature
import imutils
import numpy as np
import mahotas
import cv2

class FeaturesExtractor:
    def __init__(self, descriptors=[]):
      # initialize the set of descriptors to be applied
      self.descriptors = [LIST_DESCRIPTORS[d]() for d in descriptors]
      
    def describe(self, image):
      # initialize the features
      features = []
      
      # loop over all the descriptors
      for d in self.descriptors:
        feature = d.describe(image)
        features.append(feature)
        
      return np.hstack(features)

class BGRStats: 
  def describe(self, image):
    # compute and return the means and standard 
    # deviation for each channel in RGB color space
    (means, stds) = cv2.meanStdDev(image)
    return np.concatenate([means, stds]).flatten()

class HSVStats: 
  def describe(self, image):
    # compute and return the means and standard 
    # deviation for each channel
    (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    return np.concatenate([means, stds]).flatten()
  
class LabStats: 
  def describe(self, image):
    # compute and return the means and standard 
    # deviation for each channel
    (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    return np.concatenate([means, stds]).flatten()
      
class HaralickTextures: 
  def describe(self, image):
    # convert the image to gray if the image is BGR
    if len(image.shape) == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
    # return the haralick texture encoding
    return mahotas.features.haralick(image).mean(axis=0)

class HuMoment: 
  def describe(self, image):
    # compute the Hu Moments feature vector for the entire image
    return cv2.HuMoments(cv2.moments(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))).flatten()
  
class HOG:
  def __init__(self, cvt=True, canny=True,
    dim=(256,256), pxl_p_cel=(32,32), cel_p_blk=(2,2)):
    # initialize the HOG parameters
    self.cvt = cvt
    self.canny = canny
    self.dim = dim
    self.pxl_p_cel = pxl_p_cel
    self.cel_p_blk = cel_p_blk
    
  def describe(self, image):    
    # convert the image to gray if the image is BGR
    # and apply canny edge detection
    if len(image.shape) == 3 and self.cvt:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    # edge detection using automatized canny parameters
    if self.canny:
      image = imutils.auto_canny(image)
  
    # resize the image ignoring aspect ratio
    # to have similar descriptor for any input image
    resized = cv2.resize(image, self.dim)
      
    # return the HOG
    return feature.hog(resized, pixels_per_cell=pxl_p_cel,
      cells_per_block=cel_p_blk, transform_sqrt=True, block_norm="L1")

LIST_DESCRIPTORS = dict(
  BGRStats=BGRStats,
  HSVStats=HSVStats,
  LabStats=LabStats,
  HaralickTextures=HaralickTextures,
  HuMoment=HuMoment,
  HOG=HOG
)  