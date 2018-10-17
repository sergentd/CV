# import necessary packages
from skimage import feature
import imutils
import numpy as np
import cv2

class HOGDescriptor:
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
    return feature.hog(resized, pixels_per_cell=self.pxl_p_cel,
      cells_per_block=self.cel_p_blk, transform_sqrt=True, block_norm="L1")