# import necessary packages
import cv2

class HuMomentDescriptor: 
  def describe(self, image):
    # compute the Hu Moments feature vector for the entire image
    return cv2.HuMoments(cv2.moments(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))).flatten()