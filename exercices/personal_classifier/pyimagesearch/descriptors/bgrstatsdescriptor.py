# import necessary packages
import numpy as np
import cv2

class BGRStatsDescriptor:
  def describe(self, image):
    # compute and return the means and standard 
    # deviation for each channel in RGB color space
    (means, stds) = cv2.meanStdDev(image)
    return np.concatenate([means, stds]).flatten()