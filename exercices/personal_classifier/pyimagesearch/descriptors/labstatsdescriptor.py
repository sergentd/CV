# import necessary packages
import numpy as np
import cv2

class LabStatsDescriptor: 
  def describe(self, image):
    # compute and return the means and standard 
    # deviation for each channel
    (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    return np.concatenate([means, stds]).flatten()