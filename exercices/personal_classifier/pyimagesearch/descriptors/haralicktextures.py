# import necessary packages
import numpy as np
import cv2
import mahotas
 
class HaralickTextures: 
  def describe(self, image):
    # convert the image to gray if the image is BGR
    if len(image.shape) == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
    # return the haralick texture encoding
    return mahotas.features.haralick(image).mean(axis=0)