# import necessary packages
from . import collection as clt
import numpy as np
import cv2

class FeaturesExtractor:
    def __init__(self, descriptors=[]):
      # initialize the set of descriptors to be applied
      # via the collection manager
      self.descriptors = clt.descriptors(descriptors)
      
    def describe(self, image):
      # initialize the features
      features = []
      
      # loop over all the descriptors
      for d in self.descriptors:
        feature = d.describe(image)
        features.append(feature)
      
      # return the total features as an unique vector      
      return np.hstack(features)
      
    def add(self, descriptor):
      # add the descriptor to the list of descriptors
      # assuming it is an *instance* and NOT a keyword
      self.descriptors.append(descriptor)
      
    def add_by_keyword(self, keyword, parameters=dict()):
      # instantiate the descriptor and add it to the set
      # possibility to parametrize the instantiation
      descriptor = clt.descriptor(keyword, parameters)
      self.add(descriptor)
 