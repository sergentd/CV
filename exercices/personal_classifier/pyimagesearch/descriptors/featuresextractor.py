# import necessary packages
from . import collection as clt
import numpy as np
import cv2

class FeaturesExtractor:
    def __init__(self, descriptors=[], preprocessors=[]):
      # initialize the set of descriptors to be applied
      # via the collection manager
      self.descriptors = clt.descriptors(descriptors)
      self.preprocessors = preprocessors
      
    def describe(self, image):
      # initialize the features which will store
      # all the features describing the image
      features = []
      
      # loop over all preprocessors we need to apply
      for p in self.preprocessors:
        image = p.preprocess(image)
      
      # loop over all the descriptors and compute the
      # describing feature, then add it to the 
      for d in self.descriptors:
        feature = d.describe(image)
        features.append(feature)
      
      # return the total features as an unique vector      
      return np.hstack(features)
      
    def add(self, descriptor):
      # check to see if the descriptor is a keyword and
      # add the descriptor to the list of descriptors
      # assuming it can perform ~.describe(image)
      if isinstance(descriptor, str):
        self.add_by_keyword(descriptor)
      else:
        self.descriptors.append(descriptor)
      
    def add_by_keyword(self, keyword, parameters=dict()):
      # instanciate the descriptor and add it to the set
      # --possibility to parametrize the instanciation
      descriptor = clt.descriptor(keyword, parameters)
      self.add(descriptor)
 