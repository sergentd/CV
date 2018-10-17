# import necessary packages
from . import collection as clt
import numpy as np

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
      
    def add(descriptor):
      # add the descriptor to the list of descriptors
      # assuming it is an *instance* and NOT a keyword
      self.descriptors.append(descriptor)
      
    def add_by_keyword(keyword):
      # instantiate the descriptor and add it to the set
      descriptor = clt.descriptor(keyword)
      self.descriptors.add(descriptor)
 