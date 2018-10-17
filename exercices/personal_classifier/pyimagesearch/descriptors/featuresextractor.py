# import necessary packages
import .collection as clt
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
 