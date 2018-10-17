# import necessary packages
from .bgrstats import BGRStats
from .hsvstats import HSVStats
from .labstats import LabStats
from .haralicktextures import HaralickTextures
from .humoment import HuMoment
from .hog import HOG

LST_DESC = {
  "bgr":BGRStats,
  "hsv":HSVStats,
  "lab":LabStats,
  "haralick":HaralickTextures,
  "hu":HuMoment,
  "hog":HOG
}

def descriptors(descriptors):
  # initialize the set of descriptors instances
  instances = []
  
  # loop over all descriptors to instanciate them
  for name in descriptors:
    instances.append(descriptor(name))
  
  # return the set of generated descriptors instances
  return instances
  
def descriptor(keyword):
  # check to see if the keyword is known
  if keyword not in LST_DESC:
    raise ValueError("Value of 'keyword' must be in {}".format(LST_DESC.keys()))
  
  # return the descriptor instance
  return LST_DESC[keyword]()
  