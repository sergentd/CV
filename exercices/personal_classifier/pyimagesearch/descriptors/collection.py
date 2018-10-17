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
  # initialize the set of descriptors     
       = []
  
  # loop over all descriptors to instanciate them
  for name in descriptors:
    d = descriptor(name)
	if d is not None:
          .append(d)
  
  # return the set of generated descriptors     
  return     
  
def descriptor(keyword):
  # check to see if the keyword is known
  if keyword not in LST_DESC:
    print("warning: descriptor {} not loaded (unknown keyword) \n"
    "Value of 'keyword' must be in {}".format(keyword, LST_DESC.keys()))
    return None
  
  # return the descriptor instance
  return LST_DESC[keyword]()
  