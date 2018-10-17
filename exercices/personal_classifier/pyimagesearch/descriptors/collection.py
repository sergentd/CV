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

def descriptors(descs)
  # initialize the set of descriptors instances
  instances = []
  
  # loop over all descriptors to instanciate them
  for d in descs:
    instances.append(LST_DESC[d]())
  
  # return the set of generated descriptors instances
  return instances