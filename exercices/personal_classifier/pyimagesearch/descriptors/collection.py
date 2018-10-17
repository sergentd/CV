# import necessary packages
from .bgrstatsdescriptor import BGRStatsDescriptor
from .hsvstatsdescriptor import HSVStatsDescriptor
from .labstatsdescriptor import LabStatsDescriptor
from .bgrdescriptor import BGRDescriptor
from .hsvdescriptor import HSVDescriptor
from .labdescriptor import LabDescriptor
from .haralickdescriptor import HaralickDescriptor
from .humomentdescriptor import HuMomentDescriptor
from .hogdescriptor import HOGDescriptor

LST_DESC = {
  "bgr":BGRDescriptor,
  "hsv":HSVDescriptor,
  "lab":LabDescriptor,
  "lab_s":LabStatsDescriptor,
  "hsv_s":HSVStatsDescriptor,
  "bgr_s":BGRStatsDescriptor,
  "haralick":HaralickDescriptor,
  "hu":HuMomentDescriptor,
  "hog":HOGDescriptor
}

def descriptors(descriptors):
  # initialize the set of descriptors instances
  instances = []
  
  # loop over all descriptors to instanciate them
  for name in descriptors:
    d = descriptor(name)
    if d is not None:
      instances.append(d)
  
  # return the set of generated descriptors instances
  return instances
  
def descriptor(keyword, parameters=dict()):
  # check to see if the keyword is known
  if keyword not in LST_DESC:
    print("warning: descriptor {} not loaded (unknown keyword) \n"
    "Value of 'keyword' must be in {}".format(keyword, LST_DESC.keys()))
    return None
  
  # return the descriptor instance
  return LST_DESC[keyword](**parameters)
  