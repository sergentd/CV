# import necessary packages
from imutils import paths
import progressbar
import cv2
import os

class SimpleDatasetRenamer:
  def __init__(self, path, prefix=None, suffix=None, move=False, remove=False):
    # store the prefix and the paths
    self.prefix = prefix
    self.suffix = suffix
    self.directory = path
    self.move = move
    self.remove = remove
    
  def rename(self):
    # grab the reference to the list of images
    imagePaths = sorted(list(paths.list_images(self.directory)))
    
    # initialize the progressbar (feedback to user on the task progress)
    widgets = ["Renaming Dataset: ", progressbar.Percentage(), " ",
      progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(imagePaths),
      widgets=widgets).start()
    
    # loop over each image, then rename it and save it in its
    # original data format
    for (i, path) in enumerate(imagePaths):
      # grab the original image and the data format 
      # to encode the file with the same after renaming
      image = cv2.imread(path)
      original = path.split(os.path.sep)[-1]
      dataFormat = original.split(".")[1]
      
      # create a *unique* ID for this image relative to other processed images
      # *at the same time*
      idx = str(i).zfill(6)
      
      # construct the filename based on prefix, idx, suffix and dataformat  
      prefix = str(self.prefix) if self.prefix is not None else ""
      suffix = str(self.suffix) if self.suffix is not None else ""
      filename = "{}{}{}.{}".format(prefix, idx, suffix, dataFormat)
      
      # write image to disk in the approriate directory
      if self.move:
        cv2.imwrite(filename, image)
      else:
        directory = os.path.dirname(path) + str(os.path.sep)
        filename = directory + filename
        cv2.imwrite(filename, image)
      
      # check to see if we need to remove the old file      
      if self.remove:
        os.remove(path)
    
      # update the progressbar (feedback to user)
      pbar.update(i)
    
    # close the progressbar
    pbar.finish()
