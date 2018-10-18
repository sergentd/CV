# import necessary packages
from imutils import paths
import progressbar
import cv2
import os

class SimpleDatasetRenamer:
  def __init__(self, path, prefix=None, suffix=None, move=False, remove=False,
    sequential=True, length=6, ext=None):
    # store the prefix and the paths
    self.prefix = prefix
    self.suffix = suffix
    self.directory = path
    self.move = move
    self.remove = remove
    self.sequential = sequential
    self.length = length
    self.ext = ext
    
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
      # load the image from disk
      image = cv2.imread(path)
      
      # grab the data format to encode the file with
      # the same after renaming
      if self.ext is None:
        dataFormat = (path.split(os.path.sep)[-1]).split(".")[1]
      else:
        dataFormat = str(ext)
	  
	  # initialize the directory parameter
      directory = self.move if self.move is not None else os.path.dirname(path)
      
      # create a sequential *unique* ID for this image relative to
      # other images processed *at the same time*
      # OR
      # create a random id with lowercase letters and digits
      idx = str(i).zfill(self.length) if self.sequential else self.id_generator()
      
      # see if we are using prefix and/or suffix
      prefix = str(self.prefix) if self.prefix is not None else ""
      suffix = str(self.suffix) if self.suffix is not None else ""
	  
	  # construct the filename based on this scheme :
	  # {dir}{sep}[{prefix}]{idx}[{suffix}].{df}
	  # exemple : ./img-000001-root.png
	  #           /home/user/images/ef6va2.jpg
      filename = "{}{}{}{}{}.{}".format(directory, os.path.sep,
	    prefix, idx, suffix, dataFormat)
      
      # write image to disk in the approriate directory
      cv2.imwrite(filename, image)
      
      # check to see if we need to remove the old file      
      if self.remove and filename != path:
        os.remove(path)
    
      # update the progressbar (feedback to user)
      pbar.update(i)
    
    # close the progressbar
    pbar.finish()
    
  def id_generator(self, size=6, chars=string.ascii_lowercase + string.digits):
    # generate a random id with lowercase ascii chars and digits
    return ''.join(random.choice(chars) for _ in range(size))
