# import necessary packages
from imutils import paths
import progressbar
import string
import cv2
import os

class SimpleDatasetRenamer:
  def __init__(self, path, prefix=None, suffix=None, keep_idx=False, move=None, remove=False,
    sequential=True, length=6, ext=None, index=0):
    # store the following parameters : prefix to filename, suffix to filename,
    # keep old id boolean, source directory, target directory, remove boolean,
    # sequential renaming boolean, length of idx, file extension (dataformat),
    # current index
    self.prefix = prefix
    self.suffix = suffix
    self.keep_idx = keep_idx
    self.directory = path
    self.move = move
    self.remove = remove
    self.sequential = sequential
    self.length = length
    self.ext = ext
    self.index = index
    
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
     
      # try to generate a unique filename
      filename = self.gen_filename(path)
      
      if filename is not None:
        # write image to disk in the approriate directory
        cv2.imwrite(filename, image)
      
        # check to see if we need to remove the old file      
        if self.remove and filename != path:
          os.remove(path)
    
      # update the progressbar (feedback to user)
      pbar.update(i)
    
    # close the progressbar
    pbar.finish()
  
  def gen_filename(self, path):
    # grab the data format to encode the file with
    # the same after renaming
    if self.ext is None:
      dataFormat = (path.split(os.path.sep)[-1]).split(".")[1]
    else:
      dataFormat = str(ext)
      
    # initialize the output directory parameter
    directory = self.move if self.move is not None else os.path.dirname(path)
    
    # see if we are using prefix and/or suffix
    prefix = str(self.prefix) if self.prefix is not None else ""
    suffix = str(self.suffix) if self.suffix is not None else ""
    
    while(True and self.index < 10**(self.length+1)):
      # check to see if we need to generate a unique ID	
      if not self.keep_idx:
        # loop until we have a unique id and no conflict with existing files
        # generate a tentative of unique ID
        idx = self.id_generator()
          
        # increment the current index number (preventing infinite loop)
        self.index += 1

      # generate the filename : {dir}{sep}[{prefix}]{idx}[{suffix}].{df}
      filename = os.path.sep.join([directory, "{}{}{}.{}".format(prefix, idx, suffix, dataFormat)])
          
      # allow for a unique filename only
      if not os.path.isFile(filename):
        break
        
      # if we keep the idx and the filename wasn't free at first try,
      # we will not be able to generate a unique ID so we break the loop
      # and return a None filename (so we don't erase the existing file)
      elif self.keep_idx:
        filename = None
        print("Could not save {}: existing file in target directory")
        break
        
    return filename
  
  def id_generator(self, chars=string.ascii_lowercase + string.digits):
    # create a sequential *unique* ID for this image relative to
    # other images processed *at the same time*
    # OR
    # create a random id with lowercase letters and digits
    if self.sequential:
      # return the current index number as id filled with 0s
      return str(self.index).zfill(self.length)
    else:
      # generate a random id with lowercase ascii chars and digits
      return ''.join(random.choice(chars) for _ in range(self.length))
