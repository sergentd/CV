# import necessary packages
import numpy as np
import cv2
import os

class SimpleDatasetLoader:
  def __init__(self, preprocessors=None):
    self.preprocessors = preprocessors

    # if the preprocessors are None,
    # init an empty list
    if self.preprocessors is None:
      self.preprocessors = []


  def load(self, imagePaths, verbose=-1):
    # initialize the list of features and labels
    data = []
    labels = []
    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
      # load the image and extract the class label assuming
      # that our path has the following format :
      # /path/to/dataset/{class}/{image}.jpg
      image = cv2.imread(imagePath)
      label = imagePath.split(os.path.sep)[-2]

      # loop over the preprocessors
      if self.preprocessors is not None:
        for p in self.preprocessors:
          image = p.preprocess(image)

      # treat our processed image as a feature vector
      # by updating the data list followed by labels
      data.append(image)
      labels.append(label)

      # show an update every verbose image
      if verbose > 0 and i > 0 and (i+1) % verbose == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

    # return a tuple of data and labels
    return (np.array(data),np.array(labels))
