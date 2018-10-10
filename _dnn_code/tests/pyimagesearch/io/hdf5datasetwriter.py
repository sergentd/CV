# import necessary packages
import h5py
import os

class HDF5DatasetWriter:
  def __init__(self, dims, outputPath, dataKey="images", bufSize=1000)
    # check to see if the output path exist, and if so raise an error
    if os.path.exists(outputPath):
      raise ValueError("The supplied 'outputPath' already exists"
            " and cannot be overwritten. Manually delete the"
            "file before continuing", outputPath)
    
    # open the HDF5 databse for writing and create two datasets:
    # one to store images/features, the other for class labels
    self.db = h5py.File(outputPath, "w")
    self.data   = self.db.create_dataset(dataKey, dims, dtype="float")
    self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")
    
    # store the buffer size, then init the buffer itself
    # along with the index into the datasets
    self.bufSize = bufSize
    self.buffer  = {"data": [], "labels": []}
    self.idx = 0
    
  def add(self, rows, labels):
    #add the rows dans labels to the buffer
    self.buffer["data"].extend(rows)
    self.buffer["labels"].extend(labels)
    
    # check to see if the buffer needs to be flushed to disk
    if (len(self.buffer["data"] >= self.bufSize):
      self.flush()
      
  def flush(self):
    #write the buffer to disk and then reset the buffer
    i = self.idx + len(self.buffer["data"])
    self.data[self.idx:i]   = self.buffer["data"]
    self.labels[self.idx:i] = self.buffer["labels"]
    self.idx = i
    self.buffer = {"data" : [], "labels"=[]}
    
  def storeClassLabels(self, classLabels):
    # create the dataset to store the actual class label names
    # and then store the class labels
    dt = h5py.special_dtype(vlen=unicode)
    labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt)
    labelSet[:] = classLabels
    
  def close(self):
    # check to see if there are any other entries in the
    # buffer that need to be flushed
    if len(self.buffer["data"]) > 0:
      self.flush()

    self.db.close()      