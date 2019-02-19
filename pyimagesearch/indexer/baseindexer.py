# import necessary packages
from __future__ import print_function
import numpy as np
import datetime

class BaseIndexer(object):
    def __init__(self, dbPath, estNumImages=500, maxBufferSize=50000,
        dbResizeFactor=2, verbose=True):
        # store the database path, the estimated number of images, the length
        # of the buffer, the resize factor of the database and the verbosity
        self.dbPath = dbPath
        self.estNumImages = estNumImages
        self.maxBufferSize = maxBufferSize
        self.dbResizeFactor = dbResizeFactor
        self.verbose = verbose

        # initialize the indexes dictionnary
        self.idxs = {}

    def _writeBuffers(self):
        pass

    def _writeBuffer(self, dataset, datasetName, buf, idxName, sparse=False):
        # if the buffer is a list, then compute the ending index based on
        # the list length
        if type(buf) is list:
            end = self.idxs[idxName] + len(buf)

        # otherwise, assume that the buffer is a Numpy/Scipy array, so
        # compute the ending based on the array shape
        else:
            end = self.idxs[idxName] + buf.shape[0]

        # check to see if the dataset needs to be resized
        if end > dataset.shape[0]:
            self._debug("triggering '{}' db resize".format(datasetName))
            self._resizeDataset(dataset, datasetName, baseSize=end)

        # if it is a sparse matrix, convert it to a dense one
        if sparse:
            buf = buf.toarray()

        # dump the buffer to file
        self._debug("writing '{}' buffer".format(datasetName))
        dataset[self.idxs[idxName]:end] = buf

    def _resizeDataset(self, dataset, dbName, baseSize=0, finished=0):
        # grab the original size of the dataset
        origSize = dataset.shape[0]

        # check to see if we are finished writing rows to the dataset, and if
        # so make the new size the current index
        if finished > 0:
            newSize = finished

        # otherwise, we want to enlarge the dataset and so calculate
        # the new size of the dataset
        else:
            newSize = baseSize * self.dbResizeFactor

        # determine the shape of the resized dataset
        shape = list(dataset.shape)
        shape[0] = newSize
        dataset.resize(tuple(shape))

        # show the old versus new dataset size
        self._debug("old size of '{}': {:,}; new size: {:,}".format(dbName,
            origSize, newSize))

    def _debug(self, msg, msgType="[INFO]"):
        # check to see if the message should be printed
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))

    @staticmethod
    def featureStack(array, accum=None, stackMethod=np.vstack):
        # if the accumulated array is None, create it
        if accum is None:
            accum = array

        # otherwise, stack the arrays
        else:
            accum = stackMethod([accum, array])

        # return the accumulated array
        return accum
