# import necessary packages
import imutils
import cv2

class IndexerPreprocessor:
    def __init__(self, width=320):
        # store the width of preprocessed images
        self.width = width

    def preprocess(self, image):
        # check to see if it is an image or a path to an image
        if isinstance(image, basestring):
            # load the image and convert it to grayscale
            image = cv2.imread(image)

        # resize the image and convert it to grayscale
        image = imutils.resize(image, width=self.width)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # return the preprocessed image
        return image
