# import necessary packages
import numpy as np
import datetime
import imutils
import cv2

class Colorizer:
    def __init__(self, prototxt, model, points, verbose=True):
        # store the Caffe prototxt and Caffe model
        self.prototxt = prototxt
        self.model = model
        self.points = points
        self.verbose = verbose

        # load the network and set the proper parameters of Colorizer
        self._loadNet()

    def convert(self, image, width, stream):
        self._debug("converting and rescaling image...")

        # check to see if we need to load the image
        if not stream:
            orig = cv2.imread(image)

        # otherwise, it is a videostream
        else:
            orig = image

        # check to see if we need to resize the original image
        if width > 0:
            imutils.resize(orig, width=width)

        # rescale the image into the range [0, 1]
        scaled = orig.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        return (orig, lab)

    def resize(self, image):
        # resize the image to a fixed dimension, split the channels, extract
        # the L channel and then perform mean centering
        resized = cv2.resize(image, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        # return the processed image along with the L intensity
        return (resized, L)

    def preprocess(self, image, width, stream):
        (orig, lab) = self.convert(image, width, stream)
        (resized, L) = self.resize(lab)

        # return the original image, the Lab converted image and the
        # converted + resized image
        return (orig, lab, resized, L)

    def postprocess(self, L, ab):
        # grab the original L channel and concatenate
        # it with the predicted ab channels
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

        # convert the output image from the Lab color space to RGB, then clip
        # the values into the range [0, 1]
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)

        # convert the image to a 8 unsigned integer
        colorized = (255 * colorized).astype("uint8")

        # return the deprocessed image
        return colorized


    def colorize(self, processed):
        # pass the L channel through the network which will predict the 'a' and
        # 'b' channels values
        (orig, lab, resized, L) = processed
        self._debug("colorizing image...")
        self.net.setInput(cv2.dnn.blobFromImage(L))
        ab = self.net.forward()[0, :, :, :].transpose((1, 2, 0))

        # resize the predicted 'ab' volume to the same dimensions as the input
        # image
        ab = cv2.resize(ab, (orig.shape[1], orig.shape[0]))

        # deprocess the image
        colorized = self.postprocess(cv2.split(lab)[0], ab)

        # return the deprocessed colorized image
        return colorized

    def show(self, gray, colorized, orig=None):
        # show the images
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Colorized", colorized)

        # check to see if a validation image has been given
        if orig is not None:
            cv2.imshow("Original", orig)

        # wait for the user
        cv2.waitKey(0)

    def predict(self, image, width=-1, show=False, stream=False):
        # check to see if it is a single image
        if not stream:
            orig = cv2.imread(image)

        # otherwise it is a videostream
        else:
            orig = image

        # process the image and then colorize it
        colorized = self.colorize(self.preprocess(image, width, stream))
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

        # show the results
        if show:
            self.show(gray, colorized, orig)

        # return a tuple with the original image, the grayscale image and the
        # colorized image
        return (orig, gray, colorized)

    def _loadNet(self):
        # load the serialized black and white Colorizer and cluster centers
        # from disk
        self._debug("loading model...")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        self.pts = np.load(self.points)

        # add cluster centers as 1x1 convolution to the model
        class8 = self.net.getLayerId("class8_ab")
        conv8 = self.net.getLayerId("conv8_313_rh")
        self.pts = self.pts.transpose().reshape(2, 313, 1, 1)
        self.net.getLayer(class8).blobs = [self.pts.astype("float32")]
        self.net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    def _debug(self, msg, msgType="[INFO]"):
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))
