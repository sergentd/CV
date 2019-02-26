# import necessary packages
import numpy as np
import datetime
import cv2

class Colorizer:
    def __init__(self, prototxt, model, points):
        # store the Caffe prototxt and Caffe model
        self.prototxt = prototxt
        self.model = model
        self.points = points

        # load the network and set the proper parameters of Colorizer
        self._loadNet()

    def process(self, image):
        # load and convert the image to Lab color space
        self._debug("converting and rescaling image...")
        image = cv2.imread(image)
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        # resize the image to a fixed dimension, split the channels, extract
        # the L channel and then perform mean centering
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        # return the processed image along with the L intensity
        return (lab, resized, L)

    def deprocess(self, L, ab):
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


    def colorize(self, image):
        # pass the L channel through the network which will predict the 'a' and
        # 'b' channels values
        self._debug("colorizing image...")
        (lab, resized, L) = self.process(image)
        self.net.setInput(cv2.dnn.blobFromImage(L))
        ab = self.net.forward()[0, :, :, :].transpose((1, 2, 0))

        # resize the predicted 'ab' volume to the same dimensions as the input
        # image
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

        # deprocess the image
        colorized = self.deprocess(cv2.split(lab)[0], ab)

    def show(self, image, colorized, val=None):
        # show the images
        cv2.imshow("Grayscale", image)
        cv2.imshow("Colorized", colorized)

        # check to see if a validation image has been given
        if val is not None:
            cv2.imshow("Original", val)

        # wait for the user
        cv2.waitKey(0)

    def predictAndCompare(self, image):
        # load and convert the colored image to a grayscale image,
        # and then convert it back to a BGR representation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # perform the image colorization
        colorized = self.colorize(bgr)

        # show the images for validation purpose
        self.show(bgr, colorized, image)

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
        print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))
