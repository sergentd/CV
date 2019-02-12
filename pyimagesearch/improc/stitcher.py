# import necessary packages
import numpy as np
import imutils
import datetime
import uuid
import cv2

class Stitcher:
    def __init__(self, crop=False, savePath=None, name=None, saveExt="jpg",
        verbose=True):
        # create the stitcher according to the OpenCV version
        if imutils.is_cv4():
            self.stitcher = cv2.Stitcher_create()
            self.cv = "OpenCV 4+"
        else:
            self.stitcher = cv2.createStitcher()
            self.cv = "OpenCV 3+"

        # store the cropping, saving and verbosity parameters
        self.crop = crop
        self.name = name
        self.savePath = savePath
        self.saveExt = saveExt
        self.verbose = verbose

    def stitch(self, images):
        # try to stitch the images
        msg = "Done."
        (status, stitched) = self.stitcher.stitch(images)

        # if the stitching process failed
        if status != 0:
            # set the stitching image to None
            msg = "stitching failed : error status {}".format(status)
            stitched = None

            # return a tuple of status and None
            self._debug(msg)
            return (status, stitched)

        # otherwise, check if we need to crop the center of the panorama
        self._debug_pano(stitched, "stitched panorama")
        if self.crop:
            stitched = self.cropping(stitched)
            self._debug_pano(stitched, "cropped panorama")

        if self.savePath is not None:
            name = uuid.uuid4() if self.name is None else self.name
            path = "{}/{}.{}".format(self.savePath, name, self.saveExt)
            cv2.imwrite(path, stitched)
            self._debug("saved in {}".format(path))

        # return a tuple of status and the stitched image
        return (status, stitched)

    def cropping(self, image):
        # make a copy of the image surrounded by a 10px border,
        # then treshold the grayscale version of the stitched image
        stitched = cv2.copyMakeBorder(image, 10, 10, 10, 10,
            cv2.BORDER_CONSTANT, (0, 0, 0))
        thresh = cv2.threshold(cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY),
            0, 255, cv2.THRESH_BINARY)[1]

        # get the largest contour in which the entire stitching fit
        # and grab the coordinates and dimensions of the region
        c = self.maxContour(thresh)
        (x, y, w, h) = self.findCroppingBox(thresh, c)

        # extract the final image
        cropped = stitched[y:y + h, x:x + w]
        self._debug("cropping : x={} y={} w={} h={}".format(x, y, w, h))

        # return the cropped image
        return cropped

    def findCroppingBox(self, thresh, contour):
        # allocate memory for the mask which will contain the rectangular
        # bounding box of the stitched image region
        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # create two copies of the mask : one to serve as our actual
        # minimum rectangular region and another to serve as a counter
        # for how many pixels need to be removed from the minimum
        minRect = mask.copy()
        sub = mask.copy()

        # keep looping until there are no non-zero pixel left
        while cv2.countNonZero(sub) > 0:
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)

        # return the rectangular region that fit into the panorama
        return cv2.boundingRect(self.maxContour(minRect))

    def maxContour(self, image):
        # find the largest contour
        cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # return the coordinates largest contour
        return c

    def __repr__(self):
        return "Stitcher: version={} - crop={} - path={} - ext={} - verbose={}".format(
            self.cv, self.crop, self.savePath, self.saveExt, self.verbose)

    def _debug(self, msg, msgType="[INFO]"):
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))

    def _debug_pano(self, stitched, title="Stitching results"):
        if self.verbose:
            stitched = imutils.resize(stitched, width=min(stitched.shape[1], 1024))
            cv2.imshow(title, stitched)
            cv2.waitKey(0)
