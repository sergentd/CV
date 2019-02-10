# import the necessary packages
import cv2
import datetime

class ContoursHelper:
    def __init__(self, method="left-to-right", verbose=True):
        # store the sorting method, the reverse parameter, the index flag
        # and the verbose paramater
        self.method = method
        self.verbose = verbose
        self.reverse = method in ["right-to-left", "bottom-to-top"]
        self.i = 1 if method in ["top-to-bottom", "bottom-to-top"] else 0

    def _debug(self, msg, msgType="[INFO]"):
        print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))

    def sort_contours(cnts):
    	# construct the list of bounding boxes and sort them from top to
    	# bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
            key=lambda b:b[1][self.i], reverse=self.reverse))

        if self.verbose:
            self._debug("sorted with parameters: reverse={}, i={}".format(
                self.reverse, self.i))

    	# return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    def draw_circular_contour(image, c, i):
    	# compute the center of the contour area and draw a circle
    	# representing the center
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    	# draw the countour number on the image
        cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 255, 255), 2)

    	# return the image with the contour number drawn on it
        return image
