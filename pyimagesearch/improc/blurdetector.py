# import necessary packages
import cv2

class BlurDetector:
    def __init__(self, threshold=100.0):
        # store the default threshold value
        self.threshold = threshold

    def score(image):
        # check to see if the image is already in grayscale
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # return the variance of the laplacian deviation for the image
        # the more variance, the less blurry
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def is_blurry(self, image, threshold=None):
        # initialize the threshold
        threshold = self.threshold if threshold is None else threshold

        # if the score is lower than the defined threshold, then
        # the image is tagged as not blurry
        if self.score(image) > threshold:
            return False

        # otherwise, the image is considered blurry
        else:
            return True

    def predict(image):
        # define the blurry status of the image and initialize the text
        blurry = self.is_blurry(image)
        pred = self.score(image)
        text = "Not blurry"

        # check to see if we need to update the text
        if blurry:
            text = "Blurry"

        # show the image with the text and the variance score
        cv2.putText(image, "{}: {:.2f}".format(text, pred), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
