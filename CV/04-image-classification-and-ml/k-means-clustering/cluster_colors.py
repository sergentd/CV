# import necessary packages
from sklearn.cluster import KMeans
import numpy as np
import imutils
import random
import cv2

# initialize the list of color choices
colors = [
    # shades of red, green, blue
    (138, 8, 8), (180, 4, 4), (223, 1, 1), (255, 0, 0), (250, 88, 88),
    (8, 138, 8), (4, 180, 4), (1, 223, 1), (0, 255, 0), (46, 254, 46),
    (8, 8, 138), (11, 11, 97), (4, 4, 180), (0, 0, 255), (46, 46, 254),
]

# initialize the canvas
canvas = np.ones((400, 600, 3), dtype="uint8") * 255

# loop over the canvas
for y in range (0, 400, 20):
    for x in range(0, 600, 20):
        # generate a random (x, y)-coordinates, radius and color
        (dx, dy) = np.random.randint(5, 10, size=(2,))
        r = np.random.randint(5, 8)
        color = random.choice(colors)[::-1]

        # draw the circle on the canvas
        cv2.circle(canvas, (x + dx, y + dy), r, color, -1)

# pad the border of the image
canvas = cv2.copyMakeBorder(canvas, 5, 5, 5, 5, cv2.BORDER_CONSTANT,
    value=(255, 255, 255))

# convert the canvas to grayscale, threshold it and detect contours
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# initialize the data matrix
data = []

# loop over the contours
for c in cnts:
    # construct the mask from the contour
    mask = np.zeros(canvas.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    features = cv2.mean(canvas, mask=mask)[:3]
    data.append(features)
