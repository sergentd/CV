# import necessary packages
from skimage.filters import threshold_local
from skimage import measure
import numpy as np
import cv2

# load the licence plate image from disk
plate = cv2.imread("licence_plate.png")

# extract the value component from the HSV color space and
# apply adaptative thresholding to reveal characters
V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 29, offset=15, method="gaussian")
thresh = (V < T).astype("uint8") * 255

# show the images
cv2.imshow("Licence plate",plate)
cv2.imshow("Thresh",thresh)

# perform connected components analysis on the thresholded images and
# init a mask to hold only the large components
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape,dtype="uint8")
print("[INFO] : found {} blobs".format(len(np.unique(labels))))

for (i, label) in enumerate(np.unique(labels)):
  # if this is the background, ignore it
  if label == 0:
    print("[INFO] : label = 0 (background)")
    continue
  # otherwise, construct the label mask to display only connected component
  # on the current label
  print("[INFO] : label = {} (foreground)".format(i))
  labelMask = np.zeros(thresh.shape, dtype="uint8")
  labelMask[labels == label] = 255
  numPixels = cv2.countNonZero(labelMask)

  # if the number of pixels in the component is sufficiently large,
  # add it to our mask of "large" blobs
  if numPixels > 300 and numPixels < 1500:
    mask = cv2.add(mask,labelMask)

  # show the label mask
  cv2.imshow("Label",labelMask)
  cv2.waitKey(0)

# show the largee components in the image
cv2.imshow("Large blobs", mask)
cv2.waitKey(0)
