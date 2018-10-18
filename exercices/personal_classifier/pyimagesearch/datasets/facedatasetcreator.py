# import necessary packages
import imutils
import string
import random
import cv2
import os

class FaceDatasetCreator:
  def __init__(self, cascade, width=400, scale=1.1,minNei=5, minSize=(32,32)):
    # initialize the image size to process and detector parameters
    self.detector = cv2.CascadeClassifier(cascade)
    self.width = width
    self.minNei = minNei
    self.minSize = minSize
    self.scale = scale
    
  def detect(self, frame):
    # resize the frame for a faster faces detection
    frame = self.resize(frame)
  
    # detect faces in the grayscale frame
    rects = self.detector.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=self.scale, 
        minNeighbors=self.minNei, minSize=self.minSize)
    
    # draw the faces box on the image and return it
    return self.draw(frame, rects)
   
  def id_generator(self, size=6, chars=string.ascii_lowercase + string.digits):
    # generate a random id with lowercase ascii chars and digits
    return ''.join(random.choice(chars) for _ in range(size))
    
  def write(self, frame, output):
    # grab the output path
    p = os.path.sep.join([output, "{}.png".format(self.id_generator())])
    
    # write image to disk
    cv2.imwrite(p, orig)
    
  def resize(self, frame):
    # return the resized frame respecting aspect ratio
    return imutils.resize(frame, self.width)
    
  def draw(self, frame, rects):
    # draw boxes around face detections
    for (x, y, w, h) in rects:
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      
    # return the drawed frame
    return frame