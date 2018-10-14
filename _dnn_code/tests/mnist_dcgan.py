# import the necessary packages
from pyimagesearch.nn.conv import DCGAN
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.datasets import mnist
from sklearn.utils import shuffle
from imutils import build_montages
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
  help="path to output directory")
ap.add_argument("-e", "--epochs", type=int, default=50,
  help="# epochs to train for")
ap.add_argument("-b", "--batch-size", type=int, default=128,
  help="batch size for training")
args = vars(ap.parse_args())

# store the epochs and batch size in convenience variables
NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]