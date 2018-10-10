# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import necessary packages
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

# set a high recursion limit for Theano
