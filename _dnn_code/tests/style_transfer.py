# import necessary packages
from pyimagesearch.nn.conv import NeuralStyle
from keras.applications import VGG19
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
  help="path to the input image")
ap.add_argument("-s", "--style", required=True,
  help="path to the input style")
ap.add_argument("-o", "--output", required=True,
  help="path to the output directory")
args = vars(ap.parse_args())

# initialize the settings dictionnary
SETTINGS = {
  # initialize the path to the input image and style image
  # and the output directoy
  "input_path":  args["image"],
  "style_path":  args["style"],
  "output_path": args["output"],
  
  # define the CNN to be used for style transfer, along with
  # the set of content layer and style layers, respectively
  "net":VGG19,
  "content_layer": "block4_conv2",
  "style_layers": ["block1_conv1", "block2_conv1",
    "block3_conv1", "block4_conv1", "block5_conv1"],
    
  # store the content, style and total variation weights
  # respectively
  "content_weight": 1.0,
  "style_weight": 100.0,
  "tv_weight": 10.0,
  
  # number of iterations
  "iterations": 50
}

# perform neural style transfer
ns = NeuralStyle(SETTINGS)
ns.transfer()