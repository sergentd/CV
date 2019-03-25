import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
    help="text to display")
args = vars(ap.parse_args())

print(args["input"])
os.mkdir("output/"+args["input"])
