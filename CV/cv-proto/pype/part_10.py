from __future__ import print_function
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-i", "--input", required=True)
args = vars(ap.parse_args())

f = open(args["input"], "r").read()
print(f, args["output"])
