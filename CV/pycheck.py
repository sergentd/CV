from os import listdir
from os.path import isfile, join
import argparse
import glob
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="input path")
args = vars(ap.parse_args())

filter = "{}/**/*.{}".format(args["input"], "py")

for f in glob.iglob(filter, recursive=True):
    if (f.split(os.path.sep)[-1]).split(".")[-1] == "py":
        print("File :{}".format(f))
        source = open(f, 'r').read() + '\n'
        compile(source, f, 'exec')
