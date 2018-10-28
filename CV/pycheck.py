# import necessary packages
from os.path import isfile, exists
from pathlib import Path
import argparse
import glob
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="input path")
ap.add_argument("-e", "--excluded", help="excluded directory")
args = vars(ap.parse_args())

# check to see if it is a file or a directory
if isfile(args["input"]):
    # compile it
    source = open(args["input"], 'r').read() + "\n"
    compile(source, args["input"], 'exec')
    print("File: {}".format(args["input"]))

# check to see if the directory exist
elif exists(args["input"]):
    # create the glob filter
    filter = "**/*.py"
    list = list(Path(args["input"]).glob(filter))

    # loop over each python file
    for f in list:
        if args["excluded"] not in str(f).split(os.path.sep):
            print("[INFO] processed {}".format(f))
            source = f.open().read() + '\n'
            compile(source, str(f), 'exec')

    # print a message to the user
    print("[SUCCESS] successfully compiled all files")

# else the directory do not exist, cancel the analyse
else:
    # print a message to the user
    print("[FAIL] unknown path")
