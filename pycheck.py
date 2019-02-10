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
ap.add_argument("-v", "--verbose", default=None)
args = vars(ap.parse_args())

# check to see if it is a file or a directory
if isfile(args["input"]):
    # compile it
    source = open(args["input"], 'r').read() + "\n"
    compile(source, args["input"], 'exec')

    # if successfully compiled
    print("[SUCCESS]File: {}".format(args["input"]))

# check to see if the directory exist
elif exists(args["input"]):
    # create the glob filter
    filter = "**/*.py"
    scripts = list(Path(args["input"]).glob(filter))
    success = 0
    failure = 0

    # loop over each python file
    for path in scripts:
        # check to see if we need to exlude this element
        if args["excluded"] not in str(path).split(os.path.sep):
            # compile the source file
            try:
                source = path.open().read() + '\n'
                compile(source, str(path), 'exec')

                # if display is required
                if args["verbose"] is not None:
                    print("[INFO] processed {}".format(path))

                # increment the counter
                success += 1

            except Exception as e:
                print("[FAIL] {} - {}".format(path, e))
                failure += 1

    # conveniance variable for printing the results
    excluded = len(scripts) - success - failure

    # print a message to the user
    print("[RESULTS] successfully compiled {} files ({} excluded)"
        "\n[RESULTS] files with compilation failure : {}".format(success,
            excluded, failure))
# else the directory do not exist, cancel the analyse
else:
    # print a message to the user
    print("[FAIL] unknown path")
