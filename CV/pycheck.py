# import necessary packages
from os.path import isfile, exists
from pathlib import Path
import argparse
import glob
import os

def get_arguments():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="input path")
    ap.add_argument("-e", "--excluded", help="excluded directory")
    ap.add_argument("-v", "--verbose", default=None)

    # return the arguments
    return vars(ap.parse_args())

def main():
    # grab the arguments
    args = get_arguments()

    # check to see if it is a file or a directory
    if isfile(args["input"]):
        # compile it
        source = open(args["input"], 'r').read() + "\n"
        compile(source, args["input"], 'exec')

        # if successfully compiled
        print("[SUCCESS]File: {}".format(args["input"]))

    # check to see if the directory exist
    elif exists(args["input"]):
        # create the glob filter and get the list of python files
        filter = "**/*.py"
        scripts = list(Path(args["input"]).glob(filter))

        # initialize the counter of processed files
        count = 0

        # loop over each python file
        for path in scripts:
            # check to see if we need to exlude this element by
            # looking if the *exact* excluded argument is in the
            # relative path to the file
            if args["excluded"] not in str(path).split(os.path.sep):
                # compile the source file
                source = path.open().read() + '\n'
                compile(source, str(path), 'exec')

                # if display is required
                if args["verbose"] is not None:
                    print("[INFO] processed {}".format(path))

                # increment the counter
                count += 1

        # print a message to the user
        print("[SUCCESS] successfully compiled {}"
            " files ({} excluded)".format(count, (len(scripts) - count)))

    # else the directory not exist, cancel the analyse
    else:
        # print a message to the user
        print("[FAIL] unknown path")

if __name__ == "__main__":
    main()
