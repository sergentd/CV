import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="text to display")
args = vars(ap.parse_args())

print(args["input"])
