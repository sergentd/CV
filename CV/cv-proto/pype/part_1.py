import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
args = vars(ap.parse_args())

f = open(args["input"], "w")
f.write(str(args["input"]))
f.close()
