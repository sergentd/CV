# import the necessary packages
from pypeline import Pause
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--network", required=True,
    help="name of network")
ap.add_argument("-d", "--dataset", required=True,
    help="name of dataset")
args = vars(ap.parse_args())

# define the paths to the training logs
logs = [
    (30, "log/gender_training_0.log"),	# lr=1e-3
    (40, "log/gender_training_30.log"),	# lr=1e-4
    (85, "log/gender_training_40.log"),	# lr=1e-5
]

# initialize the list of train rank-1 and rank-5 accuracies, along
# with the training loss
(trainRank1, trainLoss) = ([], [])

# initialize the list of validation rank-1 and rank-5 accuracies,
# along with the validation loss
(valRank1, trainOneOff, valOneOff, valLoss) = ([], [], [], [])

# loop over the training logs
for (i, (endEpoch, p)) in enumerate(logs):
	# load the contents of the log file, then initialize the batch
	# lists for the training and validation data
    rows = open(p).read().strip()
    (bTrainRank1, bTrainLoss) = ([], [])
    (bValRank1, bTrainOneOff, bValOneOff, bValLoss) = ([], [], [], [])

	# grab the set of training epochs
    epochs = set(re.findall(r'Epoch\[(\d+)\]', rows))
    epochs = sorted([int(e) for e in epochs])

	# loop over the epochs
    for e in epochs:
		# find all rank-1 accuracies, rank-5 accuracies, and loss
		# values, then take the final entry in the list for each
        s = r'Epoch\[' + str(e) + '\].*accuracy=([0]*\.?[0-9]+)'
        rank1 = re.findall(s, rows)[-2]
        s = r'Epoch\[' + str(e) + '\].*cross-entropy=([0-9]*\.?[0-9]+)'
        loss = re.findall(s, rows)[-2]

		# update the batch training lists
        bTrainRank1.append(float(rank1))
        bTrainLoss.append(float(loss))

	# extract the validation rank-1 and rank-5 accuracies for each
	# epoch, followed by the loss
    bValRank1 = re.findall(r'Validation-accuracy=(.*)', rows)
    bTrainOneOff = re.findall(r'Train-one-off=(.*)', rows)
    bValOneOff = re.findall(r'Test-one-off=(.*)', rows)
    bValLoss = re.findall(r'Validation-cross-entropy=(.*)', rows)

	# convert the validation rank-1, rank-5, and loss lists to floats
    bValRank1 = [float(x) for x in bValRank1]
    bTrainOneOff = [float(x) for x in bTrainOneOff]
    bValOneOff = [float(x) for x in bValOneOff]
    bValLoss = [float(x) for x in bValLoss]

	# check to see if we are examining a log file other than the
	# first one, and if so, use the number of the final epoch in
	# the log file as our slice index
    if i > 0 and endEpoch is not None:
        trainEnd = endEpoch - logs[i - 1][0]
        valEnd = endEpoch - logs[i - 1][0]

	# otherwise, this is the first epoch so no subtraction needs
	# to be done
    else:
        trainEnd = endEpoch
        valEnd = endEpoch

	# update the training lists
    trainRank1.extend(bTrainRank1[0:trainEnd])
    trainLoss.extend(bTrainLoss[0:trainEnd])

	# update the validation lists
    valRank1.extend(bValRank1[0:valEnd])
    valLoss.extend(bValLoss[0:valEnd])
    trainOneOff.extend(bTrainOneOff[0:valEnd])
    valOneOff.extend(bValOneOff[0:valEnd])

# plot the accuracies
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(trainRank1)), trainRank1,
    label="train_rank1")
plt.plot(np.arange(0, len(trainOneOff)), trainOneOff,
    label="train_one_off")
plt.plot(np.arange(0, len(valRank1)), valRank1,
    label="val_rank1")
plt.plot(np.arange(0, len(valOneOff)), valOneOff,
    label="val_one_off")
plt.title("{}: rank-1 and one-off accuracy on {}".format(
    args["network"], args["dataset"]))
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

# plot the losses
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(trainLoss)), trainLoss,
    label="train_loss")
plt.plot(np.arange(0, len(valLoss)), valLoss,
    label="val_loss")
plt.title("{}: cross-entropy loss on {}".format(args["network"],
    args["dataset"]))
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()
