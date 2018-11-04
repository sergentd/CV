# import necessary packages
import numpy as np
import os

class ImageNetHelper:
    def __init__(self, config):
        # store the configuration object
        self.config = config

        # build the label mappings and validation blacklist
        self.labelMappings = self.buildClassLabels()
        self.valBlacklist = self.buildBlacklist()
        # self.genTrainTxtFile()
        # self.genValTxtFile()
        # self.createAndMove()

    def buildClassLabels(self):
        # load the content of the file that maps the WordNet IDs
        # to integers, then initialize the label mappings dictionnary
        rows = open(self.config.WORD_IDS).read().strip().split("\n")
        labelMappings = {}

        # loop over the labels
        for row in rows:
            # split the row into the WordNet ID, label integer
            # and human readable label
            (wordID, label, hrLabel) = row.split(" ")

            # update the label mappings dictionnary using the word ID
            # as the key and the label as the value (-1 since MATLAB
            # is 1-indexed and python is 0-indexed)
            labelMappings[wordID] = int(label)-1

        return labelMappings

    def buildBlacklist(self):
        # load the list of blacklisted image IDs and convert them
        # to a set
        rows = open(self.config.VAL_BLACKLIST).read()
        rows = set(rows.strip().split("\n"))

        # return the blacklisted image IDs
        return rows

    def buildTrainingSet(self):
        # load the content of the training input file that lists
        # the partial image IDs and image number, then initialize
        # the list of image paths and class labels
        rows = open(self.config.TRAIN_LIST).read().strip().split("\n")
        paths = []
        labels = []

        # loop over the rows in the input training file
        for row in rows:
            # break the row into the partial path and image
            # number (number is sequential)
            (partialPath, imageNum) = row.strip().split(" ")

            # construct the full path to the training image, then
            # grab the word id from the path and use it to determine
            # the integer class label
            path = os.path.sep.join([self.config.IMAGES_PATH,
                "train", "{}".format(partialPath)])
            wordID = partialPath.split("/")[0]
            label = self.labelMappings[wordID]

            # update the respective paths and labels lists
            paths.append(path)
            labels.append(label)

        # return a tuple of image paths and associated integer class labels
        return (np.array(paths), np.array(labels))

    def buildValidationSet(self):
        # initialize the list of image paths and class labels
        paths = []
        labels = []

        # load the content of the file that lists the partial
        # validation image filename
        valFilenames = open(self.config.VAL_LIST).read()
        valFilenames = valFilenames.strip().split("\n")

        # load the content of the file that contains the actual
        # ground-truth integer class labels for the validation set
        valLabels = open(self.config.VAL_LABELS).read()
        valLabels = valLabels.strip().split("\n")

        # loop over the validation data
        for (row, label) in zip(valFilenames, valLabels):
            # break the row into partial path and image number
            (partialPath, imageNum) = row.strip().split(" ")

            # if the image number is in the blacklist set,
            # then ignore this validation image
            if imageNum in self.valBlacklist:
                continue

            # construct the full path to the validation image,
            # then update the respective paths and labels lists
            path = os.path.sep.join([self.config.IMAGES_PATH, "val",
                "{}".format(partialPath)])
            paths.append(path)
            labels.append(int(label) - 1)

        # return a tuple of images paths associated with their class
        # integer labels
        return (np.array(paths), np.array(labels))

    def genTrainTxtFile(self):
        # initialize the list of paths, the index and the text file
        # where the informations will be write
                # grab the list of paths for every image in the train dataset
        basepath = os.path.sep.join([self.config.IMAGES_PATH, "train"])
        paths = os.listdir(basepath)
        i = 1
        f = open(self.config.TRAIN_LIST, "w+")

        # loop over all directories in train directory
        for wnid in paths:
            if wnid.split(".")[-1] != "tar":
                # construct the fulle directory name
                # and grab the list of all images in the directory
                dirpath = os.path.sep.join([basepath, wnid])
                images = os.listdir(dirpath)

                # loop over all images in the directory
                for im in images:
                    # construct the row that will be inserted in the file
                    row = "{} {}\n".format(os.path.sep.join([wnid, im]), i)
                    f.write(row)
                    i += 1

        # close the file
        f.close()

    def genValTxtFile(self):
        # initialize the list of paths and output file
        paths = os.listdir(os.path.sep.join([self.config.IMAGES_PATH, "val"]))
        f = open(self.config.VAL_LIST, "w+")

        # loop over the images and write one row for each
        for (i, im) in enumerate(paths):
            row = "{} {}\n".format(os.path.basename(im), i+1)
            f.write(row)

        # close the file
        f.close()

    def createAndMove(self):
        # grab the list of paths for every image in the train dataset
        basepath = os.path.sep.join([self.config.IMAGES_PATH, "train"])
        paths = os.listdir(basepath)

        # loop over all images
        for p in paths:
            # skip all files with tar extensions
            if p.split(".")[-1] == "tar" or os.path.isdir(p):
                continue

            # create the directory if it doesnt exists
            dirname = p.split("_")[0]
            dirpath = os.path.sep.join([basepath, dirname])
            if not os.path.exists(dirpath):
                # print("[DIRPATH] : {}".format(dirpath))
                os.mkdir(dirpath)

            # the full image path (mover)
            oldpath = os.path.sep.join([basepath, p])
            newpath = os.path.sep.join([basepath, dirname, p])

            # print("[OLDPATH]: {}".format(oldpath))
            # print("[NEWPATH]: {}".format(newpath))
            # move the file into the right folder
            os.rename(oldpath, newpath)
