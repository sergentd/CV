# import necessary packages
from imutils import paths
import progressbar
import string
import random
import cv2
import os

# set the array of accepted false values and extensions
FALSE_VALUES = ["false", "no", "f", "n", "0", "-1"]
VALID_EXT = ["jpg", "jpeg", "gif", "tiff", "png", "bmp"]

class SimpleDatasetRenamer:
    def __init__(self, directory=".", prefix=None, suffix=None, keep_idx=False,
        sequential=True, length=6, ext=None, index=0):
        # store the following parameters : prefix to filename, suffix to filename,
        # keep old filename boolean, source directory, sequential, length of
        # filename, file extension (dataformat), current index
        self.prefix = prefix
        self.suffix = suffix
        self.keep_idx = keep_idx
        self.directory = directory
        self.sequential = sequential
        self.length = length
        self.ext = ext
        self.index = index

    def rename(self, directory=None):
        # grab the reference to the list of images
        if directory is not None:
            imagePaths = sorted(list(paths.list_images(directory)))
        else:
            imagePaths = sorted(list(paths.list_images(self.directory)))

        # initialize the progressbar (feedback to user on the task progress)
        widgets = ["Renaming Dataset: ", progressbar.Percentage(), " ",
            progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(imagePaths),
            widgets=widgets).start()

        # loop over each image, then rename it and save it in its
        # original data format
        for (i, path) in enumerate(imagePaths):
            # generate a filename
            filename = self.gen_filename(path)

            # check to see if we successfuly get a new filename
            if filename is not None:
            # rename the image
                os.rename(path, filename)

            # update the progressbar (feedback to user)
            pbar.update(i)

        # close the progressbar
        pbar.finish()

    def gen_filename(self, path):
        # initialize the filename
        filename = None

        # grab the data format to encode the file with
        # the same after renaming
        ext = str(self.ext).lower() if str(self.ext).lower() in VALID_EXT \
            else (path.split(os.path.sep)[-1]).split(".")[1]


        # check to see if a valid extension is found, else set it to 'jpg'
        ext = ext.lower() if ext.lower() in VALID_EXT else "jpg"
        # see if we are using prefix and/or suffix
        prefix = str(self.prefix) if self.prefix is not None else ""
        suffix = str(self.suffix) if self.suffix is not None else ""

        # set the default idx to the filename without extension
        idx = os.path.basename(path).split(".")[0]

        # check to see if we need to generate a unique ID
        if not self.keep_idx:
        # loop to find a unique filename
            while True:
                # loop until we have a unique id and no conflict with existing files
                # generate a tentative of unique ID
                idx = id_generator(sequential=self.sequential,
                    length=self.length, index=self.index)

                # increment the current index number (preventing infinite loop)
                self.index += 1

                # generate the filename : {dir}{sep}[{prefix}]{idx}[{suffix}].{df}
                filename = os.path.sep.join([os.path.dirname(path),
                    "{}{}{}.{}".format(prefix, idx, suffix, ext)])

                # allow for a unique filename only
                if (not os.path.isfile(filename)) or \
                    (self.index >= 10**(self.length + 1)):
                    break

        # generate the filename : {dir}{sep}[{prefix}]{idx}[{suffix}].{df}
        filename = os.path.sep.join([os.path.dirname(path),
            "{}{}{}.{}".format(prefix, idx, suffix, ext)])

        # if we keep the idx and the filename wasn't free at first try,
        # we will not be able to generate a unique ID so we return a None
        # filename (so we don't erase the existing file)
        if os.path.isfile(filename):
            print("Could not save {}: existing file in target directory".format(
                filename))
            return None

        # return the filename
        return filename

    @staticmethod
    def id_generator(sequential=True, length=6, index=0,
        chars=string.ascii_lowercase + string.digits):
        # create a random id with lowercase letters and digits
        if str(sequential).lower() in FALSE_VALUES:
            return ''.join(random.choice(chars) for _ in range(length))

        # otherwise create a sequential an ID based on the index
        else:
            return str(index).zfill(length)
