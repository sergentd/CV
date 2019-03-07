# USAGE
# python neural_style_transfer_video.py
# during the process :
#   press 'q' to quit
#   press 'n' for next model
#   press 'a' for auto models rotation switch (on/off)
#   press 'l' to hide/display the legend
#   press 's' to save the picture

# import the necessary packages
from helpers import SimpleDatasetRenamer
from helpers.pypeline import Step
from helpers.views import MailView
from imutils.video import VideoStream
from conf import config as conf
from imutils import paths
import itertools
import argparse
import imutils
import time
import cv2
import os
import sh

# grab the paths to all neural style transfer models in our 'models'
# directory, provided all models end with the '.t7' file extension
modelPaths = paths.list_files(conf.MODELS, validExts=(".t7",))
modelPaths = sorted(list(modelPaths))

# initialize the filename creator
sdr = SimpleDatasetRenamer()

# auto-rotation parameters initialization
auto = conf.AUTO_ROTATION_MODE
cnt = 0
switch = {True:"on", False:"off"}
legend = conf.LIVE_LEGEND
email = conf.LIVE_EMAIL

# generate unique IDs for each of the model paths, then combine the
# two lists together
models = list(zip(range(0, len(modelPaths)), (modelPaths)))

# use the cycle function of itertools that can loop over all model
# paths, and then when the end is reached, restart again
modelIter = itertools.cycle(models)
(modelID, modelPath) = next(modelIter)

# load the neural style transfer model from disk
print("[INFO] loading style transfer model...")
net = cv2.dnn.readNetFromTorch(modelPath)

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
print("[INFO] {}. {}".format(modelID + 1, modelPath))

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
    frame = vs.read()

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
    copy = frame.copy()
    frame = imutils.resize(frame, width=conf.LIVE_INPUT_WIDTH)
    (h, w) = frame.shape[:2]

	# construct a blob from the frame, set the input, and then perform a
	# forward pass of the network
    blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h),
        (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()

	# reshape the output tensor, add back in the mean subtraction, and
	# then swap the channel ordering
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)

    # resize the output and draw the legend on it
    output = imutils.resize(output, width=max(conf.LIVE_OUTPUT_WIDTH,
        conf.LEGEND_WIDTH))

    # check to see if we want to draw the legend
    if legend:
        output[0:conf.L_LEGEND_HEIGHT, 0:conf.LEGEND_WIDTH] = 0
        cv2.putText(output, "Press 'n' for next model", (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(output, "Press 'a' to switch {} auto-rotation".format(
            switch[not auto]), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 0, 0), 2)
        cv2.putText(output, "Press 's' to save the picture", (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

	# show the original frame along with the output neural style
	# transfer
    cv2.imshow("Input", frame)
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

    # check to see if we are in auto mode
    if auto:
        # if we reached the right number of frames, load the next model and then
        # reset the frame counter
        if cnt >= conf.AUTO_MAX_FRAME:
            (modelID, modelPath) = next(modelIter)
            print("[INFO] {}. {}".format(modelID + 1, modelPath))
            net = cv2.dnn.readNetFromTorch(modelPath)
            cnt = 0

        # increment the number of frames on which we applied the same model
        cnt += 1

    # if the key 'a' (for "auto") is pressed, switch the auto-rotation mode
    if key == ord("a"):
        cnt = 0
        auto = True if auto == False else False
        print("[INFO] switching auto mode to: {}".format(switch[auto]))

    # if the key 'l' (for "legend") is pressed, switch the display mode
    if key == ord("l"):
        legend = True if legend == False else False
        print("[INFO] switching legend mode to: {}".format(switch[legend]))

    # if the key 'e' (for "email") is pressed, switch the email enabled state
    if key == ord("e"):
        email = True if email == False else False
        print("[INFO] switching email capture mode to: {}".format(switch[email]))

	# if the `n` key is pressed (for "next"), load the next neural
	# style transfer model
    if key == ord("n"):
		# grab the next nueral style transfer model model and load it
        (modelID, modelPath) = next(modelIter)
        print("[INFO] {}. {}".format(modelID + 1, modelPath))
        net = cv2.dnn.readNetFromTorch(modelPath)
        cnt = 0

    # if the 's' key is pressed (for "save"), we want to save
    # the image to disk
    if key == ord("s"):
        # check to see if the destination directories exists
        # and if not, create it
        if not os.path.exists(conf.ORIG_DIR):
            os.makedirs(conf.ORIG_DIR)
        if not os.path.exists(conf.DEEP_DIR):
            os.makedirs(conf.DEEP_DIR)

        # generate the paths to images
        filename = str(sdr.id_generator(sequential=False)) + ".jpg"
        origname = os.path.sep.join([conf.ORIG_DIR, filename])
        deepname = os.path.sep.join([conf.DEEP_DIR, filename])

        # copy the frame to disk and then create a pypeline Step which
        # will launch the neural_style_transfer.py script as a shell
        # command line so we give the required arguments for the script
        cv2.imwrite(origname, copy)
        transfer = Step("neural_style_transfer.py",
            "apply neural style transfer to frame",
            [["model", modelPath],
            ["image", origname],
            ["width", conf.NEURAL_INPUT_WIDTH],
            ["output-width", conf.NEURAL_OUTPUT_WIDTH],
            ["output-path", deepname]])
        transfer.execute()

        # initialize the output message
        msg = "[INFO] frame saved as {}".format(filename)

        # show a view to enter email information
        if email:
            interface = MailView()
            interface.create()
            interface.show()

            # save the mapping between file <-> mail to disk
            f = open("mails.csv", "a+")
            f.write("{} {}\n".format(filename, interface.mail))
            f.close()
            msg += " for {}".format(interface.mail)

        # print the output message about the filename and possibly the
        # mapping with mail
        print(msg)

	# if the `q` (for "quit") key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
