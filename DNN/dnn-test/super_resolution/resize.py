# import necessary packages
from conf import config
from keras.models import load_model
from scipy import misc
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-b", "--baseline", required=True,
    help="path to baseline image")
ap.add_argument("-o", "--output", required=True,
    help="path to ouput super-resolution image")
args = vars(ap.parse_args())

# load the pre-trained model
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# load the input image, grab the dimensions of the input image
# and crop the image such that it tiles nicely
print("[INFO] generating images...")
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
w -= int(w % config.SCALE)
h -= int(h % config.SCALE)
image = image[0:h, 0:w]

# resize the input image using bicubic interpolation and then
# write the baseline image to disk
scaled = misc.imresize(image, config.SCALE / 1.0,
    interp="bicubic")
cv2.imwrite(args["baseline"], scaled)

# allocate memory for output image
output = np.zeros(scaled.shape)
(h, w) = output.shape[:2]

for y in range(0, h - config.INPUT_DIM + 1, config.LABEL_SIZE):
    for x in range(0, w - config.INPUT_DIM + 1, config.LABEL_SIZE):
        # crop the ROI from the scaled image
        crop = scaled[y:y + config.INPUT_DIM,
            x:x + config.INPUT_DIM]

        # make a prediction on the crop and store it in the output
        # image
        P = model.predict(np.expand_dims(crop, axis=0))
        P = P.reshape((config.LABEL_SIZE, config.LABEL_SIZE, 3))
        output[y + config.PAD:y + config.PAD + config.LABEL_SIZE,
               x + config.PAD:x + config.PAD + config.LABEL_SIZE] = P

# remove any black border in the output image caused by
# the padding and then clip values between [0, 255]
output = output[config.PAD:h - ((h % config.INPUT_DIM) + config.PAD),
                config.PAD:w + ((w % config.INPUT_DIM) + config.PAD)]
output = np.clip(output, 0, 255).astype("uint8")

# write the output image to disk
cv2.imwrite(args["output"], output)
