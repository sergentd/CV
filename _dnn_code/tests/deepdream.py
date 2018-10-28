# import necessary packages
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K
from scipy import ndimage
import numpy as np
import argparse
import cv2

def preprocess(p):
  # load the input image and convert it to a keras-compatible
  # format. Expand the dimensions so we can pass it through the
  # model, and finaly preprocess it for input to inception network
  image = load_img(p)
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  image = preprocess_input(image)

  # return the preprocessed image
  return image

def deprocess(p):
  # we are using channels last ordering
  image = image.reshape((image.shape[1], image.shape[2], 3))

  # undo the preprocessing
  image /= 255
  image += 0.5
  image *= 255
  image = np.clip(image, 0, 255).astype("uint8")

  # we have been processing images in RGB, so convert
  # to BGR for OpenCV
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  # return the deprocessed image
  return image

def fetchLossGrads(X):
  pass

def resize_image(image, size):
  # resize the image
  resized = np.copy(image)
  resized = ndimage.zoom(resized,
    (1, float(size[0]) / resized.shape[1],
    float(size[1]) / resized.shape[2], 1), order=1)

  # return the resized image
  return resized

def eval_loss_and_gradients(X):
  # fetch the loss and gradients given the input
  output = fetchLossGrads([X])
  (loss, G) = (output[0], output[1])

  # return tuple of loss and gradients
  return (loss, G)

def gradient_ascent(X, iters, alpha, maxLoss=-np.inf):
  # loop over our number of iterations
  for i in range(0, iters):
    # compute the loss and gradient
    (loss, G) = eval_loss_and_gradients(X)

    # if the loss is greater than the max loss, break from
    # the loop early to prevent strange effects
    if loss > maxLoss:
      break

    # take a step
    print("[INFO] Loss at {}: {}".format(i, loss))
    X += alpha*G

  # return the output of gradient ascent
  return X

# ***********************************************************************************

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
  help="path to the input image")
ap.add_argument("-o", "--output", required=True,
  help="path to the output image")
args = vars(ap.parse_args())

# define the dictionnary that include (1) the layers we are
# going to use for the dreams and (2) their respective weights
# (i.e the larger the weights, the more the layer is contributing)
LAYERS = {
  "mixed2" : 2.0,
  "mixed3" : 5.0
}

# define the number of octaves, octave scale, alpha (step for gradient
# ascent), number of iterations and max loss --tweaking theses values
# will produce different dreams
NUM_OCTAVE = 3
OCTAVE_SCALE = 1.4
ALPHA = 0.001
NUM_ITERS = 50
MAX_LOSS = 10.0

# indicate that keras *should not* be update the weights of any
# layer during the deep dream
K.set_learning_phase(0)

# load the pre-trained Inception model from disk, then grab
# reference variable to the input tensor of the model (which we'll
# then be using to perform our CNN hallucinations)
print("[INFO] loading InceptionV3 model...")
model = InceptionV3(weights="imagenet", include_top=False)
dream = model.input

# define the loss value, then build the dictionnary that maps the
# *name* of each layer inside the Inception to the actual *layer*
# object itself -- we'll need this mapping when building the loss
# of the dreams
loss = K.variable(0.0)
layerMap = {layer.name: layer for layer in model.layers}

# loop over the layers that will be utilized in the dream
for layerName in LAYERS:
  # grab the output of the layer we will use for dreaming, then
  # add the L2-norm on the features to the layer to the loss (we
  # use array slicing here to avoid border artifacts caused by
  # border pixels)
  x = layerMap[layerName].output
  coeff = LAYERS[layerName]
  scaling = K.prod(K.cast(K.shape(x), "float32"))
  loss += coeff * K.sum(K.square(x[:, 2:-2, 2:-2, :])) / scaling

# compute the gradients
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

# define a function that can retrieve the value of the loss
# and the gradients given an input image
outputs = [loss, grads]
fetchLossGrads = K.function([dream], outputs)

# load and preprocess the input image, then grab the (original) input
# height and width
image = preprocess(args["image"])
dims = image.shape[1:3]

# in order to perform deep dreaming we need to build multiple scales
# of the original input image (i.e set of images at lower and lower resolutions)
# -- this list stores the spatial dimensions that we will be resizing our image to
octaveDims = [dims]

# loop over the octaves (resolutions)
for i in range (1, NUM_OCTAVE):
  # compute the spatial dimensions (i.e width and height) for the
  # current octave, then update the dimensions list
  size = [int(d / (OCTAVE_SCALE ** i)) for d in dims]
  octaveDims.append(size)

# reverse the octave dimensions list order so the smallest
# is at first position
octaveDims = octaveDims[::-1]

# clone the original image and then create a resized input image
# that matches the smallest dimension
orig = np.copy(image)
shrunk = resize_image(image, octaveDims[0])

# loop over the octave dimensions from smallest to largest
for (o, size) in enumerate(octaveDims):
  # resize the image and then apply gradient ascent
  print("[INFO] starting octave {}...".format(o))
  image = resize_image(image, size)
  image = gradient_ascent(image, iters=NUM_ITERS, alpha=ALPHA,
    maxLoss=MAX_LOSS)

  # to compute the lost details we need two images :
  # (1) the shrunk image that has been upscaled to the current octave
  # (2) the original image that has been downscaled to the current octave
  upscaled = resize_image(shrunk, size)
  downscaled = resize_image(orig, size)

  # the lost detail is computed via a simple subtraction which we
  #immediately back in to the image we applied gradient ascent on
  lost = downscaled - upscaled
  image += lost

  # make the original image be the new shrunk image so we can
  #repeat the process
  shrunk = resize_image(orig, size)

# deprocess image, show it and write it to disk
image = deprocess(image)
cv2.imshow("DeepDrem", image)
cv2.imwrite(args["output"], image)
cv2.waitKey(0)
cv2.destroyAllWindows()
