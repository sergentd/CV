# import necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Reshape

class DCGAN:
  @staticmethod
  def build_generator(dim, depth, channels=1, inputDim=100,
    outputDim=512):
    
    # initialize the model along with the input shape to be
    # channels last
    model = Sequential()
    inputShape = (dim, dim, 3)
    chanDim = -1
    
    # first set of FC => RELU => BN layers
    model.add(Dense(input_dim=inputDim, units=outputDim))
    model.add(Activation("relu"))
    moel.add(BatchNormalization())
    
    # second set of FC => RELU => BN layers
    model.add(Dense(dim * dim * depth))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    
    # reshape the output of the previous layer set
    # upsample + transposed convolution + relu + bn
    model.add(Reshape(inputShape))
    model.add(Conv2DTranspose(32, (5, 5), strides=(2,2),
      padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    
    # upsample + TCONV => RELU => BN
    model.add(Conv2DTranspose(channels, (5,5), strides=(2,2),
      padding="same"))
    model.add(Activation("tanh"))
    
    # return the generator model
    return model
    
  @staticmethod
  def build_discriminator(width, height, depth, alpha=0.2):
    # initialize the inputshape to be channel last
    model = Sequential()
    inputShape = (height, width, depth)
    
    # first set of CONV => RELU layers
    model.add(Conv2D(32, (5,5), padding="same", strides=(2, 2)))
    model.add(LeakyReLU(alpha=alpha))
    
    # set of FC => RELU
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=alpha))
    
    # sigmoid layer, single value output
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    
    # return the discriminator model
    return model