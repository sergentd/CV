# import necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.regularizers import l2
from keras import backend as K

class AlexNet:
  @staticmethod
  def build(width, height, depth, classes, reg=0.0002):
    # initialize the model along with the input shape
    # to be channels last and the channels dimension itself
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1
  
    # if we are using channel first, update the input shape
    if K.image_data_format() == "channels_first":
      inputShape = (dept, height, width)
      chanDim = 1
  
    # Block #1: CONV => RELU => BN => POOL => DO
    model.add(Conv2D(96, (11,11), strides=(4,4),
      input_shape=inputShape, padding="same",
      kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Dropout(0.25))
  
    # Block #2: CONV => RELU => BN => POOL => DO
    model.add(Conv2D(256, (5,5), padding="same",
      kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Dropout(0.25))
  
    # Block #3: [CONV => RELU => BN] * 3 => POOL => DO
    model.add(Conv2D(384, (3,3), padding="same",
      kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(384, (3,3), padding="same",
      kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(256, (3,3), padding="same",
      kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Dropout(0.25))
  
    # Block #4: first set of FC => RELU => BN => DO
    model.add(Flatten())
    model.add(Dense(4096, kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.5))
  
    # Block #5: second set of FC => RELU => BN => DO
    model.add(Dense(4096, kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.5))
  
    # softmax
    model.add(classes, kernel_regularizer=l2(reg))
    model.add(Activation("softmax"))
	
	return model