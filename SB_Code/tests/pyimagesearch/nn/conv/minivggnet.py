# import necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class MiniVGGNet:
  @staticmethod
  def build(width, height, depth, classes):
    # initialize the model along with the input shape
	# channels last and the channels dimensions itself
	model = Sequential()
	inputShape = (height, width, depth)
	chanDim = -1
	
	# if we are using "channels first", update the input shape
	if K.image_data_format() == "channels_first":
	  inputShape = (depth, height, width)
	  chanDim = 1
	
	# first set of CONV => RELU => BN =>
	# CONV => RELU => BN => POOL = DO
	model.add(Conv2D(32, (3,3), padding="same",
	    input_shape=inputShape))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(32, (3,3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	
	# second set of CONV => RELU => BN =>
	# CONV => RELU => BN => POOL => DO
	model.add(Conv2D(64, (3,3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(64, (3,3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	
	# FC => RELU => BN => DO => FC => SOFTMAX
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Dropout(0.5))
	model.add(Dense(classes))
	model.add(Activation("softmax"))
	
	# return the constructed network architecture
	return model