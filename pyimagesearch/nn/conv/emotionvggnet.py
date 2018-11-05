# import necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolution import Conv2D
from keras.layers.convolution import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class EmotionVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model to be channels last
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # check to see of we are using channels first
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # block1 : CONV => ELU => CONV => ELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same",
            kernel_initializer="he_normal", input_shape=inputShape))
        model.add(ELU())
        mode.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), kernel_initializer="he_normal",
            padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # block2 : CONV => ELU => CONV => ELU => POOL
        model.add(Conv2D(64, (3, 3), padding="same",
            kernel_initializer="he_normal", input_shape=inputShape))
        model.add(ELU())
        mode.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal",
            padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # block3 : CONV => ELU => CONV => ELU => POOL
        model.add(Conv2D(128, (3, 3), padding="same",
            kernel_initializer="he_normal", input_shape=inputShape))
        model.add(ELU())
        mode.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), kernel_initializer="he_normal",
            padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # block4 : FC => ELU
        model.add(Flatten())
        model.add(Dense(64, kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # block5 : FC => ELU
        model.add(Dense(64, kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # block6: softmax classifier
        model.add(Dense(classes, kernel_initializer="he_normal"))
        model.add(Activation("softmax"))

        # return the constructed architecture
        return model
