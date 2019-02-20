# import necessary packages
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Input

class RegressNet:
    @staticmethod
    def build(width, height, depth, filters=(16, 32, 64), regress=False):
        # initialize the input shape and channel dimension
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using channel first, update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # define the model input
        inputs = Input(shape=inputShape)

        # loop over the number of filters
        for (i, f) in enumerate(filters):
            # if first layer of the network, set the input properly
            if i == 0:
                x = inputs

            # CONV => RELU => BN => POOL
            x = Conv2D(f, (3, 3), padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # flatten the volume, then FC => RELU => BN => DROPOUT
        x = Flatten()(x)
        x = Dense(16)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.5)(x)

        # apply another FC layer to match the number of nodes coming
        # out of the Multilayer Perceptron
        x = Dense(4)(x)
        x = Activation("relu")(x)

        # check to see if the regression node should be used
        if regress:
            x = Dense(1, activation="linear")(x)

        # construct the CNN
        model = Model(inputs, x)

        # return the CNN model
        return model
