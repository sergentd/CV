# import necessary packages
from keras.models import Sequential
from keras.layers.core import Dense

class MultilayerPerceptron:
    @staticmethod
    def build(inputDim, regress=False):
        # define de MLP network architecture
        model = Sequential()
        model.add(Dense(8, input_dim=inputDim, activation="relu"))
        model.add(Dense(4, activation="relu"))

        # check to see if the regression node should be added
        if regress:
            model.add(Dense(1, activation="linear"))

        # return the model
        return model
