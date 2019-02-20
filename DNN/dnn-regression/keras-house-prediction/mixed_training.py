# import necessary packages
from pyimagesearch.utils import dataset
from pyimagesearch.nn.conv import RegressNet
from pyimagesearch.nn import MultilayerPerceptron
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import argparse
import locale
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
    help="path to input dataset of house images")
args = vars(ap.parse_args())

# construct the path to the input .txt file that contains informations
# on each house in the dataset and then load the dataset
print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = dataset.load_house_attributes(inputPath)

# load the house images and then scale the pixel intensities to the range [0, 1]
print("[INFO] loading house images...")
images = dataset.load_house_images(df, args["dataset"])
images = images / 255.0

# split the datas into training and testing set with 75% for training
print("[INFO] processing data...")
(trainAttrX, testAttrX, trainImagesX, testImagesX) = train_test_split(df,
    images, test_size=0.25, random_state=42)

# scale the houses prices according to the largest price for better convergence
maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

# process the house attributes data by performing min-max scaling
# on continuous features
(trainAttrX, testAttrX) = dataset.process_house_attributes(df, trainAttrX,
    testAttrX)

# create the MLP and the CNN
print("[INFO] building model...")
mlp = MultilayerPerceptron.build(trainAttrX.shape[1], regress=False)
cnn = RegressNet.build(64, 64, 3, regress=False)

# combine the outputs of the networks
combined = concatenate(mlp.output, cnn.output)

# final FC Head
x = Dense(4, activation="relu")(combined)
x = Dense(1, activation="linear")(x)

# final model which accepts multiple inputs and output the final single value
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# compile the model
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(
    [trainAttrX, trainImagesX], trainY,
    validation_data=([testAttrX, testImagesX]),
    epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict([testAttrX, testImagesX])

# compute the difference between the predicted house price and the actual house
# prices, then compute the percentage and the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics about the model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price:{}, std house price:{}".format(
    locale.currency(df["price"].mean(), grouping=True),
    locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean:{:.2f}%, std:{:.2f}%".format(mean, std))
