# import necessary packages
from pyimagesearch.nn.conv import LeNet
from keras.utils import plot_model

# initialize lenet and write the network architecture
model = LeNet.build(28,28,1,10)
plot_model(model, to_file="lenet.png", show_shapes=True)
