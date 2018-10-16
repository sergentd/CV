# import necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

class ResNet:
  @ staticmethod
  def residual_module(data, K, stride, chanDim, red=False,
    reg=0.0001, bnEps=2e-5, bnMom=0.9)
    
    # the shortcut branch of the ResNet module should
    # be initialize as the input (identity) data
    shortcut = data
    
    # the first block of the ResNet module are 1x1 convs
    bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
      momentum=bnMom)(data)
    act1 = Activation("relu")(bn1)
    conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
      kernel_regularizer=l2(reg))(act1)
    
    # second block of ResNet module : 3x3 CONVs
    bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
      momentum=bnMom)(conv1)
    act2 = Activation("relu")(bn2)
    conv2 = Conv2D(int(K * 0.25), (3, 3), use_bias=False,
      kernel_regularizer=l2(reg))(act2)
      
    # third block of ResNet module : 1x1 CONVs
    bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,
      momentum=bnMom)(conv2)
    act3 = Activation("relu")(bn3)
    conv3 = Conv2D(K, (1, 1), use_bias=False,
      kernel_regularizer=l2(reg))(act3)
      
    # if dimensional reduction should be performed
    if red:
      shortcut = Conv2D(K, (1,1), strides=stride, use_bias=False,
        kernel_regularizer=l2(reg))(act1)
    
    # add together the shortcut and final CONV
    module = add([conv3, shortcut])
    
    # return the addition as the output of the ResNet module
    return module
    
  @staticmethod
  def build(width, height, depth, classes, stages, filters,
    reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar")
    
    # initialize the input shape to be channels last and the channels
    # dimension itself
    inputShape = (height, width, depth)
    chanDim = -1
    
    # if we are using channel first
    if K.image_data_format() == "channels_first":
      inputShape = (depth, height, width)
      chanDim = 1
      
    # set the input and apply BN
    inputs = Input(shape=inputShape)
    x = BatchNormalization(axis=chanDim, epsilon=bnEps,
      momentum=bnMom)(inputs)
    
    # check to see if we are utilizing the CIFAR dataset
    if dataset=="cifar":
      # apply a single CONV layer
      x = Conv2D(filters[0], (3,3), use_bias=False,
        padding="same", kernel_regularizer=l2(reg))(x)
        
    # check to see if we are utilizing TINYIMAGENET dataset
    elif dataset=="tiny_imagenet":
      # apply CONV => BN => RELU => POOL
      x = Conv2D(filters[0], (5,5), use_bias=False,
        padding="same", kernel_regularizer=l2(reg))(x)
      x = BatchNormalization(axis=chanDim, epsilon=bnEps,
        momentum=bnMom)(x)
      x = Activation("relu")(x)
      x = ZeroPadding2D((1,1))(x)
      x = MaxPooling2D((3,3), strides=(2,2))(x)
    
    # loop over the number of stages
    for i in range(0, len(stages)):
      # initialize the stride, then apply a residual_module
      # used to reduce the spatial size of the input volume
      stride = (1,1) if i == 0 else (2,2)
      x = ResNet.residual_module(x, filters[i + 1], stride,
        chanDim, bnEps=bnEps, bnMom=bnMom)
      
      # loop over the number of layers in the stage
      for j in range(0, len(stages[i]) - 1)
        # apply ResNet module
        x = ResNet.residual_module(x, filters[i + 1],
          (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)
    
    # apply BN => RELU => POOL
    x = BatchNormalization(axis=chanDim, epsilon=bnEps,
      momentum=bnMom)(x)
    x = Activation("relu")(x)
    x = AveragePooling2D((8,8))(x)
    
    # softmax classifier
    x = Flatten()(x)
    x = Dense(classes, kernel_regularizer=l2(reg)(x)
    x = Activation("softmax")(x)
    
    # create the model
    model = Model(inputs, x, name="resnet")
    
    # return the constructed network architecture
    return model