# import necessary packages
import mxnet as mx

class MxAgeGenderNet:
    @staticmethod
    def build(classes):
        # data input
        data = mx.sym.Variable("data")

        # block 1: CONV => RELU => POOL layer set
        conv1_1 = mx.sym.Convolution(data=data, kernel=(7, 7),
            stride=(4, 4), num_filter=96)
        act1_1 = mx.sym.Activation(data=conv1_1, act_type="relu")
        bn1_1 = mx.sym.BatchNorm(data=act1_1)
        pool1 = mx.sym.Pooling(data=bn1_1, pool_type="max",
            kernel=(3, 3), stride=(2, 2))
        do1 = mx.sym.Dropout(data=pool1, p=0.25)

        # block 2: CONV => RELU => POOL layer set
        conv2_1 = mx.sym.Convolution(data=do1, kernel=(5, 5),
            stride=(4, 4), num_filter=256)
        act2_1 = mx.sym.Activation(data=conv2_1, act_type="relu")
        bn2_1 = mx.sym.BatchNorm(data=act2_1)
        pool2 = mx.sym.Pooling(data=bn2_1, pool_type="max",
            kernel=(3, 3), stride=(2, 2))
        do2 = mx.sym.Dropout(data=pool2, p=0.25)

        # block 3: CONV => RELU => POOL layer set
        conv3_1 = mx.sym.Convolution(data=do2, kernel=(5, 5),
            stride=(4, 4), num_filter=384)
        act3_1 = mx.sym.Activation(data=conv3_1, act_type="relu")
        bn3_1 = mx.sym.BatchNorm(data=act3_1)
        pool3 = mx.sym.Pooling(data=bn3_1, pool_type="max",
            kernel=(3, 3), stride=(2, 2))
        do3 = mx.sym.Dropout(data=pool3, p=0.25)

        # block 4: FC => RELU layer set
        flatten = mx.sym.Flatten(data=do3)
        fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=512)
        act4_1 = mx.sym.Activation(data=fc1, act_type="relu")
        bn4_1 = mx.sym.BatchNorm(data=act4_1)
        do4 = mx.sym.Dropout(data=bn4_1, p=0.5)

        # block 5: FC => RELU layer set
        fc2 = mx.sym.FullyConnected(data=do4, num_hidden=512)
        act5_1 = mx.sym.Activation(data=fc1, act_type="relu")
        bn5_1 = mx.sym.BatchNorm(data=act5_1)
        do5 = mx.sym.Dropout(data=bn4_1, p=0.5)

        # softmax classifier
        fc3 = mx.sym.FullyConnected(data=do5, num_hidden=classes)
        model = mx.sym.SoftmaxOutput(data=fc3, name="softmax")

        return model
