# import necessary packages
import mxnet as mx

class MxResNet:
    @staticmethod
    def residual_module(data, K, stride, red=False, bnEps=2e-5, bnMom=0.9):
        # the shortcut branch of the ResNet module should be
        # initialized as the input (identity) data
        shortcut = data

        # first block of the ResNet module are 1x1 conv
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=bnEps,
            momentum=bnMom)
        act1 = mx.sym.Activation(data=bn1, act_type="relu")
        conv1 = mx.sym.Convolution(data=act1, pad=(0, 0),
            kernel=(1, 1), stride=(1, 1), num_filter=int(K * 0.25),
            no_bias=True)

        # second block of ResNet module are 3x3 conv
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=bnEps,
            momentum=bnMom)
        act2 = mx.sym.Activation(data=bn2, act_type="relu")
        conv2 = mx.sym.Convolution(data=act2, pad=(0, 0),
            kernel=(3, 3), stride=stride, num_filter=int(K * 0.25),
            no_bias=True)

        # third block of ResNet is 1x1 conv
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=bnEps,
            momentum=bnMom)
        act3 = mx.sym.Activation(data=bn3, act_type="relu")
        conv3 = mx.sym.Convolution(data=act3, pad=(0, 0),
            kernel=(1, 1), stride=stride, num_filter=K, no_bias=True)

        # if we are to reduce the spatial size, apply a CONV layer
        # to the shortcut
        if red:
            shortcut = mx.sym.Convolution(data=act1, pad=(0, 0),
                kernel=(1, 1), stride=stride, num_filter=K, no_bias=True)

        # add together the shortcut and the final CONV
        add = conv3 + shortcut

        # return the addiction as the output of ResNet module
        return add

    @staticmethod
    def build(classes, stages, filters, bnEps=2e-5, bnMom=0.9):
        # data input
        data = mx.sym.Variable("data")
        
        pass
