import os
import numpy as np
import sys
import cPickle as pickle
import glob
import random
from tqdm import tqdm

import theano
import theano.tensor as T
import lasagne

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import Conv1DLayer, DropoutLayer, Pool1DLayer, GlobalPoolLayer
from lasagne.layers import SliceLayer, concat, DenseLayer

from lasagne.nonlinearities import softmax, rectify
from lasagne.utils import floatX

from os import listdir
from os.path import isfile, join
import numpy as np


def buildNetwork(input_var=None):
    net = {}

    net['input'] = InputLayer((None, 12, 300), input_var=input_var)
    print "input: {}".format(net['input'].output_shape[1:])
    # conv1
    net['conv1'] = Conv1DLayer(net['input'], num_filters=256, filter_size=4, nonlinearity=rectify)
    print "conv1: {}".format(net['conv1'].output_shape[1:])
    # pool1
    net['pool1'] = Pool1DLayer(net['conv1'], pool_size=4)
    print "pool1: {}".format(net['pool1'].output_shape[1:])

    # conv2
    net['conv2'] = Conv1DLayer(net['conv1'], num_filters=256, filter_size=4, nonlinearity=rectify)
    print "conv2: {}".format(net['conv2'].output_shape[1:])
    # pool2
    net['pool2'] = Pool1DLayer(net['conv2'], pool_size=1)
    print "pool2: {}".format(net['pool2'].output_shape[1:])

    # conv3
    net['conv3'] = Conv1DLayer(net['conv2'], num_filters=512, filter_size=4)
    print "conv3: {}".format(net['conv3'].output_shape[1:])

    # global pool
    net['pool3_1'] = GlobalPoolLayer(net['conv3'], pool_function=T.mean)
    print "pool3_1: {}".format(net['pool3_1'].output_shape[1:])

    net['pool3_2'] = GlobalPoolLayer(net['conv3'], pool_function=T.max)
    print "pool3_2: {}".format(net['pool3_2'].output_shape[1:])

    net['pool3'] = concat((net['pool3_1'], net['pool3_2']), axis=1)
    print "pool3: {}".format(net['pool3'].output_shape[1:])

    # fc6
    net['fc6'] = DenseLayer(net['pool3'], num_units=2048,
                            nonlinearity=lasagne.nonlinearities.rectify)
    print "fc6: {}".format(net['fc6'].output_shape[1:])
    # fc7
    net['fc7'] = DenseLayer(net['fc6'], num_units=2048,
                            nonlinearity=lasagne.nonlinearities.rectify)
    print "fc7: {}".format(net['fc7'].output_shape[1:])
    # output
    net['output'] = DenseLayer(net['fc7'], num_units=256,
                               nonlinearity=lasagne.nonlinearities.sigmoid)
    print "output: {}".format(net['output'].output_shape[1:])

    return net


if __name__ == "__main__":
    inputImage = T.tensor3()
    output = T.imatrix()

    net = buildNetwork(inputImage)

    epochToLoad = 10

    with np.load("modelWights{:04d}.npz".format(epochToLoad)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net['output'], param_values)

    test_prediction = lasagne.layers.get_output(net['output'], deterministic=True)
    predict_fn = theano.function([inputImage], test_prediction)

    x = list()
    y = list()

    npzfile = np.load('./data/obj_00.npz')
    x.extend(npzfile['x'])
    y.extend(npzfile['y'])

    x_test = np.array(x[:20], theano.config.floatX)
    y_test = np.array(y[:20])

    x_test = x_test.transpose(0, 2, 1)

    result = np.squeeze(predict_fn(x_test))

    print result.shape

    np.savez("prediction.npz", *result)
    np.savez("ground_truth.npz", *y_test)
