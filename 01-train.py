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
from lasagne.init import GlorotUniform, Constant

from os import listdir
from os.path import isfile, join
import numpy as np


def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))


def buildNetwork(input_var=None):
    net = {}

    # The input shape is (freq, time) -> (130,300)
    net['input'] = InputLayer((None, 129, 300), input_var=input_var, W=GlorotUniform('relu'), b=Constant(0.0))
    print "input: {}".format(net['input'].output_shape[1:])
    # conv1
    net['conv1'] = Conv1DLayer(net['input'], num_filters=256, filter_size=4, W=GlorotUniform('relu'), b=Constant(0.0))
    print "conv1: {}".format(net['conv1'].output_shape[1:])
    # pool1
    net['pool1'] = Pool1DLayer(net['conv1'], pool_size=4)
    print "pool1: {}".format(net['pool1'].output_shape[1:])

    # conv2
    net['conv2'] = Conv1DLayer(net['pool1'], num_filters=256, filter_size=4, W=GlorotUniform('relu'), b=Constant(0.0))
    print "conv2: {}".format(net['conv2'].output_shape[1:])
    # pool2
    net['pool2'] = Pool1DLayer(net['conv2'], pool_size=2)
    print "pool2: {}".format(net['pool2'].output_shape[1:])

    # conv3
    net['conv3'] = Conv1DLayer(net['pool2'], num_filters=512, filter_size=4, W=GlorotUniform('relu'), b=Constant(0.0))
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
                            nonlinearity=lasagne.nonlinearities.rectify, W=GlorotUniform('relu'), b=Constant(0.0))
    print "fc6: {}".format(net['fc6'].output_shape[1:])

    # fc7
    net['fc7'] = DenseLayer(net['fc6'], num_units=2048,
                            nonlinearity=lasagne.nonlinearities.rectify, W=GlorotUniform('relu'), b=Constant(0.0))
    print "fc7: {}".format(net['fc7'].output_shape[1:])

    # output
    net['output'] = DenseLayer(net['fc7'], num_units=100,
                               nonlinearity=lasagne.nonlinearities.sigmoid, W=GlorotUniform('relu'), b=Constant(0.0))
    print "output: {}".format(net['output'].output_shape[1:])

    return net


def categorical_crossentropy_logdomain(log_predictions, targets):
    return -T.sum(targets * log_predictions, axis=1)


def batch_gen(X, y, N):
    while True:
        idx = np.random.choice(len(y), N)
        yield X[idx].astype('float32'), y[idx].astype('int32')



if __name__ == "__main__":

    # Load data

    # with open( 'validationData.pickle', 'rb') as f:
    # validationData = pickle.load( f )

    # with open( 'testData.pickle', 'rb') as f:
    # testData = pickle.load( f )

    # valData = validationData[0:2000]

    # Create network

    # X_train = np.random.randn(1000,12, 300)
    # y_train = np.random.random_integers(0, 256, 1000)

    x = list()
    y = list()

    inputImage = T.tensor3()
    output = T.imatrix()


    net = buildNetwork(inputImage)

    prediction = lasagne.layers.get_output(net['output'])
    test_prediction = lasagne.layers.get_output(net['output'], deterministic=True)
    loss = lasagne.objectives.squared_error(prediction, output)
    # loss = lasagne.objectives.categorical_crossentropy(prediction, output)
    loss = loss.mean()

    init_learningrate = 0.01
    momentum = 0.0  # start momentum at 0.0
    max_momentum = 0.9
    min_learningrate = 0.00001
    lr = theano.shared(np.array(init_learningrate, dtype=theano.config.floatX))
    mm = theano.shared(np.array(momentum, dtype=theano.config.floatX))

    params = lasagne.layers.get_all_params(net['output'], trainable=True)

    updates_sgd = lasagne.updates.sgd(loss, params, learning_rate=lr)
    updates = lasagne.updates.apply_momentum(updates_sgd, params, momentum=mm)

    train_fn = theano.function([inputImage, output], loss, updates=updates, allow_input_downcast=True)

    # val_fn = theano.function([inputImage, output], loss)

    predict_fn = theano.function([inputImage], test_prediction)

    BATCH_SIZE = 128
    numEpochs = 50

    # batchIn = np.zeros((batchSize, 12, 300), theano.config.floatX)
    # batchOut = np.zeros((batchSize, 256), theano.config.floatX)
    print "loading data..."
    for f in [f for f in listdir('/home/manel/Downloads') if '.npz' in f]:
        npzfile = np.load('/home/manel/Downloads/'+ f)
        x.extend(npzfile['x'])
        y.extend(npzfile['y'])

    #x_train = np.random.randint(9,255,size=(190, 129, 300))
    #y_train = np.random.randint(0,2,size=(190,100))

    x_train = np.array(x)
    y_train = np.array(y)

    x_train = x_train.transpose(0, 2, 1)

    print x_train.shape
    print y_train.shape

    train_batches = batch_gen(x_train, y_train, BATCH_SIZE)
    N_BATCHES = len(x_train) // BATCH_SIZE

    # print 'Loading training data...'
    # with open(pathToImagesPickle, 'rb') as f:
    #    trainData = pickle.load(f)
    # print '-->done!'

    for currEpoch in tqdm(range(numEpochs),ncols=20):

        # random.shuffle(trainData)
        # random.shuffle( valData )

        train_err = 0.
        # val_err = 0.

        for _ in range(N_BATCHES):
            X, y = next(train_batches)
            train_err += train_fn(X, y)
        '''
        for currChunk in chunks(trainData, batchSize):

            if len(currChunk) != batchSize:
                continue

            for k in range(batchSize):
                batchIn[k, ...] = (currChunk[k].astype(theano.config.floatX) / 255.
                batchOut[k, ...] = (currChunk[k].output.data.astype(theano.config.floatX))

            train_err += train_fn(batchIn, batchOut)

        for currChunk in chunks(valData, batchSize):

            if len(currChunk) != batchSize:
                continue

            for k in range( batchSize ):
                batchIn[k,...] = (currChunk[k].image.data.astype(theano.config.floatX).transpose(2,0,1)-imageMean)/255.
                batchOut[k,...] = (currChunk[k].saliency.data.astype(theano.config.floatX))/255.
            val_err += val_fn( batchIn, batchOut )
        '''
        print 'Epoch:', currEpoch, ' ->', train_err  # (train_err, val_err )

        if currEpoch % 10 == 0:
            np.savez("modelWights{:04d}.npz".format(currEpoch), *lasagne.layers.get_all_param_values(net['output']))
