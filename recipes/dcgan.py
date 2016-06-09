from __future__ import print_function
import sys
import os
from time import time
from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T

from lasagne import layers
from lasagne.layers import Deconv2DLayer
from lasagne.nonlinearities import tanh, rectify, LeakyRectify, sigmoid
from lasagne.regularization import l2, regularize_network_params
from lasagne import init, updates


from skimage.io import imsave

from utils import (floatX,
                   ConvCondConcat,
                   DenseCondConcat,
                   to_categorical, mkdir_path,
                   tile_raster_images,
                   iterate_minibatches)


leaky_rectify = LeakyRectify(0.2)


def build_dcgan(z_dim=100, w=28, h=28, c=1, nb_outputs=10):
    x_in = layers.InputLayer((None, c, w, h), name="input")
    y_in = layers.InputLayer((None, nb_outputs), name="label")
    z_in = layers.InputLayer((None, z_dim), name="z")

    # discrimimator

    X = ConvCondConcat((x_in, y_in))

    X = layers.Conv2DLayer(
        X,
        num_filters=64,
        filter_size=(5, 5),
        stride=2,
        nonlinearity=leaky_rectify,
        W=init.Normal(0.02)
    )
    X = ConvCondConcat((X, y_in))

    X = layers.Conv2DLayer(
        X,
        num_filters=128,
        filter_size=(5, 5),
        stride=2,
        nonlinearity=leaky_rectify,
        W=init.Normal(0.02)
    )
    X = layers.batch_norm(X)
    X = ConvCondConcat((X, y_in))
    X = layers.DenseLayer(
        X,
        128,
        nonlinearity=leaky_rectify,
        W=init.Normal(0.02)

    )
    X = layers.batch_norm(X)
    X = DenseCondConcat((X, y_in))
    X = layers.DenseLayer(
        X,
        1,
        nonlinearity=sigmoid,
        W=init.Normal(0.02)
    )
    out_discr = X

    # generator

    Z = DenseCondConcat((z_in, y_in))
    Z = layers.DenseLayer(
        Z,
        1024,
        nonlinearity=rectify,
        W=init.Normal(0.02)

    )
    Z = layers.batch_norm(Z)
    Z = DenseCondConcat((Z, y_in))
    Z = layers.DenseLayer(
        Z,
        128*7*7,
        nonlinearity=rectify,
        W=init.Normal(0.02)

    )

    Z = layers.batch_norm(Z)

    Z = layers.ReshapeLayer(
        Z,
        ([0], 128, 7, 7)
    )
    Z = ConvCondConcat((Z, y_in))
    Z = Deconv2DLayer(
        Z,
        num_filters=64,
        filter_size=(4, 4),
        stride=2,
        crop=1,
        nonlinearity=rectify,
        W=init.Normal(0.02),
    )
    Z = layers.batch_norm(Z)
    Z = ConvCondConcat((Z, y_in))
    Z = Deconv2DLayer(
        Z,
        num_filters=c,
        filter_size=(4, 4),
        stride=2,
        crop=1,
        nonlinearity=sigmoid,
        W=init.Normal(0.02),
    )
    out_gen = Z
    return x_in, y_in, z_in, out_gen, out_discr


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
    X_train = floatX(X_train)
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    X_test = floatX(X_train)
    X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)

    return X_train, y_train, X_test, y_test


def main(out_folder='out/'):
    lr_initial = 0.0002
    nb_epochs = 1000
    z_dim = 100
    batch_size = 128
    nb_outputs = 10
    c, w, h = 1, 28, 28
    l2_coef = 2.5e-5
    rng = np.random
    lr = theano.shared(floatX(np.array(lr_initial)))

    # load data
    X_train, y_train, X_test, y_test = load_dataset()
    # save a sample of real data in a file
    mkdir_path(out_folder)
    xdisp = X_train[0:400]
    img = tile_raster_images(xdisp, tile_spacing=(1, 1))
    filename = os.path.join(out_folder, 'real_data.png')
    imsave(filename, img)

    # build and compile net
    Y = T.matrix()
    X_real = T.tensor4()
    Z = T.matrix()

    x_in, y_in, z_in, out_gen, out_discr = build_dcgan(
        z_dim=z_dim, w=w, h=h, c=c, nb_outputs=nb_outputs)

    inputs = {z_in: Z, y_in: Y}
    X_gen = layers.get_output(out_gen, inputs)

    inputs = {x_in: X_real, y_in: Y}
    p_real = layers.get_output(out_discr, inputs)

    inputs = {x_in: X_gen, y_in: Y}
    p_gen = layers.get_output(out_discr, inputs)

    # cost of discr : predict 0 for gen and 1 for real

    d_cost_real = T.nnet.binary_crossentropy(
        p_real,
        T.ones(p_real.shape)).mean()
    d_cost_gen = T.nnet.binary_crossentropy(
        p_gen,
        T.zeros(p_gen.shape)).mean()

    # cost of gen : make the discr predict 1 for gen
    g_cost_d = T.nnet.binary_crossentropy(p_gen, T.ones(p_gen.shape)).mean()

    d_cost = d_cost_real + d_cost_gen
    d_cost_reg = l2_coef * regularize_network_params(out_discr, l2)

    g_cost = g_cost_d
    g_cost_reg = l2_coef * regularize_network_params(out_gen, l2)

    cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

    discrim_params = layers.get_all_params(out_discr, trainable=True)
    gen_params = layers.get_all_params(out_gen, trainable=True)

    d_updates = updates.adam(d_cost + d_cost_reg, discrim_params,
                             learning_rate=lr, beta1=0.5)
    g_updates = updates.adam(g_cost + g_cost_reg, gen_params,
                             learning_rate=lr, beta1=0.5)

    all_updates = d_updates.copy()
    all_updates.update(g_updates)

    inputs = [X_real, Z, Y]

    train_g = theano.function(inputs, cost, updates=g_updates)
    train_d = theano.function(inputs, cost, updates=d_updates)

    inputs = [Z, Y]
    gen = theano.function(inputs, X_gen)


    # prepare training
    history = []
    n_updates = 0
    update_generator = False

    # save the model at initialization
    model_filename = os.path.join(out_folder, 'model.pkl')
    save_model(out_gen, out_discr, history, model_filename)

    # Train
    for epoch in range(1, nb_epochs + 1):
        total_g_loss = 0
        total_d_loss = 0
        nb_g_updates = 0
        nb_d_updates = 0
        t = time()
        for mb_X, mb_y in iterate_minibatches(X_train, y_train, batch_size):
            mb_Z = floatX(rng.uniform(-1., 1., size=(len(mb_X), z_dim)))
            mb_y = floatX(to_categorical(mb_y))
            inputs = [mb_X, mb_Z, mb_y]
            if update_generator:
                total_g_loss += (train_g(*inputs))[0]
                nb_g_updates += 1
                update_generator = False
            else:
                total_d_loss += (train_d(*inputs))[1]
                nb_d_updates += 1
                update_generator = True
            n_updates += 1
        print(nb_g_updates, nb_d_updates)
        stats = OrderedDict()
        stats['epoch'] = epoch
        stats['g_loss'] = total_g_loss / nb_g_updates
        stats['d_loss'] = total_d_loss / nb_d_updates
        stats['train_time(sec)'] = time() - t
        history.append(stats)

        for k, v in stats.items():
            print('{}:{:<20} '.format(k, v), end='')
        print('')

        if epoch % 5 == 0:
            nb_samples_per_output = 40
            nb_samples = nb_outputs * nb_samples_per_output
            sample_Z = floatX(rng.uniform(-1., 1., size=(nb_samples, z_dim)))
            sample_Y = np.repeat(np.arange(0, nb_outputs),
                                 nb_samples_per_output)
            sample_Y = floatX(to_categorical(sample_Y))
            sample_X = gen(sample_Z, sample_Y)
            img = tile_raster_images(sample_X, tile_spacing=(1, 1))
            filename = os.path.join(out_folder,
                                    'samples{:05d}.png'.format(epoch))
            imsave(filename, img)
            save_model(out_gen, out_discr, history, model_filename)
    save_model(out_gen, out_discr, history, model_filename)


def save_model(net_gen, net_discr, stats, filename):
    return {'generator_weights': layers.get_all_param_values(net_gen),
            'discrimimator_weights': layers.get_all_param_values(net_discr),
            'stats': stats}


if __name__ == '__main__':
    main()
