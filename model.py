from lasagne import layers, init
from lasagne.nonlinearities import rectify, sigmoid, linear, tanh, LeakyRectify, elu
import theano.tensor as T
from layers import DenseCondConcat, ConvCondConcat
from helpers import Deconv2DLayer
from lasagne.layers import batch_norm

import theano
import numpy as np

leaky_rectify = LeakyRectify(0.2)


def cond_dcgan_28x28(z_dim=100, w=28, h=28, c=1, nb_outputs=10):
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
    )
    X = ConvCondConcat((X, y_in))

    X = layers.Conv2DLayer(
        X,
        num_filters=64,
        filter_size=(5, 5),
        stride=2,
        nonlinearity=leaky_rectify,
    )
    X = batch_norm(X)
    X = ConvCondConcat((X, y_in))
    X = layers.DenseLayer(
        X,
        1024,
        nonlinearity=leaky_rectify,
    )
    X = batch_norm(X)
    X = DenseCondConcat((X, y_in))
    X = layers.DenseLayer(
        X,
        1,
        nonlinearity=sigmoid,
    )
    out_discr = X

    # generator

    Z = DenseCondConcat((z_in, y_in))
    Z = layers.DenseLayer(
        Z,
        1024,
        nonlinearity=rectify,
    )
    Z = batch_norm(Z)
    Z = DenseCondConcat((Z, y_in))
    Z = layers.DenseLayer(
        Z,
        128*7*7,
        nonlinearity=rectify,
    )

    Z = batch_norm(Z)

    Z = layers.ReshapeLayer(
        Z,
        ([0], 128, 7, 7)
    )
    Z = ConvCondConcat((Z, y_in))
    Z = Deconv2DLayer(
        Z,
        num_filters=64,
        filter_size=(5, 5),
        stride=2,
        pad=(5 - 1)/2,
        nonlinearity=rectify,

    )
    Z = batch_norm(Z)
    Z = ConvCondConcat((Z, y_in))
    Z = Deconv2DLayer(
        Z,
        num_filters=c,
        filter_size=(5, 5),
        stride=2,
        pad=(5-1)/2,
        nonlinearity=tanh,
    )
    out_gen = Z
    return x_in, y_in, z_in, out_gen, out_discr


def dcgan_28x28(z_dim=100, w=28, h=28, c=1):
    x_in = layers.InputLayer((None, c, w, h), name="input")
    z_in = layers.InputLayer((None, z_dim), name="z")

    # discrimimator
    X = layers.Conv2DLayer(
        x_in,
        num_filters=64,
        filter_size=(5, 5),
        stride=2,
        nonlinearity=leaky_rectify
    )

    X = layers.Conv2DLayer(
        X,
        num_filters=64,
        filter_size=(5, 5),
        stride=2,
        nonlinearity=leaky_rectify
    )
    #X = batch_norm(X)
    X = layers.DenseLayer(
        X,
        1024,
        nonlinearity=leaky_rectify
    )
    #X = batch_norm(X)
    X = layers.DenseLayer(
        X,
        1,
        nonlinearity=sigmoid
    )
    out_discr = X
    # generator

    Z = layers.DenseLayer(
        z_in,
        1024,
        nonlinearity=rectify
    )
    Z = batch_norm(Z)
    Z = layers.DenseLayer(
        Z,
        64*7*7,
        nonlinearity=rectify
    )
    Z = batch_norm(Z)
    Z = layers.ReshapeLayer(
        Z,
        ([0], 64, 7, 7)
    )
    Z = Deconv2DLayer(
        Z,
        num_filters=64,
        filter_size=(5, 5),
        stride=2,
        pad=(5-1)/2,
        nonlinearity=rectify
    )
    Z = batch_norm(Z)
    Z = Deconv2DLayer(
        Z,
        num_filters=c,
        filter_size=(5, 5),
        stride=2,
        pad=(5-1)/2,
        nonlinearity=tanh
    )
    out_gen = Z
    return x_in, z_in, out_gen, out_discr


def dcgan_64x64(z_dim=100, w=64, h=64, c=1):
    x_in = layers.InputLayer((None, c, w, h), name="input")
    z_in = layers.InputLayer((None, z_dim), name="z")

    # discrimimator
    X = layers.Conv2DLayer(
        x_in,
        num_filters=64,
        filter_size=(5, 5),
        stride=2,
        nonlinearity=leaky_rectify
    )

    X = layers.Conv2DLayer(
        X,
        num_filters=64,
        filter_size=(5, 5),
        stride=2,
        nonlinearity=leaky_rectify
    )
    X = layers.Conv2DLayer(
        X,
        num_filters=64,
        filter_size=(5, 5),
        stride=2,
        nonlinearity=leaky_rectify
    )
    X = layers.Conv2DLayer(
        X,
        num_filters=64,
        filter_size=(5, 5),
        stride=2,
        nonlinearity=leaky_rectify
    )
    X = layers.DenseLayer(
        X,
        1,
        nonlinearity=sigmoid
    )
    out_discr = X

    # generator
    Z = layers.DenseLayer(
        z_in,
        1024*4*4,
        nonlinearity=rectify
    )
    Z = layers.ReshapeLayer(
        Z,
        ([0], 1024, 4, 4)
    )
    Z = Deconv2DLayer(
        Z,
        num_filters=512,
        filter_size=(5, 5),
        stride=2,
        pad=(5-1)/2,
        nonlinearity=rectify
    )
    Z = batch_norm(Z)
    Z = Deconv2DLayer(
        Z,
        num_filters=256,
        filter_size=(5, 5),
        stride=2,
        pad=(5-1)/2,
        nonlinearity=rectify
    )
    Z = Deconv2DLayer(
        Z,
        num_filters=128,
        filter_size=(5, 5),
        stride=2,
        pad=(5-1)/2,
        nonlinearity=rectify
    )
    Z = layers.Conv2DLayer(
        Z,
        num_filters=128,
        filter_size=(5 - 1, 5 - 1),
        nonlinearity=rectify
    )
    Z = Deconv2DLayer(
        Z,
        num_filters=c,
        filter_size=(5, 5),
        stride=2,
        nonlinearity=tanh
    )
    out_gen = Z
    return x_in, z_in, out_gen, out_discr


def dcgan(z_dim=100, w=64, h=64, c=1,
          num_filters_g=1024,
          num_filters_d=128,
          start_w=4, start_h=4, filter_size=5, do_batch_norm=True):

    assert 2**int(np.log2(w)) == w
    assert 2**int(np.log2(h)) == h

    nb_layers = int(np.log2(w) - np.log2(start_w))
    x_in = layers.InputLayer((None, c, w, h), name="input")
    z_in = layers.InputLayer((None, z_dim), name="z")

    nonlin_discr = leaky_rectify
    nonlin_gen = rectify
    # discrimimator
    X = x_in
    for i in range(nb_layers):
        X = layers.Conv2DLayer(
            X,
            num_filters=num_filters_d,
            filter_size=(filter_size, filter_size),
            stride=2,
            nonlinearity=nonlin_discr,
        )
        if do_batch_norm and i > 0:
             X = batch_norm(X)
        num_filters_d *= 2
    X = batch_norm(X)
    X = layers.DenseLayer(
        X,
        1,
        nonlinearity=sigmoid
    )
    out_discr = X

    # generator
    Z = layers.DenseLayer(
        z_in,
        num_filters_g*start_w*start_h,
        nonlinearity=nonlin_gen
    )
    if do_batch_norm:
        Z = batch_norm(Z)
    Z = layers.ReshapeLayer(
        Z,
        ([0], num_filters_g, start_w, start_h)
    )
    for i in range(nb_layers - 1):
        num_filters_g /= 2
        Z = Deconv2DLayer(
            Z,
            num_filters=num_filters_g,
            filter_size=(filter_size, filter_size),
            stride=2,
            nonlinearity=nonlin_gen,
            pad=(filter_size - 1) / 2
        )
        if do_batch_norm:
            Z = batch_norm(Z)
    Z = Deconv2DLayer(
        Z,
        num_filters=c,
        filter_size=(filter_size, filter_size),
        stride=2,
        nonlinearity=tanh,
        pad=(filter_size - 1) / 2
    )
    out_gen = Z
    return x_in, z_in, out_gen, out_discr


def dcgan_small(z_dim=100, w=28, h=28, c=1):
    return dcgan(z_dim=100, w=w, h=h, c=c, num_filters_g=128, num_filters_d=32, start_w=4, start_h=4, filter_size=5, do_batch_norm=True)


def dcgan_standard(z_dim=100, w=64, h=64, c=3):
    return dcgan(z_dim=z_dim, w=w, h=h, c=c, num_filters_g=1024, num_filters_d=128, start_w=4, start_h=4, filter_size=5, do_batch_norm=True)

if __name__ == '__main__':

    # cond dcgan 28x28 (mnist)
    x_in, y_in, z_in, out_gen, out_discr = cond_dcgan_28x28(z_dim=100, c=1, nb_outputs=10)

    X = T.tensor4()
    Y =  T.matrix()
    Z = T.matrix()

    gen = theano.function([Z, Y], layers.get_output(out_gen, {z_in: Z, y_in: Y} ))
    discr = theano.function([X, Y], layers.get_output(out_discr, {x_in: X, y_in: Y} ))

    z_ex = np.random.uniform(size=(20, 100)).astype(np.float32)
    y_ex = np.random.uniform(size=(20, 10)).astype(np.float32)
    x_ex = np.random.uniform(size=(20, 1, 28, 28)).astype(np.float32)
    print(gen(z_ex, y_ex).shape)
    print(discr(x_ex, y_ex).shape)


    # unsupervised dcgan 28x28 (mnist)

    x_in, z_in, out_gen, out_discr = dcgan_28x28(z_dim=100, c=1)

    X = T.tensor4()
    Z = T.matrix()

    gen = theano.function([Z], layers.get_output(out_gen, {z_in: Z}))
    discr = theano.function([X], layers.get_output(out_discr, {x_in: X}))

    z_ex = np.random.uniform(size=(20, 100)).astype(np.float32)
    x_ex = np.random.uniform(size=(20, 1, 28, 28)).astype(np.float32)
    print(gen(z_ex).shape)
    print(discr(x_ex).shape)

    # unsupervised dcgan 64x64

    x_in, z_in, out_gen, out_discr = dcgan_64x64(z_dim=100, c=3)

    X = T.tensor4()
    Z = T.matrix()

    gen = theano.function([Z], layers.get_output(out_gen, {z_in: Z}))
    discr = theano.function([X], layers.get_output(out_discr, {x_in: X}))

    z_ex = np.random.uniform(size=(20, 100)).astype(np.float32)
    x_ex = np.random.uniform(size=(20, 3, 64, 64)).astype(np.float32)
    print(gen(z_ex).shape)
    print(discr(x_ex).shape)

    # cond generic dcgan
    x_in, y_in, z_in, out_gen, out_discr = cond_dcgan(z_dim=100, c=3, w=128, h=128, start_w=4, start_h=4, nb_outputs=10)

    X = T.tensor4()
    Z = T.matrix()

    gen = theano.function([Z, Y], layers.get_output(out_gen, {z_in: Z, y_in: Y}))
    discr = theano.function([X, Y], layers.get_output(out_discr, {x_in: X, y_in: Y}))

    z_ex = np.random.uniform(size=(20, 100)).astype(np.float32)
    y_ex = np.random.uniform(size=(20, 10)).astype(np.float32)
    x_ex = np.random.uniform(size=(20, 3, 128, 128)).astype(np.float32)
    print(gen(z_ex, y_ex).shape)
    print(discr(x_ex, y_ex).shape)

    # unsupervised generic dcgan
    x_in, z_in, out_gen, out_discr = dcgan(z_dim=100, c=3, w=128, h=128, start_w=4, start_h=4)

    X = T.tensor4()
    Z = T.matrix()

    gen = theano.function([Z], layers.get_output(out_gen, {z_in: Z}))
    discr = theano.function([X], layers.get_output(out_discr, {x_in: X}))

    z_ex = np.random.uniform(size=(20, 100)).astype(np.float32)
    x_ex = np.random.uniform(size=(20, 3, 128, 128)).astype(np.float32)
    print(gen(z_ex).shape)
    print(discr(x_ex).shape)
