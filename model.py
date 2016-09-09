from lasagne import layers, init
from lasagne.nonlinearities import rectify, sigmoid, linear, tanh, LeakyRectify, elu
import theano.tensor as T
from layers import DenseCondConcat, ConvCondConcat, GenericBrushLayer, Repeat, TensorDenseLayer
from lasagne.layers import batch_norm, Conv2DLayer
from helpers import Deconv2DLayer, Deconv2DLayerScaler


import theano
import numpy as np

leaky_rectify = LeakyRectify(0.2)


def brush(z_dim=100, w=64, h=64, c=1,
          num_filters_d=8,
          start_w=4, start_h=4,
          filter_size=5,
          do_batch_norm=True,
          scale=0.02,
          nb_recurrent_layers=1,
          nb_recurrent_units=100,
          nb_fc_layers=1,
          n_steps=20,
          patch_size=5,
          nb_fc_units=[1000]):

    nb_layers = int(np.log2(w) - np.log2(start_w))
    x_in = layers.InputLayer((None, c, w, h), name="input")
    z_in = layers.InputLayer((None, z_dim), name="z")

    nonlin_discr = leaky_rectify
    nonlin_gen = rectify
    # discrimimator (same than dcgan)
    X = x_in
    for i in range(nb_layers):
        X = layers.Conv2DLayer(
            X,
            num_filters=num_filters_d,
            filter_size=(filter_size, filter_size),
            stride=2,
            nonlinearity=nonlin_discr,
            W=init.Normal(mean=0, std=scale)  # 1 for gain
        )
        if do_batch_norm and i > 0:
            X = batch_norm(X)
        #num_filters_d *= 2
    X = batch_norm(X)
    X = layers.DenseLayer(
        X,
        1,
        W=init.Normal(std=scale),
        nonlinearity=sigmoid,
    )
    out_discr = X
    # generator (brush)
    if type(nb_fc_units) != list:
        nb_fc_units = [nb_fc_units] * nb_fc_layers
    if type(nb_recurrent_units) != list:
        nb_recurrent_units = [nb_recurrent_units] * nb_recurrent_layers

    Z = z_in
    for i in range(nb_fc_layers):
        Z = layers.DenseLayer(
            Z, nb_fc_units[i],
            W=init.GlorotUniform(gain='relu'),
            nonlinearity=nonlin_gen)
        if do_batch_norm:
            Z = batch_norm(Z)

    Z = Repeat(Z, n_steps)

    recurrent_model = layers.RecurrentLayer
    for i in range(nb_recurrent_layers):
        Z = recurrent_model(Z, nb_recurrent_units[i])

    l_coord = TensorDenseLayer(Z, 5, nonlinearity=linear, name="coord")
    l_coord = batch_norm(l_coord)

    # DECODING PART
    patches = np.ones((1, c, patch_size, patch_size))
    patches = patches.astype(np.float32)

    l_brush = GenericBrushLayer(
        l_coord,
        w, h,
        n_steps=n_steps,
        patches=patches,
        col='rgb' if c == 3 else 'grayscale',
        return_seq=False,
        reduce_func=lambda x, y: x+y,
        to_proba_func=T.nnet.softmax,
        normalize_func=T.nnet.sigmoid,
        x_sigma=0.5,
        y_sigma=0.5,
        x_stride=1,
        y_stride=1,
        patch_index=0,
        color=[1],
        x_min=0,
        x_max='width',
        y_min=0,
        y_max='height',
        eps=0,
    )
    l_raw_out = l_brush
    l_scaled_out = layers.ScaleLayer(
        l_raw_out, scales=init.Constant(2.), name="scaled_output")
    l_biased_out = layers.BiasLayer(
        l_scaled_out, b=init.Constant(-1), name="biased_output")

    l_out = layers.NonlinearityLayer(
        l_biased_out,
        nonlinearity=sigmoid,
        name="output")
    out_gen = l_out
    return x_in, z_in, out_gen, out_discr


def dcgan(z_dim=100, w=64, h=64, c=1,
          num_filters_g=1024,  #start by this and divide by 2 after each layer (stop at num_filters_d)
          num_filters_d=128,   #start  by this and double after each layer (stop at num_filters_g)
          start_w=4, start_h=4, filter_size=5, do_batch_norm=True,
          scale=0.02):

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
            W=init.Normal(mean=0, std=scale)  # 1 for gain
        )
        if do_batch_norm and i > 0:
            X = batch_norm(X)
        num_filters_d *= 2
    X = batch_norm(X)
    X = layers.DenseLayer(
        X,
        1,
        W=init.Normal(std=scale),
        nonlinearity=sigmoid,
    )
    out_discr = X

    # generator
    Z = layers.DenseLayer(
        z_in,
        num_filters_g*start_w*start_h,
        nonlinearity=nonlin_gen,
        W=init.Normal(std=scale)
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
            pad=(filter_size - 1) / 2,
            W=init.Normal(mean=0, std=scale)
        )
        if do_batch_norm:
            Z = batch_norm(Z)
    Z = Deconv2DLayer(
        Z,
        num_filters=c,
        filter_size=(filter_size, filter_size),
        stride=2,
        nonlinearity=sigmoid,
        pad=(filter_size - 1) / 2,
        W=init.Normal(std=scale)  # 1 for gain
    )
    print(Z.output_shape)
    out_gen = Z
    return x_in, z_in, out_gen, out_discr


def dcgan_small(z_dim=100, w=28, h=28, c=1):
    return dcgan(z_dim=100, w=w, h=h,
                 c=c,
                 num_filters_g=128, num_filters_d=8,
                 start_w=4, start_h=4,
                 scale=0.01778279410038923,
                 filter_size=5, do_batch_norm=True)


def dcgan_standard(z_dim=100, w=64, h=64, c=3):
    return dcgan(z_dim=z_dim, w=w, h=h, c=c,
                 num_filters_g=1024, num_filters_d=128,
                 start_w=4, start_h=4, filter_size=5,
                 do_batch_norm=True)

def dcgan_a(z_dim=100, w=64, h=64, c=3):
    return dcgan(z_dim=z_dim, w=w, h=h, c=c,
                 num_filters_g=1024, num_filters_d=128,
                 start_w=4, start_h=4, filter_size=5,
                 scale=0.01778279410038923,
                 do_batch_norm=True)

def dcgan_b(z_dim=100, w=64, h=64, c=3):
    return dcgan(z_dim=z_dim, w=w, h=h, c=c,
                 num_filters_g=1024, num_filters_d=32,
                 start_w=4, start_h=4, filter_size=5,
                 scale=0.01778279410038923,
                 do_batch_norm=True)

def dcgan_test(z_dim=100, w=64, h=64, c=1,
          num_filters_g=1024,  #start by this and divide by 2 after each layer (stop at num_filters_d)
          num_filters_d=128, # start  by this and double after each layer (stop at num_filters_g)
          start_w=4, start_h=4, filter_size=5, do_batch_norm=True,
          scale=0.02):

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
            W=init.Normal(mean=0, std=scale)  # 1 for gain
        )
        if do_batch_norm and i > 0:
            X = batch_norm(X)
        num_filters_d *= 2
    X = batch_norm(X)
    X = layers.DenseLayer(
        X,
        1,
        W=init.Normal(std=scale),
        nonlinearity=sigmoid,
    )
    out_discr = X

    # generator
    Z = layers.DenseLayer(
        z_in,
        num_filters_g*start_w*start_h,
        nonlinearity=nonlin_gen,
        W=init.Normal(std=scale)
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
            pad=(filter_size - 1) / 2,
            W=init.Normal(mean=0, std=scale)
        )
        if do_batch_norm:
            Z = batch_norm(Z)
    Z = Deconv2DLayerScaler(
        Z,
        num_filters=c,
        filter_size=(filter_size, filter_size),
        stride=2,
        nonlinearity=sigmoid,
        pad=(filter_size - 1) / 2,
        W=init.Normal(std=scale)  # 1 for gain
    )
    print(Z.output_shape)
    out_gen = Z
    return x_in, z_in, out_gen, out_discr

# some examples for clarity purpose


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
    X = batch_norm(X)
    X = ConvCondConcat((X, y_in))
    X = layers.DenseLayer(
        X,
        128,
        nonlinearity=leaky_rectify,
        W=init.Normal(0.02)

    )
    X = batch_norm(X)
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
    Z = batch_norm(Z)
    Z = DenseCondConcat((Z, y_in))
    Z = layers.DenseLayer(
        Z,
        128*7*7,
        nonlinearity=rectify,
        W=init.Normal(0.02)

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
        W=init.Normal(0.02),
        #func='alec'

    )
    Z = batch_norm(Z)
    Z = ConvCondConcat((Z, y_in))
    Z = Deconv2DLayerScaler(
        Z,
        num_filters=c,
        filter_size=(5, 5),
        stride=2,
        pad=(5-1)/2,
        nonlinearity=tanh,
        W=init.Normal(0.02),
        #W=init.GlorotUniform()
        #func='alec'
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
    Z = Deconv2DLayerScaler(
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


if __name__ == '__main__':

    # brush

    x_in, z_in, out_gen, out_discr = brush(z_dim=100, c=3, w=128, h=128, start_w=4, start_h=4)

    X = T.tensor4()
    Z = T.matrix()

    gen = theano.function([Z], layers.get_output(out_gen, {z_in: Z}))
    discr = theano.function([X], layers.get_output(out_discr, {x_in: X}))

    z_ex = np.random.uniform(size=(20, 100)).astype(np.float32)
    x_ex = np.random.uniform(size=(20, 3, 128, 128)).astype(np.float32)
    print(gen(z_ex).shape)
    print(discr(x_ex).shape)

    # cond dcgan 28x28 (mnist)
    x_in, y_in, z_in, out_gen, out_discr = cond_dcgan_28x28(z_dim=100, c=1, nb_outputs=10)

    X = T.tensor4()
    Y = T.matrix()
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
