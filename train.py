from lasagne import layers, updates, init
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import theano
from helpers import iterate_minibatches, to_categorical, floatX, dispims, mkdir_path
from tabulate import tabulate
import os
from collections import OrderedDict
import dill
import model
import click
from tqdm import tqdm
import pandas as pd
import theano.tensor as T
from lasagne.regularization import l2, regularize_network_params
from time import time
from skimage.io import imsave
from skimage.io import imread_collection
from skimage.transform import resize
from data import load_data
import json


@click.command()
@click.option('--outdir', default='.', help='Output directory', required=False)
@click.option('--pattern', default='', help='Pattern of image filenames to train on', required=False)
@click.option('--model_name', default='dcgan_small', help='Model name', required=False)
@click.option('--w', default=64, help='rescale images to a width of w', required=False)
@click.option('--h', default=64, help='rescale images to a height of h', required=False)
@click.option('--c', default=3, help='1 if grayscale images otherwise 3', required=False)
@click.option('--data-in-memory/--no-data-in-memory', default=True, help='', required=False)
@click.option('--dataset', default='', help='only if pattern is not provided', required=False)
@click.option('--params', default='', help='only if pattern is not provided', required=False)
def traincollection(outdir, pattern, model_name, w, h, c, data_in_memory, dataset, params, **kw):
    train(outdir, pattern, model_name, w, h, c, data_in_memory, dataset, params, **kw)

def train(outdir='.', 
          pattern='',
          model_name='dcgan',
          w=64, h=64, c=1, 
          data_in_memory=True,
          dataset='mnist', 
          params='', **kw):
    if params.endswith('.json'):
        kw.update(json.load(open(params)))
    elif params:
        kw.update(json.loads(params))
    mkdir_path(outdir)
    w = int(w)
    h = int(h)
    c = int(c)
    # assume w and h are power of two
    lr_initial = kw.get('lr', 0.0002)
    nb_epochs = kw.get('nb_epochs', 2000)
    z_dim = kw.get('z_dim', 100)
    batch_size = kw.get('batch_size', 128)
    lr = theano.shared(floatX(np.array(lr_initial)))
    rng = np.random.RandomState(1234)
    subset_ratio = kw.get('subset_ratio', 1)
    b1 = kw.get('b1', 0.5)
    l2_coef = kw.get('l2_coef', 0)
    epoch_start_decay = kw.get('epoch_start_decay', None)
    lr_decay = kw.get('lr_decay', 0.97)
    apply_crop = kw.get('apply_crop', False)
    crop_h = kw.get('crop_h', None)
    crop_w = kw.get('crop_w', None)
    seed = kw.get('seed', None)
    nb_discr_updates = kw.get('nb_discriminator_updates', 1)
    nb_gen_updates = kw.get('nb_generator_updates', 1)
    algo = kw.get('algo', 'adam')
    eps = kw.get('eps', 0)
    discr_loss = kw.get('discriminator_loss', 'cross_entropy')
    nb_examples = kw.get('nb_examples')
    np.random.seed(seed)

    def resize_input(X, wh):
        w, h = wh
        if apply_crop:
            X = crop(X, crop_w, crop_h)
        X = resize(X, (w, h), preserve_range=True)
        return X

    def crop(x, target_w, target_h):
        w, h = x.shape[0:2]
        if h >= target_h:
            a = (h - target_h) / 2
            b = h - target_h - a
            x = x[a:-b]
        if w >= target_w:
            a = (w - target_w) / 2
            b = w - target_w - a
            x = x[:, a:-b]
        return x

    def rescale_input(X):
        # assumes X has shape (B, w, h, c) and returns (B, ww, hh, c)
        X_rescaled = np.empty((len(X), w, h, c))
        for i in range(len(X)):
            Xi = np.array(X[i])
            if len(Xi.shape) == 3:
                Xi = Xi[:, :, 0:c]
                X_rescaled[i] = resize_input(Xi, (w, h))
            else:
                X_rescaled[i, :, :, 0] = resize_input(Xi, (w, h))
        X_rescaled = floatX(X_rescaled) / X_rescaled.max()
        return X_rescaled

    def preprocess_input(X):
        if not isinstance(X, np.ndarray):
            X = floatX(np.array(X))
        if len(X.shape) == 3:
            X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
            return X
        else:
            X = X.transpose((0, 3, 1, 2))
            return X

    def deprocess_input(X):
        X = X.transpose((0, 2, 3, 1))
        return X

    def generate_samples(nb_samples):
        sample_Z = floatX(rng.uniform(-1., 1., size=(nb_samples, z_dim)))
        sample_X = gen(sample_Z)
        sample_X = deprocess_input(sample_X)
        if sample_X.shape[3] not in (1, 3):
            channel = rng.randint(0, sample_X.shape[3])
            sample_X = sample_X[:, :, :, channel:channel+1]
        img = dispims(sample_X, border=1)
        return img

    # load data
    if pattern != '':
        X = imread_collection(pattern)
        if nb_examples:
            X = X[0:nb_examples]
        X = np.array(X)
        indices = np.arange(len(X))
        np.random.shuffle(X)
    else:
        assert dataset != ''
        data_train, data_valid = load_data(dataset, training_subset=subset_ratio, valid_ratio=0, shuffle=kw.get('shuffle', True))
        X = data_train.X
        if nb_examples:
            X = X[0:nb_examples]

    if data_in_memory is True:
        if X[0].shape[0:2] != (w, h):
            X = rescale_input(X)
        if X.shape[3] not in (1, 3):
            xdisp = X[:, :, :, 0:1]
        else:
            xdisp = X
        xdisp = xdisp[0:100]
        img = dispims(xdisp[0:100], border=1)
        filename = os.path.join(outdir, 'real_data.png')
        imsave(filename, img)
    print(X.shape)
    # compile net
    X_real = T.tensor4()
    Z = T.matrix()

    builder = getattr(model, model_name)
    builder_args = dict(z_dim=z_dim, w=w, h=h, c=c)
    builder_args.update(kw.get('model', {}))
    x_in, z_in, out_gen, out_discr = builder(**builder_args)

    X_gen = layers.get_output(out_gen, {z_in: Z} )

    discr_layers = layers.get_all_layers(out_discr)
    gen_layers = layers.get_all_layers(out_gen)
    p_layer = out_discr
    p_real = layers.get_output(p_layer, {x_in: X_real} )
    pred_discr = theano.function([X_real], p_real)
    p_gen = layers.get_output(p_layer, {x_in: X_gen} )

    # cost of discr : predict 0 for gen and 1 for real
    p_real = theano.tensor.clip(p_real, eps, 1 - eps)
    d_cost_real = T.nnet.binary_crossentropy(p_real, T.ones(p_real.shape)).mean()
    p_gen = theano.tensor.clip(p_gen, eps, 1 - eps)
    d_cost_gen = T.nnet.binary_crossentropy(p_gen, T.zeros(p_gen.shape)).mean()
    # cost of gen
    if discr_loss == 'cross_entropy':
        #cost of gen : make the discr predict 1 for gen
        g_cost_d = T.nnet.binary_crossentropy(p_gen, T.ones(p_gen.shape)).mean()
    elif discr_loss == 'feature_matching':
        flatten_ = lambda x:x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
        def flatten(layer):
            shape = (None, np.prod(layer.output_shape[1:]))
            return layers.ExpressionLayer(layer, flatten_, output_shape=shape)
        #cost of gen : match stats of real data
        conv_layers = (layer for layer in discr_layers if 'conv' in layer.name)
        conv_layers = map(flatten, conv_layers)
        f_layer = layers.ConcatLayer(conv_layers, axis=1)
        f_real = layers.get_output(f_layer, {x_in: X_real})
        f_gen = layers.get_output(f_layer, {x_in: X_gen})
        g_cost_d = ((f_real.mean(axis=1) - f_gen.mean(axis=1)) ** 2).mean()
        print(discr_loss)
    else:
        raise ValueError()

    d_cost = d_cost_real + d_cost_gen
    g_cost = g_cost_d
    
    cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

    d_cost_reg = l2_coef * regularize_network_params(p_layer, l2)
    g_cost_reg = l2_coef * regularize_network_params(out_gen, l2)

    discrim_params = layers.get_all_params(out_discr, trainable=True)
    gen_params = layers.get_all_params(out_gen, trainable=True)

    algo_kw = {'learning_rate': lr}
    if algo == 'adam':
        algo_kw['beta1'] = b1

    algo = {'adam': updates.adam, 'adadelta': updates.adadelta}[algo]

    d_updates = algo(d_cost + d_cost_reg, discrim_params, **algo_kw)
    g_updates = algo(g_cost + g_cost_reg, gen_params, **algo_kw)

    all_updates = d_updates.copy()
    all_updates.update(g_updates)

    train_g = theano.function([X_real, Z], cost, updates=g_updates)
    train_d = theano.function([X_real, Z], cost, updates=d_updates)
    gen = theano.function([Z], X_gen)

    # Train
    model_filename = os.path.join(outdir, 'model.pkl')
    save_model(builder, builder_args,
               out_gen, out_discr,
               model_filename)

    history = []
    n_updates = 0
    for epoch in range(1, nb_epochs + 1):
        total_g_loss = 0
        total_d_loss = 0
        nb_g_updates = 0
        nb_d_updates = 0
        t = time()
        for train_X in tqdm(iterate_minibatches(X, targets=None, batchsize=batch_size)):
            if data_in_memory is False:
                train_X = rescale_input(train_X)
            train_X = preprocess_input(train_X)
            train_Z = floatX(rng.uniform(-1., 1., size=(len(train_X), z_dim)))

            for i in range(nb_discr_updates):
                total_d_loss += train_d(train_X, train_Z)[1]
                nb_d_updates += 1

            for i in range(nb_gen_updates):
                total_g_loss += train_g(train_X, train_Z)[0]
                nb_g_updates += 1

            if n_updates % 1000 == 0:
                nb_samples = 400
                img = generate_samples(nb_samples)
                filename = os.path.join(outdir, 'samples{:05d}_mb{:05d}.png'.format(epoch, n_updates))
                imsave(filename, img)

            n_updates += 1

        stats = OrderedDict()
        stats['epoch'] = epoch
        assert nb_g_updates > 0 and nb_d_updates > 0
        stats['g_loss'] = total_g_loss / nb_g_updates
        stats['d_loss'] = total_d_loss / nb_d_updates
        stats['train_time'] = time() - t
        sample_Z = floatX(rng.uniform(-1., 1., size=(128, z_dim)))
        sample_X = gen(sample_Z)
        stats['pred_acc'] = (pred_discr(sample_X) < 0.5).mean()
        history.append(stats)

        print(tabulate([stats], 'keys'))

        if epoch % 5 == 0:
            nb_samples = 400
            img = generate_samples(nb_samples)
            filename = os.path.join(outdir, 'samples{:05d}.png'.format(epoch))
            imsave(filename, img)
            save_model(builder, builder_args, out_gen, out_discr, model_filename)

            fig = plt.figure()
            l = [s['g_loss'] for s in history]
            plt.plot(l)
            plt.savefig(os.path.join(outdir, 'g_loss.png'))
            plt.close(fig)

            fig = plt.figure()
            l = [s['d_loss'] for s in history]
            plt.plot(l)
            plt.savefig(os.path.join(outdir, 'd_loss.png'))
            plt.close(fig)

        a = [s['g_loss'] for s in history]
        b = [s['d_loss'] for s in history]
        pd.DataFrame({'g_loss': a, 'd_loss': b}).to_csv("{}/stats.csv".format(outdir))
        if epoch_start_decay and epoch >= epoch_start_decay:
           lr.set_value(floatX(np.array(lr.get_value() * lr_decay)))
           print('lr becomes : {}'.format(lr.get_value()))
    save_model(builder, builder_args, out_gen, out_discr, model_filename)
    return history


def save_model(builder, args, net_gen, net_discr, filename):
    data = dict(
        builder=builder,
        args=args,
        generator_weights=layers.get_all_param_values(net_gen),
        discrimimator_weights=layers.get_all_param_values(net_discr)
    )
    with open(filename, "w") as fd:
        dill.dump(data, fd)


def load_model(filename, **kw):
    with open(filename, "r") as fd:
        data = dill.load(fd)
    builder = data['builder']
    builder_args = data['args']
    builder_args.update(kw)
    res = builder(**builder_args)
    gen, discr = res[len(res) - 2:]
    layers.set_all_param_values(gen, data['generator_weights'])
    layers.set_all_param_values(discr, data['discrimimator_weights'])
    return res


@click.command()
@click.option('--source', help='src', required=True)
@click.option('--dest', help='dst', required=True)
def dump(source, dest):
    dump_model(source, dest)

def dump_model(filename, dest_filename, **kw):
    import pickle
    model = load_model(filename, **kw)
    gen, discr = model[len(model) - 2:]
    gen_weights = layers.get_all_param_values(gen)
    disc_weights = layers.get_all_param_values(discr)
    data = dict(generator_weights=gen_weights, discriminator_weights=disc_weights, generator=gen, discriminator=discr)
    fd = open(dest_filename, 'w')
    pickle.dump(data, fd)
    fd.close()
