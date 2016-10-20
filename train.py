import sys
from functools import partial
from itertools import imap, cycle
from tabulate import tabulate
import os
from collections import OrderedDict
import dill
import click
import json
from tqdm import tqdm
from time import time
import logging
import glob

import numpy as np
import theano
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
from skimage.io import imread_collection

import theano.tensor as T
from lasagne import layers, updates, init
from lasagne.regularization import l2, regularize_network_params

import model
from helpers import iterate_minibatches, to_categorical, floatX, dispims, mkdir_path
from helpers import resize, resize_dataset

import datakit

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@click.command()
@click.option('--params', default='', help='only if pattern is not provided', required=False)
def traincollection(params):
    params = json.load(open(params))
    train(params)

unif = partial(np.random.uniform, low=-1, high=1)
def train(params):
    outdir = params.get('outdir', 'out')
    mkdir_path(outdir)
    mkdir_path(outdir + '/mb')
    w = params['w']
    h = params['h']
    c = params['c']
    pipeline = params.get('data_pipeline')
    model_name = params['model_name']
    # assume w and h are power of two
    nb_examples = params.get('nb_examples')
    data_in_memory = params.get('data_in_memory', False)
    lr_initial = params.get('lr', 0.0002)
    nb_epochs = params.get('nb_epochs', 2000)
    z_dim = params.get('z_dim', 100)
    batch_size = params.get('batch_size', 128)
    lr = theano.shared(floatX(np.array(lr_initial)))
    rng = np.random.RandomState(1234)
    subset_ratio = params.get('subset_ratio', 1)
    b1 = params.get('b1', 0.5)
    l2_coef = params.get('l2_coef', 0)
    epoch_start_decay = params.get('epoch_start_decay', None)
    lr_decay = params.get('lr_decay', 0.97)
    apply_crop = params.get('apply_crop', False)
    crop_h = params.get('crop_h', None)
    crop_w = params.get('crop_w', None)
    seed = params.get('seed', None)
    nb_discr_updates = params.get('nb_discriminator_updates', 1)
    nb_gen_updates = params.get('nb_generator_updates', 1)
    algo = params.get('algo', 'adam')
    eps = params.get('eps', 0)
    discr_loss = params.get('discriminator_loss', 'cross_entropy')
    nb_examples = params.get('nb_examples')
    shuffle = params.get('shuffle', True)
    model_params = params.get('model', {})
    np.random.seed(seed)
    
    # load data
    logger.info('Load data iterator...')
    if not nb_examples:
        nb_examples = len(list(datakit.image.pipeline_load(pipeline[0:1])))
    logger.info('Total nb of examples : {}'.format(nb_examples))
    train_iter = datakit.image.pipeline_load(pipeline)
    train_iter = datakit.helpers.minibatch(train_iter, batch_size=batch_size)
    train_iter = datakit.helpers.expand_dict(train_iter)
    train_iter = imap(partial(datakit.helpers.dict_apply, fn=floatX, cols=['X']), train_iter)
    train_iter = cycle(train_iter)

    logger.info('Saving real data into file...')
    xdisp = next(train_iter)['X']
    xdisp = xdisp[0:100]
    xdisp = xdisp.transpose((0, 2, 3, 1))
    xdisp = np.clip(xdisp, 0, 1)
    img = dispims(xdisp[0:100], border=1, normalize=False)
    filename = os.path.join(outdir, 'real_data.png')
    imsave(filename, img)

    # compile net
    logger.info('Compile network...')
    X_real = T.tensor4()
    Z = T.matrix()
    builder = getattr(model, model_name)
    builder_args = dict(z_dim=z_dim, w=w, h=h, c=c)
    builder_args.update(model_params)
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
        normalize_ = lambda x:(x / (eps + T.sqrt((x**2).sum(axis=1, keepdims=True)))) 
        def normalize(x):
            return layers.ExpressionLayer(x,  normalize_)
        #cost of gen : match stats of real data
        conv_layers = (layer for layer in discr_layers if 'conv' in layer.name)
        conv_layers = map(flatten, conv_layers)
        #conv_layers = map(normalize, conv_layers)
        conv_layers = conv_layers[::-1][0:1]
        f_layer = layers.ConcatLayer(conv_layers, axis=1)
        f_real = layers.get_output(f_layer, {x_in: X_real})
        f_gen = layers.get_output(f_layer, {x_in: X_gen})
        g_cost_d = ((f_real.mean(axis=1) - f_gen.mean(axis=1)) ** 2).mean()
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
    minibatches_per_epoch = (nb_examples / batch_size) + (1 if nb_examples % batch_size else 0)
    for epoch in range(nb_epochs):
        logger.info('Starting epoch {}...'.format(epoch))
        t = time()
        g_loss = []
        d_loss = []
        pred_real = []
        pred_gen = []
        for batch_index in tqdm(range(minibatches_per_epoch)):
            train = next(train_iter)
            train_X = train['X']
            train_Z = floatX(unif(size=(len(train_X), z_dim)))
            if batch_index % 2 == 0:
                for i in range(nb_discr_updates):
                    loss = train_d(train_X, train_Z)[1]
                d_loss.append(loss)
            else:
                for i in range(nb_gen_updates):
                    loss = train_g(train_X, train_Z)[0]
                g_loss.append(loss)
            pred_real.append(np.mean(pred_discr(train_X)))
            pred_gen.append(np.mean(pred_discr(gen(train_Z))))
        delta_time = time() - t
        # save stats
        logger.info('Saving stats...')
        stats = {}
        stats['g_loss'] = np.mean(g_loss)
        stats['d_loss'] = np.mean(d_loss)
        stats['pred_real'] = np.mean(pred_real)
        stats['pred_gen'] = np.mean(pred_gen)
        stats['duration'] = delta_time
        history.append(stats)
        pd.DataFrame(history).to_csv('{}/hist.csv'.format(outdir))
        # show
        print(tabulate([stats], headers='keys'))
        # generate
        logger.info('Generating and saving samples...')
        img = generate_samples(nb_samples=100, z_dim=z_dim, gen_func=gen, prior=unif)
        filename = os.path.join(outdir, 'samples{:05d}.png'.format(epoch))
        imsave(filename, img)
        save_model(builder, builder_args, out_gen, out_discr, model_filename)
        # change lr
        if epoch_start_decay and epoch >= epoch_start_decay:
           lr.set_value(floatX(np.array(lr.get_value() * lr_decay)))
           logger.info('Learning rate becomes : {}'.format(lr.get_value()))
    save_model(builder, builder_args, out_gen, out_discr, model_filename)
    return history

def generate_samples(nb_samples, z_dim, gen_func, prior=unif):
    sample_Z = floatX(prior(size=(nb_samples, z_dim)))
    sample_X = gen_func(sample_Z)
    sample_X = deprocess_input(sample_X)
    sample_X = np.clip(sample_X, 0, 1)
    img = dispims(sample_X, border=1, normalize=False)
    return img

def deprocess_input(X):
    return X.transpose((0, 2, 3, 1))

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
