from lasagne import layers, updates
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

@click.group()
def main():
    pass


@click.command()
@click.option('--outdir', default='.', help='Output directory', required=False)
@click.option('--pattern', default='', help='Pattern of image filenames to train on', required=False)
@click.option('--model_name', default='dcgan_small', help='Model name', required=False)
@click.option('--w', default=64, help='rescale images to a width of w', required=False)
@click.option('--h', default=64, help='rescale images to a height of h', required=False)
@click.option('--c', default=3, help='1 if grayscale images otherwise 3', required=False)
@click.option('--data_in_memory', default=True, help='', required=False)
@click.option('--dataset', default='', help='only if pattern is not provided', required=False)
def traincollection(outdir, pattern, model_name, w, h, c, data_in_memory, dataset, **kw):
    train(outdir, pattern, model_name, w, h, c, data_in_memory, dataset, **kw)


def train(outdir='.', pattern='', model_name='dcgan',
          w=64, h=64, c=1, data_in_memory=True,
          dataset='mnist', **kw):
    import theano.tensor as T
    from time import time
    from skimage.io import imsave
    from skimage.io import imread_collection
    from skimage.transform import resize
    from data import load_data
    mkdir_path(outdir)
    w = int(w)
    h = int(h)
    c = int(c)
    # assume w and h are power of two
    lr_initial = kw.get('lr', 0.0002)
    nb_epochs = kw.get('nb_epochs', 200)
    z_dim = kw.get('z_dim', 100)
    batch_size = kw.get('batch_size', 128)
    lr = theano.shared(floatX(np.array(lr_initial)))
    rng = np.random.RandomState(1234)
    subset_ratio = kw.get('subset_ratio', 0.01)
    b1 = kw.get('b1', 0.5)

    def resize_input(X, wh):
        w, h = wh
        return resize(X, (w, h), preserve_range=True)

    def rescale_input(X):
        X_rescaled = np.empty((len(X), w, h, c))
        for i in range(len(X)):
            Xi = np.array(X[i])
            if len(Xi.shape) == 3:
                Xi = Xi[:, :, 0:c]
                X_rescaled[i] = resize_input(Xi, (w, h))
            else:
                X_rescaled[i, :, :, 0] = resize_input(Xi, (w, h))
        X_rescaled = floatX(X_rescaled) / X_rescaled.max()
        print(X_rescaled.min(), X_rescaled.max())
        return X_rescaled

    def preprocess_input(X):
        if len(X.shape) == 3:
            X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
            return X
        else:
            X = X.transpose((0, 3, 1, 2))
            return X

    def deprocess_input(X):
        X = X.transpose((0, 2, 3, 1))
        return X

    # load data
    if pattern != '':
        X = imread_collection(pattern)
    else:
        assert dataset != ''
        data_train, data_valid = load_data(dataset, training_subset=subset_ratio)
        X = data_train.X

    if data_in_memory is True:
        X = rescale_input(X)
        img = dispims(X[0:100], border=1)
        filename = os.path.join(outdir, 'real_data.png')
        imsave(filename, img)
        X = preprocess_input(X)

    # compile net
    X_real = T.tensor4()
    Z = T.matrix()

    builder = getattr(model, model_name)
    builder_args = dict(z_dim=z_dim, w=w, h=h, c=c)
    builder_args.update(kw.get('model', {}))
    x_in, z_in, out_gen, out_discr = builder(**builder_args)
    X_gen = layers.get_output(out_gen, {z_in: Z} )

    p_real = layers.get_output(out_discr, {x_in: X_real} )
    p_gen = layers.get_output(out_discr, {x_in: X_gen} )

    # cost of discr : predict 0 for gen and 1 for real
    d_cost_real = T.nnet.binary_crossentropy(p_real, T.ones(p_real.shape)).mean()
    d_cost_gen = T.nnet.binary_crossentropy(p_gen, T.zeros(p_gen.shape)).mean()

    # cost of gen : make the discr predict 1 for gen
    g_cost_d = T.nnet.binary_crossentropy(p_gen, T.ones(p_gen.shape)).mean()

    d_cost = d_cost_real + d_cost_gen
    g_cost = g_cost_d

    cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

    discrim_params = layers.get_all_params(out_discr, trainable=True)
    gen_params = layers.get_all_params(out_gen, trainable=True)

    d_updates = updates.adam(d_cost, discrim_params, learning_rate=lr, beta1=b1)
    g_updates = updates.adam(g_cost, gen_params, learning_rate=lr, beta1=b1)

    #d_updates = updates.rmsprop(d_cost, discrim_params, learning_rate=lr, rho=0.8)
    #g_updates = updates.rmsprop(g_cost, gen_params, learning_rate=lr, rho=0.8)

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
        for train_X in tqdm(iterate_minibatches(X, targets=None, batchsize=batch_size, shuffle=True)):
            if data_in_memory == False:
                train_X = rescale_input(train_X)
                train_X = preprocess_input(train_X)
            train_Z = floatX(rng.uniform(-1., 1., size=(len(train_X), z_dim)))
            if n_updates % 2 == 0:
                total_d_loss += train_d(train_X, train_Z)[1]
                nb_g_updates += 1
            else:
                total_g_loss += train_g(train_X, train_Z)[0]
                nb_d_updates += 1
            n_updates += 1
        stats = OrderedDict()
        stats['epoch'] = epoch
        stats['g_loss'] = total_g_loss / nb_g_updates
        stats['d_loss'] = total_d_loss / nb_d_updates
        stats['train_time'] = time() - t
        history.append(stats)

        print(tabulate([stats], 'keys'))

        if epoch % 5 == 0:
            nb_samples = 400
            sample_Z = floatX(rng.uniform(-1., 1., size=(nb_samples, z_dim)))
            sample_X = gen(sample_Z)
            sample_X = deprocess_input(sample_X)
            img = dispims(sample_X, border=1)
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
        #if epoch > 12:
        #   lr.set_value(floatX(np.array(lr.get_value() * 0.95)))
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

def load_model(filename):
    with open(filename, "r") as fd:
        data = dill.load(fd)
    builder = data['builder']
    builder_args = data['args']
    gen, discr = builder(**builder_args)
    layers.set_all_param_values(gen, data['generator_weights'])
    layers.set_all_param_values(discr, data['discrimimator_weights'])
    return gen, discr


@click.command()
@click.option('--run', default=0, help='nb jobs to run', required=False)
def hyperjob(run):
    from lightjob.cli import load_db
    from lightjob.db import SUCCESS, RUNNING, AVAILABLE, PENDING
    from lightjob.utils import summarize

    def js1():
        jobs = (
            dict(
                lr=lr,
                b1=b1,
                nb_epochs=100,
                model=dict(scale=scale,
                           num_filters_g=num_filters_g,
                           num_filters_d=num_filters_d,
                           start_w=4,
                           start_h=4,
                           filter_size=5,
                           do_batch_norm=True),
                subset_ratio=subset_ratio,
                dataset='mnist',
                w=32,
                h=32,
                c=1,
                model_name='dcgan')
            for scale in np.logspace(-4, -1, 5)
            for num_filters_g in [32, 64, 128]
            for num_filters_d in [4,  8, 16]
            for lr in (0.0002,)
            for b1 in (0.5,)
            for subset_ratio in (0.01, 0.1, 1)
        )
        return list(jobs)

    def insert_jobs():
        jobs = list(js1())
        nb = 0
        for content in jobs:
            s = summarize(content)
            outdir = 'results/{}'.format(s)
            nb += db.safe_add_job(content, outdir=outdir)
        return nb

    def run_jobs(nb=None):
        jobs = db.jobs_with(state=AVAILABLE)
        jobs = list(jobs)
        if nb is not None:
            jobs = jobs[0:nb]
        for j in jobs:
            db.modify_state_of(j['summary'], PENDING)
        print('starting to run')
        for j in jobs:
            db.modify_state_of(j['summary'], RUNNING)
            kw = j['content']
            kw['outdir'] = j['outdir']
            hist = train(**kw)
            db.update({'hist': hist}, j['summary'])
            db.modify_state_of(j['summary'], SUCCESS)

    db = load_db()
    nb = insert_jobs()
    print('nb jobs inserted : {}'.format(nb))
    run_jobs(nb=run)


if __name__ == '__main__':
    main.add_command(traincollection)
    main.add_command(hyperjob)
    main()
