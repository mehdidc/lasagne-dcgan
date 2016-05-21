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


@click.group()
def main():
    pass

@click.command()
@click.option('--outdir', default='', help='Out directory', required=False)
def train(outdir):
	from data import load_data
	import theano.tensor as T
	from time import time
	from skimage.io import imsave
	#l2 = 2.5e-5
	nb_outputs = 10
	c, w, h = 1, 28, 28
	lr_initial = 0.0002
	b1 = 0.5
	nb_epochs = 200

	z_dim = 100
	batch_size = 128
	nb_discr_updates = 1
	lr = theano.shared(floatX(np.array(lr_initial)))
	rng = np.random.RandomState(1234)

	def preprocess_input(X):
		#X = X - 0.5
		return X.reshape((X.shape[0], c, w, h))

	def preprocess_output(y):
		return floatX(to_categorical(y))
	
	# load data	
	data_train, data_valid, data_test = load_data('mnist')#, training_subset=0.01)
	data_train.X = preprocess_input(data_train.X)
	data_train.y = preprocess_output(data_train.y)

	# compile net
	Y = T.matrix()
	X_real = T.tensor4()
	Z = T.matrix()
	
	builder_args = dict(z_dim=z_dim, w=w, h=h, c=c, nb_outputs=nb_outputs)
	builder = model.cond_dcgan_28x28
	x_in, y_in, z_in, out_gen, out_discr = builder(**builder_args)
	X_gen = layers.get_output(out_gen, {z_in: Z, y_in: Y} )

	p_real = layers.get_output(out_discr, {x_in: X_real, y_in: Y} )

	p_gen = layers.get_output(out_discr, {x_in: X_gen, y_in: Y} )

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
	all_updates = d_updates.copy() 
	all_updates.update(g_updates)

	train_g = theano.function([X_real, Z, Y], cost, updates=g_updates)
	train_d = theano.function([X_real, Z, Y], cost, updates=d_updates)
	gen = theano.function([Z, Y], X_gen)

	# Train
	model_filename = os.path.join(outdir, 'model.pkl')
	save_model(builder, builder_args,
			   out_gen, out_discr, 
			   model_filename)
	mkdir_path(outdir)

	history = []
	n_updates = 0
	for epoch in range(1, nb_epochs + 1):
		total_g_loss = 0
		total_d_loss = 0
		nb_g_updates = 0
		nb_d_updates = 0
		t = time()
		for train_X, train_y in tqdm(iterate_minibatches(data_train.X, data_train.y, batch_size)):
			train_Z = floatX(rng.uniform(-1., 1., size=(len(train_X), z_dim)))
			if n_updates % (nb_discr_updates + 1) == 0:
				total_g_loss += (train_g(train_X, train_Z, train_y))[0]
				nb_g_updates += 1
			else:
				total_d_loss += (train_d(train_X, train_Z, train_y))[1]
				nb_d_updates += 1
			n_updates += 1
		stats = OrderedDict()
		stats['epoch'] = epoch
		stats['g_loss'] = total_g_loss / nb_g_updates
		stats['d_loss'] = total_d_loss / nb_d_updates
		stats['train_time'] = time() - t
		print(tabulate([stats], headers="keys"))
		history.append(stats)

		if epoch % 5 == 0:
			nb_samples_per_output = 40
			nb_samples = nb_outputs * nb_samples_per_output
			sample_Z = floatX(rng.uniform(-1., 1., size=(nb_samples, z_dim)))
			sample_Y = floatX(to_categorical(np.repeat(np.arange(0, nb_outputs), nb_samples_per_output)))
			sample_X = gen(sample_Z, sample_Y)
			img = dispims(sample_X.reshape((sample_X.shape[0], w, h, 1)), border=1)
			filename = os.path.join(outdir, 'samples{:05d}.png'.format(epoch))
			imsave(filename, img)
			save_model(builder, builder_args, out_gen, out_discr, model_filename)



@click.command()
@click.option('--outdir', default='.', help='Output directory', required=False)
@click.option('--pattern', default='*.png', help='Pattern of image filenames to train on', required=False)
@click.option('--model_name', default='dcgan_small', help='Model name', required=False)
@click.option('--w', default=64, help='rescale images to a width of w', required=False)
@click.option('--h', default=64, help='rescale images to a height of h', required=False)
@click.option('--c', default=3, help='1 if grayscale images otherwise 3', required=False)
@click.option('--data_in_memory', default=True, help='', required=False)
def traincollection(outdir, pattern, model_name, w, h, c, data_in_memory):
	import theano.tensor as T
	from time import time
	from skimage.io import imsave
	from skimage.io import imread_collection
	from skimage.transform import resize
	w = int(w)
	h = int(h)
	c = int(c)
	# assume w and h are power of two

	lr_initial = 0.00002
	b1 = 0.5
	l2 = 1e-5
	nb_epochs = 1000

	z_dim = 100
	batch_size = 128
	nb_discr_updates = 1
	lr = theano.shared(floatX(np.array(lr_initial)))
	rng = np.random.RandomState(1234)

	def resize_input(X, wh):
		w, h  = wh
		#if w < X.shape[0] and h < X.shape[1]:
		#	#crop
		#	sx = (X.shape[0] - w) / 2
		#	sy = (X.shape[1] - h) / 2
		#	return X[sx:sx+w, sy:sy+h]
		#else:
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
		#X_rescaled = X_rescaled * 2 - 1
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
	X = imread_collection(pattern)
	if data_in_memory == True:
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
	mkdir_path(outdir)

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
		print(tabulate([stats], headers="keys"))
		history.append(stats)

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
		#if epoch > 12:
		#	lr.set_value(floatX(np.array(lr.get_value() * 0.95)))

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

if __name__ == '__main__':
	main.add_command(train)
	main.add_command(traincollection)
	main()