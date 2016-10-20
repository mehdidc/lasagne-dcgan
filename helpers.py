from functools import partial
from skimage.transform import resize
from lasagne import init
import numpy as np
import os

import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
from lasagne.layers import Deconv2DLayer as DeconvLasagne
from lasagne.layers import batch_norm, Conv2DLayer
from lasagne.nonlinearities import linear

def floatX(x):
    return np.array(x).astype(theano.config.floatX)

def to_categorical(y, nb_classes=None):
    '''
    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with 
    categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def iterate_minibatches(inputs, targets=None, batchsize=128, shuffle=False):
    if targets is not None:
        assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if targets is not None:
            yield inputs[excerpt], targets[excerpt]
        else:
            yield inputs[excerpt]

def mkdir_path(path):
    if not os.access(path, os.F_OK):
        os.makedirs(path)

def dispims(M, border=0, bordercolor=[0.0, 0.0, 0.0], shape = None, normalize=False):
    """ Display an array of rgb images.
    The input array is assumed to have the shape numimages x numpixelsY x numpixelsX x 3
    """
    bordercolor = np.array(bordercolor)[None, None, :]
    numimages = len(M)
    M = M.copy()
    if normalize:
        for i in range(M.shape[0]):
            M[i] -= M[i].flatten().min()
            M[i] /= M[i].flatten().max()
    height, width, color = M[0].shape
    if color == 1:
        M = M[:, :, :, :] * np.ones((1, 1, 1, 3))
    if shape is None:
        n0 = np.int(np.ceil(np.sqrt(numimages)))
        n1 = np.int(np.ceil(np.sqrt(numimages)))
    else:
        n0 = shape[0]
        n1 = shape[1]

    im = np.array(bordercolor)*np.ones(
                             ((height+border)*n1+border,(width+border)*n0+border, 1),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < numimages:
                im[j*(height+border)+border:(j+1)*(height+border)+border,
                   i*(width+border)+border:(i+1)*(width+border)+border,:] = np.concatenate((
                  np.concatenate((M[i*n1+j,:,:,:],
                         bordercolor*np.ones((height,border,3),dtype=float)), 1),
                  bordercolor*np.ones((border,width+border,3),dtype=float)
                  ), 0)
    return im


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


class Deconv2DLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 func='me',
                 W=lasagne.init.Orthogonal(),
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(W,
                (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
                (num_filters,),
                name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity
        self.func = func

    def get_output_shape_for(self, input_shape):
        shape = tuple(i*s - 2*p + f - 1
                for i, s, p, f in zip(input_shape[2:],
                                      self.stride,
                                      self.pad,
                                      self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):

        if self.func == 'me':
            op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
                imshp=self.output_shape,
                kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
                subsample=self.stride, border_mode=self.pad)
            conved = op(self.W, input, self.output_shape[2:])
        elif self.func == 'alec':
            conved = deconv_alec(input, self.W, subsample=[self.stride]*2, border_mode=self.pad)
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)

def deconv_alec(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    """
    sets up dummy convolutional forward pass and uses its grad as deconv
    currently only tested/working with same padding
    """
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1]).shape, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img


def Deconv2DLayerScaler(incoming, num_filters, filter_size, stride=1, pad=0, nonlinearity=lasagne.nonlinearities.rectify, use_batch_norm=True, **kwargs):
    l = DeconvLasagne(incoming, num_filters=num_filters, filter_size=filter_size, stride=stride, nonlinearity=nonlinearity, **kwargs)
    if use_batch_norm:
        l = batch_norm(l)
    l = Conv2DLayer(
         l,
         #W=init.Constant(0.),
         #b=init.Constant(0.),
         num_filters=num_filters,
         filter_size=(filter_size[0] - 1, filter_size[1] - 1),
         nonlinearity=nonlinearity,
    )
    if use_batch_norm:
        l = batch_norm(l)
    #l.params[l.W] = set()
    #l.params[l.b] = set()
    return l

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

resize = partial(resize, preserve_range=True)
def resize_dataset(X, wh):
    w, h, c = X.shape[1:]
    ww, hh = wh
    # assumes X has shape (B, w, h, c) and returns (B, ww, hh, c)
    X_rescaled = np.empty((len(X), ww, hh, c))
    for i in range(len(X)):
        Xi = np.array(X[i])
        if len(Xi.shape) == 3:
            Xi = Xi[:, :, 0:c]
            X_rescaled[i] = resize(Xi, (ww, hh))
        else:
            X_rescaled[i, :, :, 0] = resize(Xi, (ww, hh))
    X_rescaled = floatX(X_rescaled) / X_rescaled.max()
    return X_rescaled
