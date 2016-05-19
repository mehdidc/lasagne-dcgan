
from lasagne import layers
import theano
import theano.tensor as T

DenseCondConcat = layers.ConcatLayer


class ConvCondConcat(layers.MergeLayer):

    def __init__(self, incomings, **kwargs):
        super(ConvCondConcat, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        xshape, yshape = input_shapes
        shape = xshape[0], xshape[1] + yshape[1], xshape[2], xshape[3]
        return shape

    def get_output_for(self, inputs, **kwargs):
        x, y = inputs
        return T.concatenate([x, y.dimshuffle(0, 1, 'x', 'x') * T.ones((x.shape[0], y.shape[1], x.shape[2], x.shape[3]))], axis=1)

class Deconv2DLayer(layers.Conv2DLayer):
    
    def __init__(self, incoming, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(inv_conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, pad)))

    #def get_W_shape(self):
    #    shape = super(Deconv2DLayer, self).get_W_shape()
    #    return (shape[1], shape[0]) + shape[2:]

    def convolve(self, input, **kwargs):
        shape = self.get_output_shape_for(input.shape)
        fake_output = T.alloc(0., *shape)
        border_mode = 'half' if self.pad == 'same' else self.pad
        
        w_shape = self.get_W_shape()
        w_shape = (w_shape[1], w_shape[0]) + w_shape[2:]
        shape = self.get_output_shape_for(self.input_layer.output_shape)
        W = self.W.transpose((1, 0, 2, 3))

        conved = self.convolution(fake_output, W,
                                  shape, w_shape,
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        return theano.grad(None, wrt=fake_output, known_grads={conved: input})

def deconv_same(X, **kw):
    X = Deconv2DLayer(X, **kw)
    X = layers.Conv2DLayer(
    )


def inv_conv_output_length(input_length, filter_size, stride, pad=0):
    if input_length is None:
        return None
    if pad == 'full':
        output_length = (input_length + 1) * stride + filter_size
    elif pad == 'valid':
        output_length = (input_length - 1) * stride + filter_size
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = (input_length + 2 * pad - 1) * stride + filter_size
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))
    return output_length


class Depool2DLayer(Deconv2DLayer):

    def __init__(self, incoming, pool_size=(2, 2),
                 pool_stride=(2, 2), **kwargs):
        super(Depool2DLayer, self).__init__(
                incoming,
                num_filters=incoming.output_shape[1],
                filter_size=pool_size,
                stride=pool_stride,
                **kwargs)