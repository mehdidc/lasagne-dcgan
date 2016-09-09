
from lasagne import layers, init, nonlinearities
import theano
import theano.tensor as T
#from sparsemax_theano import sparsemax
import numpy as np

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

def recurrent_accumulation(X, apply_func, reduce_func,
                           init_val, n_steps, **scan_kwargs):

    def step_function(input_cur, output_prev):
        return reduce_func(apply_func(input_cur), output_prev)

    sequences = [X]
    outputs_info = [init_val]
    non_sequences = []

    result, updates = theano.scan(fn=step_function,
                                  sequences=sequences,
                                  outputs_info=outputs_info,
                                  non_sequences=non_sequences,
                                  strict=False,
                                  n_steps=n_steps,
                                  **scan_kwargs)
    return result, updates


class Repeat(layers.Layer):
    def __init__(self, incoming, n, **kwargs):
        super(Repeat, self).__init__(incoming, **kwargs)
        self.n = n

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0], self.n] + list(input_shape[1:]))

    def get_output_for(self, input, **kwargs):
        tensors = [input]*self.n
        stacked = theano.tensor.stack(*tensors)
        dim = [1, 0] + range(2, input.ndim + 1)
        return stacked.dimshuffle(dim)


def over_op(prev, new):
    prev = (prev)
    new = (new)
    return prev + new * (1 - prev)


def correct_over_op(alpha):
    def fn(prev, new):
        return (prev * (1 - alpha) + new) / (2 - alpha)
    return fn


def max_op(prev, new):
    return T.maximum(prev, new)


def thresh_op(theta):
    def fn(prev, new):
        return (new > theta) * new +  (new <= theta ) * prev
    return fn


def sum_op(prev, new):
    # fix this
    return prev + new


def axis_softmax(x, axis=1):
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out




class GenericBrushLayer(layers.Layer):

    def __init__(self, incoming, w, h,
                 patches=np.ones((1, 1, 3, 3)),
                 col='grayscale',
                 n_steps=10,
                 return_seq=False,
                 reduce_func=sum_op,
                 to_proba_func=T.nnet.softmax,
                 normalize_func=T.nnet.sigmoid,
                 x_sigma='predicted',
                 y_sigma='predicted',
                 x_stride='predicted',
                 y_stride='predicted',
                 patch_index='predicted',
                 color='predicted',
                 x_min=0,
                 x_max='width',
                 y_min=0,
                 y_max='height',
                 eps=0,
                 **kwargs):
        """
        w : width of resulting image
        h : height of resulting image
        patches : (nb_patches, color, ph, pw)
        col : 'grayscale'/'rgb' or give the nb of channels as an int
        n_steps : int
        return_seq : True returns the seq (nb_examples, n_steps, c, h, w)
                    False returns (nb_examples, -1, c, h, w)
        reduce_func : function used to update the output, takes prev
                      output as first argument and new output
                      as second one.
        normalize_func : function used to normalize between 0 and 1
        x_sigma : if 'predicted' taken from input else use the provided
                  value
        y_sigma : if 'predicted' taken from input else use the provided
                  value
        x_stride : if 'predicted' taken from input else use the provided
                   value
        y_stride : if 'predicted' taken from input else use the provided
                   value
        patch_index: if 'predicted' taken from input then apply to_proba_func to
                     obtain probabilities, otherwise it is an int
                     which denotes the index of the chosen patch, that is,
                     patches[patch_index]
        color : if 'predicted' taken from input then merge to patch colors.
                if 'patch' then use patch colors only.
                otherwise it should be a number if col is 'grayscale' or
                a 3-tuple if col is 'rgb' and then then same color is
                merged to the patches at all time steps.
        x_min : the minimum value for the coords in the w scale
        x_max : if 'width' it is equal to w, else use the provided value

        y_min : the minimum value for the coords in the w scale
        y_max : if 'height' it is equal to h, else use the provided value

        """
        super(GenericBrushLayer, self).__init__(incoming, **kwargs)
        self.incoming = incoming
        self.w = w
        self.h = h
        self.nb_col_channels = (3 if col == 'rgb' else
                                1 if col == 'grayscale'
                                else col)
        assert self.nb_col_channels in (1, 3)
        self.n_steps = n_steps
        self.patches = patches
        self.return_seq = return_seq

        self.reduce_func = reduce_func
        self.normalize_func = normalize_func
        self.to_proba_func = to_proba_func
        self.x_sigma = x_sigma
        self.y_sigma = y_sigma
        self.x_stride = x_stride
        self.y_stride = y_stride
        self.x_min = x_min
        self.x_max = w if x_max == 'width' else x_max
        self.y_min = y_min
        self.y_max = h if y_max == 'height' else y_max
        self.patch_index = patch_index
        self.color = color
        self.eps = 0

        self._nb_input_features = incoming.output_shape[2]
        self.assign_ = {}

    def get_output_shape_for(self, input_shape):
        if self.return_seq:
            return (input_shape[0], self.n_steps,
                    self.nb_col_channels, self.w, self.h)
        else:
            return (input_shape[0], self.nb_col_channels, self.w, self.h)

    def apply_func(self, X):
        w = self.w
        h = self.h
        nb_patches = self.patches.shape[0]
        ph = self.patches.shape[2]
        pw = self.patches.shape[3]
        nb_features = self._nb_input_features

        gx, gy = X[:, 0], X[:, 1]

        gx = self.normalize_func(gx) * self.x_max + self.x_min
        gy = self.normalize_func(gy) * self.y_max + self.y_min

        pointer = 2
        if self.x_stride == 'predicted':
            sx = X[:, pointer]
            sx = self.normalize_func(gx) * self.x_max + self.x_min
            self.assign_['x_stride'] = pointer
            pointer += 1
        else:
            sx = T.ones_like(gx) * self.x_stride

        if self.y_stride == 'predicted':
            sy = X[:, pointer]
            sy = self.normalize_func(gy) * self.y_max + self.y_min
            self.assign_['y_stride'] = pointer
            pointer += 1
        else:
            sy = T.ones_like(gy) * self.y_stride

        if self.x_sigma == 'predicted':
            log_x_sigma = X[:, pointer]
            x_sigma = T.exp(log_x_sigma)
            self.assign_['x_sigma'] = pointer
            pointer += 1
        else:
            x_sigma = T.ones_like(gx) * self.x_sigma

        if self.y_sigma == 'predicted':
            log_y_sigma = X[:, pointer]
            y_sigma = T.exp(log_y_sigma)
            self.assign_['y_sigma'] = pointer
            pointer += 1
        else:
            y_sigma = T.ones_like(gy) * self.y_sigma

        if self.patch_index == 'predicted':
            patch_index = X[:, pointer:pointer + nb_patches]
            self.assign_['patch_index'] = (pointer, pointer + nb_patches)
            pointer += nb_patches
        else:
            patch_index = self.patch_index

        if self.color == 'predicted':
            colors = X[:, pointer:pointer + self.nb_col_channels]
            colors = self.normalize_func(colors)
            self.assign_['color'] = (pointer, pointer + self.nb_col_channels)
            pointer += self.nb_col_channels
        elif self.color == 'patches':
            colors = theano.shared(T.ones((1, 1, 1, 1)))
        else:
            colors = self.color

        assert nb_features >= pointer, "The number of input features to Brush should be {} insteaf of {} (or at least bigger)".format(pointer, nb_features)

        a, _ = np.indices((w, pw))
        a = a.astype(np.float32)
        a = a.T
        a = theano.shared(a)
        b, _ = np.indices((h, ph))
        b = b.astype(np.float32)
        b = b.T
        b = theano.shared(b)
        # shape of a (pw, w)
        # shape of b (ph, h)
        # shape of sx : (nb_examples,)
        # shape of sy : (nb_examples,)
        ux = (gx.dimshuffle(0, 'x') +
              (T.arange(1, pw + 1) - pw/2. - 0.5) * sx.dimshuffle(0, 'x'))
        # shape of ux : (nb_examples, pw)
        a_ = a.dimshuffle('x', 0, 1)
        ux_ = ux.dimshuffle(0, 1, 'x')

        x_sigma_ = x_sigma.dimshuffle(0, 'x', 'x')
        y_sigma_ = y_sigma.dimshuffle(0, 'x', 'x')

        Fx = T.exp(-(a_ - ux_) ** 2 / (2 * x_sigma_ ** 2))
        Fx = Fx / (Fx.sum(axis=2, keepdims=True) + self.eps)

        uy = (gy.dimshuffle(0, 'x') +
              (T.arange(1, ph + 1) - ph/2. - 0.5) * sy.dimshuffle(0, 'x'))
        # shape of uy : (nb_examples, ph)
        b_ = b.dimshuffle('x', 0, 1)
        uy_ = uy.dimshuffle(0, 1, 'x')
        Fy = T.exp(-(b_ - uy_) ** 2 / (2 * y_sigma_ ** 2))
        # shape of Fy : (nb_examples, ph, h)
        Fy = Fy / (Fy.sum(axis=2, keepdims=True) + self.eps)
        patches = theano.shared(self.patches)
        # patches : (nbp, c, ph, pw)
        # Fy : (nb_examples, ph, h)
        # Fx : (nb_examples, pw, w)
        o = T.tensordot(patches, Fy, axes=[2, 1])
        # -> shape (nbp, c, pw, nb_examples, h)
        o = o.transpose((3, 0, 1, 4, 2))
        # -> shape (nb_examples, nbp, c, h, pw)
        o = T.batched_tensordot(o, Fx, axes=[4, 1])
        # -> shape (nb_examples, nbp, c, h, w)

        if self.patch_index == 'predicted':
            patch_index_ = self.to_proba_func(patch_index)
            patch_index_ = patch_index_.dimshuffle(0, 1, 'x', 'x', 'x')
            o = o * patch_index_
            o = o.sum(axis=1)
            # -> shape (nb_examples, c, h, w)
        else:
            o = o[:, patch_index]
            # -> shape (nb_examples, c, h, w)

        if self.color == 'predicted':
            o = o * colors.dimshuffle(0, 1, 'x', 'x')
        elif self.color == 'patches':
            pass
        else:
            colors_ = theano.shared(np.array(colors))
            colors_ = colors_.dimshuffle('x', 0, 'x', 'x')
            o = o * colors_
        return o

    def reduce_func(self, prev, new):
        return self.reduce_func(prev, new)

    def get_output_for(self, input, **kwargs):
        output_shape = (
            (input.shape[0],) +
            (self.nb_col_channels, self.h, self.w))
        init_val = T.zeros(output_shape)
        init_val = T.unbroadcast(init_val, 0, 1, 2, 3)
        output, _ = recurrent_accumulation(
            # 'time' should be the first dimension
            input.dimshuffle(1, 0, 2),
            self.apply_func,
            self.reduce_func,
            init_val,
            self.n_steps)
        output = output.dimshuffle(1, 0, 2, 3, 4)
        if self.return_seq:
            return output
        else:
            return output[:, -1]


class TensorDenseLayer(layers.Layer):
    """
    used to perform embeddings on arbitray input tensor
    X : ([0], [1], ...,  T)
    W : (T, E) where E is the embedding size and T is last dim input size
    returns tensordot(X, W) + b which is : ([0], [1], ..., E)
    """
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(TensorDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units
        num_inputs = self.input_shape[-1]
        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape[0:-1] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        activation = T.tensordot(input, self.W, axes=[(input.ndim - 1,), (0,)])
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)
