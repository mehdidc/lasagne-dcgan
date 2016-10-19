
from lasagne import layers, init, nonlinearities
import theano
import theano.tensor as T
#from sparsemax_theano import sparsemax
import numpy as np
from collections import defaultdict

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
                 w_left_pad=0,
                 w_right_pad=0,
                 h_left_pad=0,
                 h_right_pad=0,
                 color_min=0,
                 color_max=1,
                 stride_normalize=False,
                 eps=0,
                 learn_patches=False,
                 coords='continuous',
                 **kwargs):
        """
        w : width of resulting image
        h : height of resulting image
        patches : (nb_patches, color, ph, pw)
        col : 'grayscale'/'rgb' or give the nb of channels as an int
        n_steps : int, nb of time steps
        return_seq : if True returns the seq (nb_examples, n_steps, c, h, w)
                     if False returns (nb_examples, -1, c, h, w)
        reduce_func : function used to update the output, takes prev
                      output as first argument and new output
                      as second one.
        normalize_func : if a function, it is used to normalize between 0 and 1 for :
                            - coordinates
                            - stride if stride=='predicted' (for x and y)
                            - sigma if sigma=='predicted' (for x and y)
                            - color if it is ndarray
                            - color if color=='predicted'. 
                         if a dict, then specify functions separately:
                            {'coords': ..., 'stride': ..., 'sigma': ..., 'color': }
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
                if ndarray, then we have a discrete number of learned colors,
                and the array represents the initial colors and its shape
                is (nb_colors, nb_col_channels).
                
                otherwise it should be a number if col is 'grayscale' or
                a 3-tuple if col is 'rgb' and then then same color is
                merged to the patches at all time steps. 

        x_min : the minimum value for the coords in the w scale
        x_max : if 'width' it is equal to w, else use the provided value

        y_min : the minimum value for the coords in the w scale
        y_max : if 'height' it is equal to h, else use the provided value
        
        w_left_pad  : int/'half_patch'.
                      augment virtually the resulting image with padding to take into account pixels outside
                      the image to have proper normalization of Fx.
                      if 'half_patch', then the padding is the half of the patch width so that a coordinate
                      of 0, 0 with x_min=0 and x_max='width' will show the bottom right quarter of the patch
        w_right_pad : same than w_left_pad but right of the image
        h_left_pad  : like w_left_pad but for height
        h_right_pad :  like w_right_pad but for height

        color_min : min val of color. this and color_max can be helpful to implement negative colors, negative colors can be used
                    to predicted delta color instead of color so that when canvas are summed up something like opacity could be implemented
                    . for instance if we have two overlapping brushes (first is bigger than second) and we want want with color [1 0 0] and the second
                    [0 1 0], what we can do is to predict the color [1 0 0] for the first and the color [-1 1 0] for the second so that the red
                    component is cancelled.
        color_max : max val of color
        stride_normalize : if True multiply Fx by stride_x and Fy by stride_y, this is useful when summing canvas
                           which has different strides, stride_nornalize makes canvas of different stride on the same scale.
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
        if not isinstance(normalize_func, dict):
            self.normalize_func = defaultdict(lambda:normalize_func)
        else:
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
        self.learn_patches = learn_patches
        self.w_left_pad = w_left_pad
        self.w_right_pad = w_right_pad
        self.h_left_pad = h_left_pad
        self.h_right_pad = h_right_pad
        self.color_min = color_min
        self.color_max = color_max
        self.stride_normalize = stride_normalize
        self.coords = coords

        if learn_patches:
            if isinstance(self.patches, np.ndarray):
                shape = self.patches.shape
            else:
                shape = self.patches.get_value().shape
            assert shape[1] == self.nb_col_channels
            self.ph, self.pw = shape[2:]
            self.patches_ = self.add_param(self.patches, shape, name="patches")
        else:
            if isinstance(self.patches, np.ndarray):
                shape = self.patches.shape
            else:
                shape = self.patches.get_value().shape
            assert shape[1] == self.nb_col_channels
            self.ph, self.pw = shape[2:]
            self.patches_ = theano.shared(self.patches)

        if isinstance(self.color, np.ndarray):
            assert self.color.shape[1] == self.nb_col_channels
            self.colors_ = self.add_param(self.color, self.color.shape, name="colors")
        elif isinstance(self.color, theano.compile.SharedVariable):
            assert self.color.get_value().shape[1] == self.nb_col_channels
            self.colors_ = self.add_param(self.color, self.color.get_value().shape, name="colors")
        self.eps = eps
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
        ph = self.ph
        pw = self.pw
        nb_features = self._nb_input_features
        pointer = 0
        if self.coords == 'continuous':
            gx, gy = X[:, 0], X[:, 1]
            gx = self.normalize_func['coords'](gx) * (self.x_max - self.x_min) + self.x_min
            gy = self.normalize_func['coords'](gy) * (self.y_max - self.y_min) + self.y_min
            self.assign_['gx'] = 0
            self.assign_['gy'] = 1
            pointer += 2
        elif self.coords == 'discrete':
            nx = self.x_max - self.x_min
            cx = theano.shared(np.linspace(0, 1, nx).astype(np.float32))
            gx_pr = X[:, pointer:pointer + nx]
            gx_pr = self.to_proba_func(gx_pr)
            gx = T.dot(gx_pr, cx)
            gx = gx * (self.x_max - self.x_min) + self.x_min
            self.assign_['gx'] = (pointer, pointer + nx)
            pointer += nx
            ny = self.y_max - self.y_min
            cy = theano.shared(np.linspace(0, 1, ny).astype(np.float32))
            gy_pr = X[:, pointer:pointer + ny]
            gy_pr = self.to_proba_func(gy_pr)
            gy = T.dot(gy_pr, cy)
            gy = gy * (self.y_max - self.y_min) + self.y_min
            self.assign_['gy'] = (pointer, pointer + ny)
            pointer += ny
        else:
            raise Exception('invalid value : {} for coords'.format(self.coords))
        if self.x_stride == 'predicted':
            sx = X[:, pointer]
            sx = self.normalize_func['stride'](sx)
            self.assign_['x_stride'] = pointer
            pointer += 1
        elif type(self.x_stride) == list:
            xs = (np.array(self.x_stride).astype(np.float32))
            xs_pr = X[:, pointer:pointer + len(xs)]
            xs_pr = self.to_proba_func(xs_pr)
            sx = T.dot(xs_pr, xs)
            self.assign_['x_stride'] = (pointer, pointer + len(xs))
            pointer += len(xs)
        else:
            sx = T.ones_like(gx) * self.x_stride

        if self.y_stride == 'predicted':
            sy = X[:, pointer]
            sy = self.normalize_func['stride'](sy)
            self.assign_['y_stride'] = pointer
            pointer += 1
        elif type(self.y_stride) == list:
            ys = (np.array(self.y_stride).astype(np.float32))
            ys_pr = X[:, pointer:pointer + len(ys)]
            ys_pr = self.to_proba_func(ys_pr)
            sy = T.dot(ys_pr, ys)
            self.assign_['y_stride'] = (pointer, pointer + len(ys))
            pointer += len(ys)
        else:
            sy = T.ones_like(gy) * self.y_stride

        if self.x_sigma == 'predicted':
            log_x_sigma = X[:, pointer]
            x_sigma = T.exp(log_x_sigma)
            x_sigma = self.normalize_func['sigma'](log_x_sigma) * pw
            self.assign_['x_sigma'] = pointer
            pointer += 1
        elif type(self.x_sigma) == list:
            xs = (np.array(self.x_sigma).astype(np.float32))
            xs_pr = X[:, pointer:pointer + len(xs)]
            xs_pr = self.to_proba_func(xs_pr) * xs
            xs_pr = xs_pr.sum(axis=1)
            x_sigma = xs_pr
            self.assign_['x_sigma'] = (pointer, pointer + len(xs))
            pointer += len(xs)
        else:
            x_sigma = T.ones_like(gx) * self.x_sigma

        if self.y_sigma == 'predicted':
            log_y_sigma = X[:, pointer]
            y_sigma = T.exp(log_y_sigma)
            y_sigma = self.normalize_func['sigma'](log_y_sigma) * ph
            self.assign_['y_sigma'] = pointer
            pointer += 1
        elif type(self.y_sigma) == list:
            ys = (np.array(self.y_sigma).astype(np.float32))
            ys_pr = X[:, pointer:pointer + len(ys)]
            ys_pr = self.to_proba_func(ys_pr) * ys
            ys_pr = ys_pr.sum(axis=1)
            y_sigma = ys_pr
            self.assign_['y_sigma'] = (pointer, pointer + len(ys))
            pointer += len(ys)
        else:
            y_sigma = T.ones_like(gy) * self.y_sigma

        if self.patch_index == 'predicted':
            patch_index = X[:, pointer:pointer + nb_patches]
            self.assign_['patch_index'] = (pointer, pointer + nb_patches)
            pointer += nb_patches
        else:
            patch_index = self.patch_index
        if isinstance(self.color, np.ndarray) or isinstance(self.color, theano.compile.SharedVariable):
            if isinstance(self.color, theano.compile.SharedVariable):
                shape = self.color.get_value().shape
            else:
                shape = self.color.shape
            nb = shape[0]
            colors_pr = X[:, pointer:pointer + nb]#(nb_examples, nb_colors)
            colors_pr = self.to_proba_func(colors_pr) # (nb_examples, nb_colors)
            colors_mix = colors_pr.dimshuffle(0, 1, 'x') * self.colors_.dimshuffle('x', 0, 1) #(nb_examples, nb_colors, 1) * (1, nb_colors, nb_col_channels) = (nb_examples, nb_colors, nb_col_channels)
            colors = colors_mix.sum(axis=1) #(nb_examples, nb_col_channels)
            colors = self.normalize_func['color'](colors) * (self.color_max - self.color_min) + self.color_min
            self.assign_['color'] = (pointer, pointer + nb)
            pointer += nb
        elif self.color == 'predicted':
            colors = X[:, pointer:pointer + self.nb_col_channels]
            colors = self.normalize_func['color'](colors) * (self.color_max - self.color_min) + self.color_min
            self.assign_['color'] = (pointer, pointer + self.nb_col_channels)
            pointer += self.nb_col_channels
        elif self.color == 'patches':
            colors = T.ones((1, 1, 1, 1))
        else:
            assert len(self.color) == self.nb_col_channels
            colors = self.color

        assert nb_features >= pointer, "The number of input features to Brush should be {} instead of {} (or at least bigger)".format(pointer, nb_features)
        
        if self.w_left_pad and self.w_right_pad:
            w_left_pad = self.w_left_pad
            if w_left_pad == 'half_patch':
                w_left_pad = pw / 2
            w_right_pad = self.w_right_pad
            if w_right_pad == 'half_patch':
                w_right_pad = pw / 2
            a, _ = np.indices((w + w_left_pad + w_right_pad, pw)) - w_left_pad
        else:
            w_left_pad = 0
            w_right_pad = 0
            a, _ = np.indices((w, pw))

        a = a.astype(np.float32)
        a = a.T
        a = theano.shared(a)

        if self.w_left_pad and self.w_right_pad:
            h_left_pad = self.h_left_pad
            if h_left_pad == 'half_patch':
                h_left_pad = ph / 2
            h_right_pad = self.h_right_pad
            if h_right_pad == 'half_patch':
                h_right_pad = ph / 2
            b, _ = np.indices((h + h_left_pad + h_right_pad, pw)) - h_left_pad
        else:
            h_left_pad = 0
            h_right_pad = 0
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
        if self.stride_normalize:
            Fx = Fx * sx.dimshuffle(0, 'x', 'x')
        if w_left_pad and w_right_pad:
            Fx = Fx[:, :, w_left_pad:-w_right_pad]
        uy = (gy.dimshuffle(0, 'x') +
              (T.arange(1, ph + 1) - ph/2. - 0.5) * sy.dimshuffle(0, 'x'))
        # shape of uy : (nb_examples, ph)
        b_ = b.dimshuffle('x', 0, 1)
        uy_ = uy.dimshuffle(0, 1, 'x')
        Fy = T.exp(-(b_ - uy_) ** 2 / (2 * y_sigma_ ** 2))
        # shape of Fy : (nb_examples, ph, h)
        Fy = Fy / (Fy.sum(axis=2, keepdims=True) + self.eps)
        if self.stride_normalize:
            Fy = Fy * sy.dimshuffle(0, 'x', 'x')
        if h_left_pad and h_right_pad:
            Fy = Fy[:, :, h_left_pad:-h_right_pad]
        
        patches = self.patches_
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

        if isinstance(self.color, np.ndarray) or isinstance(self.color, theano.compile.SharedVariable):
            o = o * colors.dimshuffle(0, 1, 'x', 'x')
        elif self.color == 'predicted':
            o = o * colors.dimshuffle(0, 1, 'x', 'x')
        elif self.color == 'patches':
            pass
        else:
            colors_ = theano.shared(np.array(colors).astype(theano.config.floatX))
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
        # the above single line is to avoid this error:
        # "an input and an output are associated with the same recurrent state
        # and should have the same type but have type 'CudaNdarrayType(float32,
        # (False, True, False, False))' and 'CudaNdarrayType(float32, 4D)'
        # respectively.'))"
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

def one_step_brush_layer(*args, **kwargs):
    return GenericBrushLayer(n_steps=1, return_seq=False, *args, **kwargs)[:, 0, :, :, :]


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


class TensorLayer(layers.Layer):
    def __init__(self,
                 incoming,
                 num_units,
                 W=init.GlorotUniform(),
                 **kwargs):
        """
        num_units : tuple describing the shape of the desired tensor W to multiply
                    with each example vector.
        performs the tensor multiplication X * W where X has shape (nb_examples, A)
        and W has shape (A, B, C, D...) and returns an output with shape
        (nb_examples, B, C, D...)
        """
        super(TensorLayer, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        num_inputs = np.prod(self.input_shape[1:])
        self.W = self.add_param(W, (num_inputs,) + num_units, name="W")

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],) + self.num_units

    def get_output_for(self, input, **kwargs):
        input = input.flatten(2)
        activation = T.tensordot(input, self.W, axes=[1, 0])
        return activation

class LNGRULayer(layers.MergeLayer):
    r"""
    .. math ::
        r_t = \sigma_r(LN(x_t, W_{xr}; \gamma_{xr}, \beta_{xr}) + LN(h_{t - 1}, W_{hr}; \gamma_{xr}, \beta_{xr}) + b_r)\\ \
        u_t = \sigma_u(LN(x_t, W_{xu}; \gamma_{xu}, \beta_{xu}) + LN(h_{t - 1}, W_{hu}; \gamma_{xu}, \beta_{xu})+ b_u)\\ \
        c_t = \sigma_c(LN(x_t, W_{xc}; \gamma_{xc}, \beta_{xc}) + r_t \odot (LN(h_{t - 1}, W_{hc}); \gamma_{xc}, \beta_{xc}) + b_c)\\ \
        h_t = (1 - u_t) \odot h_{t - 1} + u_t \odot c_t \

    Notes
    -----

    .. math::
        LN(z;\alpha, \beta) = \frac{(z-\mu)}{\sigma} \odot \alpha + \beta

    """
    def __init__(self, incoming, num_units,
                 resetgate=layers.Gate(W_cell=None),
                 updategate=layers.Gate(W_cell=None),
                 hidden_update=layers.Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 alpha_init=init.Constant(1.0),
                 beta_init=init.Constant(0.0),
                 normalize_hidden_update=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, layers.Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(LNGRULayer, self).__init__(incomings, **kwargs)

        # # If the provided nonlinearity is None, make it linear
        # if nonlinearity is None:
        #     self.nonlinearity = nonlinearities.identity
        # else:
        #     self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.normalize_hidden_update = normalize_hidden_update
        self._eps = 1e-5

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    self.add_param(self.alpha_init, (num_units,),
                                   name="alpha_in_to_{}".format(gate_name)),
                    self.add_param(self.beta_init, (num_units,),
                                   name="beta_in_to_{}".format(gate_name),
                                   regularizable=False),
                    self.add_param(self.alpha_init, (num_units,),
                                   name="alpha_hid_to_{}".format(gate_name)),
                    self.add_param(self.beta_init, (num_units,),
                                   name="beta_hid_to_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.alpha_in_to_updategate, self.beta_in_to_updategate,
         self.alpha_hid_to_updategate, self.beta_hid_to_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate, 'updategate')

        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.alpha_in_to_resetgate, self.beta_in_to_resetgate,
         self.alpha_hid_to_resetgate, self.beta_hid_to_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update, self.b_hidden_update,
         self.alpha_in_to_hidden_update, self.beta_in_to_hidden_update,
         self.alpha_hid_to_hidden_update, self.beta_hid_to_hidden_update,
         self.nonlinearity_hidden_update) = add_gate_params(hidden_update, 'hidden_update')

        # Initialize hidden state
        if isinstance(hid_init, layers.Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

        # parameters for Layer Normalization of the cell gate
        if self.normalize_hidden_update:
            self.alpha_hidden_update = self.add_param(
                self.alpha_init, (num_units, ),
                name="alpha_hidden_update")
            self.beta_hidden_update = self.add_param(
                self.beta_init, (num_units, ),
                name="beta_hidden_update", regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    # Layer Normalization
    def __ln__(self, z, alpha, beta):
        output = (z - z.mean(-1, keepdims=True)) / T.sqrt(z.var(-1, keepdims=True) + self._eps)
        output = alpha * output + beta
        return output


    def __gru_fun__(self, inputs, **kwargs):
        """
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Stack alphas for input into a (3*num_units) vector
        alpha_in_stacked = T.concatenate(
            [self.alpha_in_to_resetgate, self.alpha_in_to_updategate,
             self.alpha_in_to_hidden_update], axis=0)

        # Stack betas for input into a (3*num_units) vector
        beta_in_stacked = T.concatenate(
            [self.beta_in_to_resetgate, self.beta_in_to_updategate,
             self.beta_in_to_hidden_update], axis=0)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack alphas for hidden into a (3*num_units) vector
        alpha_hid_stacked = T.concatenate(
            [self.alpha_hid_to_resetgate, self.alpha_hid_to_updategate,
             self.alpha_hid_to_hidden_update], axis=0)

        # Stack betas for hidden into a (3*num_units) vector
        beta_hid_stacked = T.concatenate(
            [self.beta_hid_to_resetgate, self.beta_hid_to_updategate,
             self.beta_hid_to_hidden_update], axis=0)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            big_ones = T.ones((seq_len, num_batch, 1))
            input = T.dot(input, W_in_stacked)
            input = self.__ln__(input,
                                T.dot(big_ones, alpha_in_stacked.dimshuffle('x', 0)),
                                beta_in_stacked) + b_stacked

        ones = T.ones((num_batch, 1))
        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous, *args):
            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked)
                input_n = self.__ln__(input_n,
                                      T.dot(ones, alpha_in_stacked.dimshuffle('x', 0)),
                                      beta_in_stacked) + b_stacked

            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)
            hid_input = self.__ln__(hid_input,
                                    T.dot(ones, alpha_hid_stacked.dimshuffle('x', 0)),
                                    beta_hid_stacked)

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            if self.grad_clipping:
                resetgate = theano.gradient.grad_clip(
                    resetgate, -self.grad_clipping, self.grad_clipping)
                updategate = theano.gradient.grad_clip(
                    updategate, -self.grad_clipping, self.grad_clipping)

            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate*hidden_update_hid
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hidden_update(hidden_update)

            if self.normalize_hidden_update:
                hidden_update = self.__ln__(hidden_update,
                                   T.dot(ones, self.alpha_hidden_update.dimshuffle('x', 0)),
                                   self.beta_hidden_update)
            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate)*hid_previous + updategate*hidden_update
            return hid

        def step_masked(input_n, mask_n, hid_previous, *args):
            hid = step(input_n, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = T.switch(mask_n, hid, hid_previous)

            return hid

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, alpha_hid_stacked, beta_hid_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked, alpha_in_stacked, beta_in_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = theano.unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=False)[0]

        return hid_out


    def get_output_for(self, inputs, **kwargs):
        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        hid_out = self.__gru_fun__(inputs, **kwargs)
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out
