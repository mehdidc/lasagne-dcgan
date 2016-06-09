import numpy as np
from collections import namedtuple

Dataset = namedtuple('Dataset', ('X', 'y'))

def load_data(name, **kw):
    from lasagnekit.datasets.mnist import MNIST
    from lasagnekit.datasets.fonts import Fonts
    from lasagnekit.datasets.insects import Insects
    from lasagnekit.datasets.rescaled import Rescaled

    if name == 'mnist':
        data = MNIST()
        data.load()
        data.X = data.X.reshape((data.X.shape[0], 1, 28, 28))
    if name == 'fonts':
        data = Fonts(labels_kind='letters')
        data.load()
        data.X = data.X.reshape((data.X.shape[0], 1, 64, 64))
    if name == 'fonts_28x28':
        data = Fonts(labels_kind='letters')
        data = Rescaled(data, (28, 28))
        data.load()
        data.X = data.X.reshape((data.X.shape[0], 1, 28, 28))
    if name == 'insects':
        data = Insects()
        data.load()
        data.X = data.X.reshape((data.X.shape[0], 64, 64, 3))
        data.X = data.X.transpose((0, 3, 1, 2))
    if name == 'chinese':
        import os
        import h5py
        DATA_PATH = os.getenv('DATA_PATH')
        filename = os.path.join(DATA_PATH, 'fonts_big', 'fonts.hdf5')
        hf = h5py.File(filename)
        X = HdfLambda(hf['trn/bitmap'], lambda x: x.transpose((0, 2, 3, 1)))
        y = hf['trn/tagcode']
        data = Dataset(X=X, y=y)
    if name == 'fonts_big':
        import h5py
        import os
        DATA_PATH = os.getenv('DATA_PATH')
        filename = os.path.join(DATA_PATH, 'fonts_big', 'fonts.hdf5')
        hf = h5py.File(filename)
        X = hf['fonts']
        X = HdfIterator(X, preprocess=lambda x: x[:, :, :, np.newaxis])
        data = Dataset(X=X, y=np.array([]))
    if name == 'fonts_big_multichannel':
        import h5py
        import os
        DATA_PATH = os.getenv('DATA_PATH')
        filename = os.path.join(DATA_PATH, 'fonts_big', 'fonts.hdf5')
        hf = h5py.File(filename)
        X = hf['fonts']
        X = HdfLambda(X, lambda x: x.transpose((0, 2, 3, 1)))
        data = Dataset(X=X, y=np.array([]))

    return split_data(data, **kw)


class HdfLambda(object):

    def __init__(self, X, fn=lambda x:x):
        self.X = X
        self.fn = fn
        self.shape = X.shape

    def __getitem__(self, key):
        x = self.X[key]
        x = self.fn(x)
        return x


class HdfIterator(object):

    def __init__(self, X, preprocess=lambda x:x):
        assert len(X.shape) >= 2
        self.X = X
        self.preprocess = preprocess

        rest = X.shape[2:]
        self.shape = (X.shape[0] * X.shape[1],) + rest

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, end = key.start, key.stop
            if end is None:
                end = self.X.shape[0] * self.X.shape[1]
            nb = end - start
            nb = min(nb, self.X.shape[0] * self.X.shape[1])
            x = self.X[start / self.X.shape[1]: end / self.X.shape[1]]
            rest = x.shape[2:]
            x = x.reshape((x.shape[0] * x.shape[1],) + rest)
            x = x[0:nb]
            x = self.preprocess(x)
            return x
        else:
            return self.X[key]


def split_data(data, training_subset=None , valid_subset=None, valid_ratio=0.16667, shuffle=True):
    train_full = data
    train, valid = split(train_full, test_size=valid_ratio, shuffle=shuffle)
    if training_subset is not None and training_subset < 1:
        nb = int(training_subset * len(train.X))
        print('training on a subset of training data of size : {}'.format(nb))
        train = Dataset(X=train.X[0:nb], y=train.y[0:nb])

    if valid_subset is not None and valid_subset < 1:
        nb = int(valid_subset * len(valid.X))
        print('validating on a subset of validation data of size : {}'.format(nb))
        valid = Dataset(X=valid.X[0:nb], y=valid.y[0:nb])
    return train, valid

def split(dataset, test_size=0.5, random_state=None, shuffle=False):
    if random_state is None:
        random_state = np.random.randint(0, 999999)
    nb = dataset.X.shape[0]
    nb_test = int(nb * test_size)
    nb_train = nb - nb_test
    rng = np.random.RandomState(random_state)
    indices = np.arange(0, nb)
    if shuffle:
        rng.shuffle(indices)
    X = dataset.X[0:nb_train]
    y = dataset.y[0:nb_train]
    dataset_train = Dataset(X=X, y=y)
    X = dataset.X[nb_train:]
    y = dataset.y[nb_train:]
    dataset_test = Dataset(X=X, y=y)
    return dataset_train, dataset_test
