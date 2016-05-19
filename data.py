from lasagnekit.datasets.mnist import MNIST
from lasagnekit.datasets.helpers import split

def load_data(name, **kw):
    if name == 'mnist':
        return load_mnist(**kw)

def load_mnist(training_subset=None , valid_subset=None, test_subset=None, valid_ratio=0.16667):
    c, w, h = 1, 28, 28
    def preprocess(data):
        data = data * 2 - 1
        return data.reshape((data.shape[0], c, w, h))

    train_full = MNIST(which='train')
    train_full.load()
    train_full.X = preprocess(train_full.X)

    train, valid = split(train_full, test_size=valid_ratio) # 10000 examples in validation set

    if training_subset is not None:
        nb = int(training_subset * len(train.X))
        print('training on a subset of training data of size : {}'.format(nb))
        train.X = train.X[0:nb]
        train.y = train.y[0:nb]

    if valid_subset is not None:
        nb = int(valid_subset * len(valid.X))
        print('validating on a subset of validation data of size : {}'.format(nb))
        valid.X = valid.X[0:nb]
        valid.y = valid.y[0:nb]

    test = MNIST(which='test')
    test.load()
    test.X = preprocess(test.X)

    if test_subset is not None:
        nb = int(test_subset * len(test.X))
        print('Testing on a subset of test data of size : {}'.format(nb))
        test.X = test.X[0:nb]
        test.y = test.y[0:nb]

    return train, valid, test
