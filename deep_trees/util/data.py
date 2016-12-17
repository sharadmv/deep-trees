import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def sample_minibatch(n, m):
    return np.random.choice(n, m, replace=False)

def load_mnist(subset_size, seed=0, one_hot=False):
    mnist = input_data.read_data_sets("/home/sharad/data/mnist/", one_hot=one_hot)
    X = mnist.train.images
    y = mnist.train.labels
    minibatch = sample_minibatch(X.shape[0], subset_size)
    X = X[minibatch]
    y = y[minibatch]
    return X, y
