import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def sample_minibatch(n, m):
    return np.random.choice(n, m, replace=False)

def load_mnist(subset_size, seed=0):
    mnist = input_data.read_data_sets("/home/sharad/data/mnist/", one_hot=True)
    X = mnist.train.images
    X = X[sample_minibatch(X.shape[0], subset_size)]
    return X
