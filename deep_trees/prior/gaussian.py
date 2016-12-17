import numpy as np
from deepx import T

from ..util import log_normal

class GaussianPrior(object):

    def __init__(self, num_data, embedding_size, c=2, sigma0=1.0):
        self.embedding_size = embedding_size

    def get_info(self, indices):
        return {}

    def log_prior(self, leaf_values):
        return T.mean(log_normal(leaf_values,
                          T.zeros_like(leaf_values, dtype='float32'),
                          T.ones_like(leaf_values, dtype='float32'),
                          self.embedding_size, dim=2))
