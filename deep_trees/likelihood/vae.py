from deepx import T
from deepx.nn import Vector, Linear, Sigmoid

from ..util import log_normal

class VAE(object):

    def __init__(self, input_size, embedding_size, q_network, likelihood_model):
        self.input_size = input_size
        self.embedding_size = embedding_size

        self.q_network = q_network >> (Linear(embedding_size), Linear(embedding_size))
        self.likelihood_model = likelihood_model

    def encode(self, batch):
        x = Vector(self.input_size, placeholder=batch, is_input=False)
        mu, sigma = (x >> self.q_network).get_graph_outputs()
        return mu

    def sample_z(self, batch, batch_noise, feed_dict={}):
        x = Vector(self.input_size, placeholder=batch, is_input=False)
        mu, sigma = (x >> self.q_network).get_graph_outputs()
        sigma = T.sqrt(T.exp(sigma))
        return mu + sigma * batch_noise

    def log_likelihood(self, batch_z, batch):
        x = Vector(self.input_size, placeholder=batch, is_input=False)
        mu, sigma = (x >> self.q_network).get_graph_outputs()
        sigma = T.sqrt(T.exp(sigma))
        return T.mean(log_normal(batch_z, mu, sigma, self.embedding_size, dim=2))

class LikelihoodModel(object):
    pass

class GaussianLikelihoodModel(object):
    pass

class BernoulliLikelihoodModel(object):

    def __init__(self, input_size, output_size, p_network):
        self.input_size = input_size
        self.output_size = output_size
        self.p_network = p_network >> Sigmoid(output_size)

    def log_likelihood(self, batch, batch_z):
        z = Vector(self.input_size, placeholder=batch_z, is_input=False)
        p = (z >> self.p_network).get_graph_outputs()[0]
        return T.mean(batch * p + (1 - batch) * T.log(1 - p + 1e-10))
