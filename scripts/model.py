import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from deep_trees.ddt import Node, Leaf, DDT

from deepx import T
from deepx.nn import Vector, Relu, Repeat, Tanh, Linear

def log_normal(x, mu, sigma):
    pre = -(D / 2.0 * np.log(2 * np.pi) + 0.5 * T.log(sigma))
    return pre #- 0.5 * sigma * T.dot(x - mu, x - mu)

if __name__ == "__main__":

    X, tree = load_data(100)
    D = 10
    M = 10
    values, times = initialize_tree(tree, D)

    def a(t):
        return c / (1 - t)

    def log_a(t):
        return T.log(c / (1 - t))

    def A(t):
        return -c * T.log(1 - t)

    def create_harmonic(M):
        return np.cumsum(1.0 / np.arange(1, M + 1)).astype(np.float32)

    T.set_default_device('/cpu:0')

    c = T.scalar(name='c')
    segments = T.matrix(dtype='int32', name='segments')

    a_idx = segments[:, 0]
    b_idx = segments[:, 1]
    leaf_segment = segments[:, 2]
    m = segments[:, 3]
    log_fac = segments[:, 4]

    x = T.matrix(name='x')
    e = T.matrix(name='e')
    q_network = Vector(X.shape[1], placeholder=x, is_input=False) >> Repeat(Tanh(200), 2)
    q_mu_network = q_network >> Linear(D)
    q_mu = q_mu_network.get_outputs()[0].get_placeholder()
    q_sigma_network = q_network >> Linear(D)
    q_sigma = tf.sqrt(tf.exp(q_sigma_network.get_outputs()[0].get_placeholder()))
    z = q_mu + e * q_sigma

    values, times = T.variable(values), T.variable(times)
    values = tf.concat(0, [z, values])
    harmonic = T.variable(create_harmonic(M))

    a_batch_values = T.gather(values, a_idx)
    a_batch_times = T.gather(times, a_idx)
    b_batch_values = T.gather(values, b_idx)
    b_batch_times = T.gather(times, b_idx)
    harmonic_m = T.gather(harmonic, m - 1)

    time_delta = b_batch_times - a_batch_times

    normal_log_prob = log_normal(b_batch_times, a_batch_times, time_delta * 1.0 / 1)

    log_pt = (log_a(a_batch_times) + (A(a_batch_times) - A(b_batch_times)) * harmonic_m)
    tree_log_prob = tf.select(tf.cast(leaf_segment, 'bool'), T.zeros_like(a_batch_times), log_pt + tf.to_float(log_fac))

    log_q_zx =  log_normal(z, q_mu, q_sigma)

    p_network = Vector(D, placeholder=z, is_input=False) >> Repeat(Tanh(200), 2)
    p_mu_network = p_network >> Linear(X.shape[1])
    p_mu = p_mu_network.get_outputs()[0].get_placeholder()
    p_sigma_network = p_network >> Linear(X.shape[1])
    p_sigma = tf.sqrt(tf.exp(p_sigma_network.get_outputs()[0].get_placeholder()))

    log_p_xz = log_normal(x, p_mu, p_sigma)# + tf.reduce_sum(tree_log_prob)

    with T.session() as sess:
        batch = set(sample_minibatch(N, M))
        subtree = tree.induced_subtree(batch)
        s = np.array([[a.get_node_id(), b.get_node_id(), int(b.is_leaf())] + b.get_node_stats() for a, b in subtree.get_segments()])
        minibatch = X[list(batch)]
        result = sess.run([values, log_p_xz, tree_log_prob, normal_log_prob], feed_dict={segments : s, c: 1.0, x: minibatch, e: np.random.normal(size=(M, D))})
