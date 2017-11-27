import numpy as np
from deepx import T

def log_normal(x, mu, sigma, D, dim=1):
    if dim == 1:
        pre_term = -(D * 0.5 * np.log(2 * np.pi) + 0.5 * D * T.log(sigma))
        delta = T.sum((x - mu) ** 2, axis=1) * 1.0 / sigma
        return pre_term + -0.5 * delta
    elif dim == 2:
        pre_term = -(D * 0.5 * np.log(2 * np.pi) + 0.5 * T.sum(T.log(sigma), axis=1))
        delta = T.sum((x - mu) * 1.0 / sigma * (x - mu), axis=1)
    return pre_term + -0.5 * delta
