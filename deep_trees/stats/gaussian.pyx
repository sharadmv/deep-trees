import numpy as np
cimport numpy as np

from .common cimport Distribution
cdef class Gaussian(Distribution):

    cpdef log_likelihood(self, x):
        sigma, mu = self.get_parameters('regular')
        d = mu.shape[-1]
        delta = x - mu
        term = np.linalg.solve(sigma, delta[..., None])[..., 0]
        exp_term = np.matmul(delta[..., None, :], term[..., None])[..., 0, 0]
        return -0.5 * (d * np.log(2 * np.pi) + np.linalg.slogdet(sigma)[1]) - 0.5 * exp_term

    cpdef log_h(self, x):
        d = x.shape[-1]
        return 0.5 * d * np.log(2 * np.pi)

    @classmethod
    def regular_to_natural(cls, regular_parameters):
        sigma, mu = regular_parameters
        sigma_inv = np.linalg.inv(sigma)
        eta1 = -0.5 * sigma_inv                              # -\frac{1}{2} \Sigma^{-1}
        eta2 = np.linalg.solve(sigma, mu[..., None])[..., 0] # \Sigma^{-1}\mu
        return Gaussian.pack([eta1, eta2])

    @classmethod
    def natural_to_regular(cls, natural_parameters):
        J, h = Gaussian.unpack(natural_parameters)
        sigma_inv = -2 * J
        mu = np.linalg.solve(sigma_inv, h[..., None])[..., 0]
        return np.linalg.inv(sigma_inv), mu

    @staticmethod
    def pack(natural_parameters):
        eta1, eta2 = natural_parameters
        return pack_dense(eta1, eta2)

    @staticmethod
    def unpack(packed_parameters):
        J, h, _, _ = unpack_dense(packed_parameters)
        return J, h

    cpdef log_z(self):
        sigma, mu = self.get_parameters('regular')
        sigma_inv = np.linalg.inv(sigma)
        mu_shape = mu.shape
        return np.einsum('...a,...ab,...b->...', mu, sigma_inv, mu) + 0.5 * np.linalg.slogdet(sigma)[1]

vs, hs = lambda x: np.concatenate(x, axis=-2), lambda x: np.concatenate(x, axis=-1)
T = lambda X: np.swapaxes(X, axis1=-1, axis2=-2)

def pack_dense(A, b, *args):
    leading_dim, N = b.shape[:-1], b.shape[-1]
    z1, z2 = np.zeros(leading_dim + (N, 1)), np.zeros(leading_dim + (1, 1))
    c, d = args if args else (z2, z2)

    A = A[...,None] * np.eye(N)[None,...] if A.ndim == b.ndim else A
    b = b[...,None]
    c, d = np.reshape(c, leading_dim + (1, 1)), np.reshape(d, leading_dim + (1, 1))

    return vs(( hs(( A,     b,  z1 )),
                hs(( T(z1), c,  z2 )),
                hs(( T(z1), z2, d  ))))

def unpack_dense(arr):
    N = arr.shape[-1] - 2
    return arr[...,:N, :N], arr[...,:N,N], arr[...,N,N], arr[...,N+1,N+1]
