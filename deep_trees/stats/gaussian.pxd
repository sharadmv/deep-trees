from .common cimport Distribution

cdef class Gaussian(Distribution):
    cpdef public log_likelihood(self, x)

    cpdef public log_h(self, x)
    cpdef public log_z(self)
