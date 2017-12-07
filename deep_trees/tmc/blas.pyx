from scipy.linalg cimport cython_blas as blas

cdef int incx = 1

cdef double[::1] axpy(double a, double[::1] x, double[::1] y, int dim):
    blas.daxpy(&dim, &a, &x[0], &incx, &y[0], &incx)
    return y

cdef double dot(double[::1] x, double[::1] y, int dim):
    return blas.ddot(&dim, &x[0], &incx, &y[0], &incx)
