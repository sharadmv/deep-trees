
cdef extern from "math.h":
    cpdef double log(double x)

cdef dict fac_cache = {}

cpdef public log_factorial(float n):
    cdef float fac = 0
    if n in fac_cache:
        return fac_cache[n]
    for i in range(1, int(n) + 1):
        fac += log(i)
    fac_cache[n] = fac
    return fac

cdef public log_n_choose_k(n, k):
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)

cdef dict tree_cache = {}
cpdef public tree_factor(int n):
    if n in tree_cache:
        return tree_cache[n]
    fac = 0
    for i in range(2, n + 1):
        fac -= log_n_choose_k(i, 2)
    tree_cache[n] = fac
    return fac
