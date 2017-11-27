cdef class Distribution:
    cdef dict parameter_cache
    cpdef public object get_parameters(self, str parameter_type)
