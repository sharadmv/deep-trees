
cdef class Distribution(object):

    def __init__(self, parameters, str parameter_type = 'regular'):
        self.parameter_cache = {}
        self.parameter_cache[parameter_type] = parameters

    cpdef public get_parameters(self, str parameter_type):
        if parameter_type not in self.parameter_cache:
            if parameter_type == 'regular':
                self.parameter_cache['regular'] = self.natural_to_regular(self.get_parameters('natural'))
            elif parameter_type == 'natural':
                self.parameter_cache['natural'] = self.regular_to_natural(self.get_parameters('regular'))
        return self.parameter_cache[parameter_type]
