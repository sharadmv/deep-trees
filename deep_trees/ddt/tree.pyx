import numpy as np
from cpython cimport bool

cdef class DDT(object):

    cdef Node root

    def __init__(self, Node root):
        self.root = root

    cpdef public Node get_root(self):
        return self.root

cdef class Node(object):

    cdef list children

    def __init__(self, list children):
        self.children = children

    cpdef public bool is_leaf(self):
        return False

    cpdef public bool is_node(self):
        return True

    cpdef public list get_children(self):
        return self.children

    def __repr__(self):
        return "Node(%s)" % (self.children)

cdef class Leaf(Node):

    cdef int index 

    def __init__(self, int index):
        super(Leaf, self).__init__([])
        self.index = index

    cpdef public bool is_leaf(self):
        return True

    cpdef public bool is_node(self):
        return False

    cpdef public int get_index(self):
        return self.index

    def __repr__(self):
        return "Leaf(%u)" % self.index
