cimport numpy as np
from cymem.cymem cimport Pool

ctypedef struct Node:
    Node* children[2]
    int num_children
    Node* parent
    int is_leaf
    int index
    double time_weight

cdef Node* make_node(Pool mem, double time_weight) except NULL
cdef Node* make_leaf(Pool mem, int index) except NULL
cdef void add_child(Node* parent, Node* child)
cdef void remove_child(Node* parent, Node* child)
cdef double branch_length(Node* node) except -1
cdef double node_time(Node* node) except -1
cdef str print_node(Node* node)

cdef class TMC:
    cdef Pool mem

    cdef Node* root
    cdef public np.ndarray data
    cdef public int num_leaves, dim
    cdef double var

    cdef Node* index(self, list idx)
    cdef dict compute_message(self, Node* node, Node* source)
    cdef double get_factor(self, Node* node, Node* source)

    cpdef time(self, list idx)
    cpdef branch_length(self, list idx)
    cpdef get_message(self, list idx)
