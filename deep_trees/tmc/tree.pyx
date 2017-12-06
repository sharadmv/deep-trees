import cython
from scipy.linalg cimport cython_blas as blas
from libc.math cimport log
import numpy as np
cimport numpy as np
from cymem.cymem cimport Pool
from .util cimport init_tree

ctypedef struct Node:
    Node* children[2]
    int num_children
    Node* parent
    int is_leaf
    int index
    double time_weight

cdef Node* make_node(Pool mem, double time_weight) except NULL:
    node = <Node*>mem.alloc(1, sizeof(Node))
    node.is_leaf = False
    node.index = -1
    node.time_weight = time_weight
    return node

cdef Node* make_leaf(Pool mem, int index) except NULL:
    node = <Node*>mem.alloc(1, sizeof(Node))
    node.is_leaf = True
    node.index = index
    node.time_weight = 1.0
    return node

cdef void add_child(Node* parent, Node* child):
    if parent.is_leaf:
        raise Exception("Leaf cannot have children")
    if parent.num_children > 1:
        raise Exception("Node already has 2 children")
    parent.children[parent.num_children] = child
    parent.num_children += 1
    child.parent = parent

cdef void remove_child(Node* parent, Node* child):
    if parent.is_leaf:
        raise Exception("Leaf cannot have children")
    if parent.num_children == 0:
        raise Exception("No children for removal")
    elif parent.num_children == 1:
        parent.children[0] = NULL
        parent.num_children -= 1
        child.parent = NULL
        return
    elif parent.num_children == 2:
        if parent.children[0] == child:
            parent.children[0] = parent.children[1]
            parent.children[1] = NULL
            parent.num_children -= 1
            child.parent = NULL
            return
        elif parent.children[1] == child:
            parent.children[1] = NULL
            parent.num_children -= 1
            child.parent = NULL
            return
    raise Exception("Couldn't find child")

cdef double branch_length(Node* node) except -1:
    if node.parent == NULL:
        raise Exception("Root has no branch length")
    return node_time(node) - node_time(node.parent)

cdef double node_time(Node* node) except -1:
    if node.parent == NULL:
        return 0.0
    parent_time = node_time(node.parent)
    return parent_time + (1 - parent_time) * node.time_weight

cdef str print_node(Node* node):
    if node.is_leaf:
        return "Leaf[%u]" % node.index
    return "(%.3f (%s))" % (node_time(node), " ".join([print_node(node.children[i]) for
                                                       i in range(node.num_children)]))

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class TMC:

    def __cinit__(self, np.ndarray[DTYPE_t, ndim=2] data, double var=1.0):
        self.mem = Pool()
        self.data = data
        self.num_leaves = data.shape[0]
        self.dim = data.shape[1]
        self.root = init_tree(self.mem, self.num_leaves)
        self.var = var

    cdef Node* index(self, list idx):
        cdef Node* node = self.root
        cdef int i
        while len(idx) > 0:
            i = idx.pop(0)
            node = node.children[i]
        return node

    cpdef time(self, list idx):
        return node_time(self.index(idx))

    cpdef branch_length(self, list idx):
        return branch_length(self.index(idx))

    cpdef get_message(self, list idx):
        cdef:
            double[::1] data
            double param0
            double[::1] param1
            Node* neighbor
            Node* neighbors[3]
        node = self.index(idx)
        neighbors[0] = node.parent
        neighbors[1] = node.children[0]
        if node.num_children == 2:
            neighbors[2] = node.children[1]
        param0, param1 = 0, np.zeros(self.dim)
        for i in range(node.num_children + 1):
            neighbor = neighbors[i]
            neighbor_message = self.compute_message(neighbor, node)
            param0 += neighbor_message['message'][0]
            for d in range(self.dim):
                param1[d] += neighbor_message['message'][1][d]
        return (param0, np.asarray(param1))

    cdef double get_factor(self, Node* node, Node* source):
        if source == node.parent:
            return branch_length(node)
        else:
            return branch_length(source)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef dict compute_message(self, Node* node, Node* source):
        cdef:
            double[::1] data
            int incx = 1
            double v, s, vs, z, param0
            double[::1] y, param1
            double y_y, factor
            Node* neighbor
            Node* neighbors[3]
        if node == NULL:
            v = self.var
            y = np.zeros(self.dim)
            return {
                'message' : (-0.5 / v, np.zeros(self.dim)),
                'z': -0.5 / v - 0.5 * self.dim * log(v)
            }
        elif node.is_leaf:
            data = self.data[node.index]
            y_y = blas.ddot(&self.dim, &data[0], &incx, &data[0], &incx)
            v, s = self.var, branch_length(node)
            vs = 1 / (v * s)
            y = np.zeros(self.dim)
            blas.daxpy(&self.dim, &vs, &data[0], &incx, &y[0], &incx)
            return {
                'message' : (-0.5 * vs, y),
                'z': -0.5 * vs * y_y - 0.5 * self.dim * (log(v) + log(s))
            }
        else:
            neighbors[0] = node.parent
            neighbors[1] = node.children[0]
            if node.num_children == 2:
                neighbors[2] = node.children[1]
            param0, param1 = 0, np.zeros(self.dim)
            z = 0
            for i in range(node.num_children + 1):
                neighbor = neighbors[i]
                if neighbor == source:
                    continue
                neighbor_message = self.compute_message(neighbor, node)
                param0 += neighbor_message['message'][0]
                for d in range(self.dim):
                    param1[d] += neighbor_message['message'][1][d]
                z += neighbor_message['z']
            factor = self.get_factor(node, source)
            factor_inv = 1 / factor
            inner_sigma = -0.5 * factor_inv + param0
            inner_sigma_inv = 1 / inner_sigma
            y = np.zeros(self.dim)
            vs = -0.5 * factor_inv * inner_sigma_inv
            blas.daxpy(&self.dim, &vs, &param1[0], &incx, &y[0], &incx)
            marginal = (
                -0.25 * (factor_inv * inner_sigma_inv * factor_inv) - 0.5 * factor_inv,
                y
            )
                # - 0.25 * np.einsum('a,ab,b->', joint[1], inner_sigma_inv, joint[1])
                # - 0.5 * np.linalg.slogdet(-2 * inner_sigma)[1]
                # - 0.5 * np.linalg.slogdet(factor)[1]),
            return {
                'message': marginal,
                'z': z
            }

    def __str__(self):
        return print_node(self.root)
