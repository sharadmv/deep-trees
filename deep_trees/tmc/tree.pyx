import cython
from scipy.linalg cimport cython_blas as blas
from libc.math cimport log
import numpy as np
cimport numpy as np
from cymem.cymem cimport Pool

from .util cimport init_tree
from .blas cimport axpy, dot

ctypedef struct Node:
    Node* children[2]
    int num_children
    Node* parent
    int is_leaf
    int index
    double time_weight
    double z

ctypedef struct Message:
    double eta1
    double[::1] eta2

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
    if node == NULL:
        return "()"
    if node.is_leaf:
        return "Leaf[%u]" % node.index
    return "(%.3f (%s))" % (node_time(node), " ".join([print_node(node.children[i]) for
                                                       i in range(node.num_children)]))

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class TMC:

    def __cinit__(self, double[:, ::1] data, double var=1.0, double prior=1.0):
        self.mem = Pool()
        self.data = data
        self.num_leaves = data.shape[0]
        self.dim = data.shape[1]
        self.root = init_tree(self.mem, self.num_leaves)
        self.var = var
        self.prior = prior

    cdef Node* index(self, list idx):
        if idx == None:
            return NULL
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

    cpdef collect_messages(self, list idx = []):
        node = self.index(idx)
        return node_time(node)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef Message* get_message(self, Node* node, Node* source) except NULL:
        cdef:
            Message* message
            Node* neighbors[3]
            double v_, factor, factor_inv, inner_sigma, inner_sigma_inv
            double eta1 = 0
            double[::1] eta2

        message = <Message*>self.mem.alloc(1, sizeof(Message))
        if node == NULL:
            v_ = 1 / self.prior
            message.eta1 = -0.5 * v_
            message.eta2 = <double[:self.dim:1]>(<double*>self.mem.alloc(self.dim, sizeof(double)))
        elif node.is_leaf:
            v_ = 1 / branch_length(node) / self.var
            message.eta1 = -0.5 * v_
            message.eta2 = <double[:self.dim:1]>(<double*>self.mem.alloc(self.dim, sizeof(double)))
            message.eta2 = axpy(v_, self.data[node.index], message.eta2, self.dim)
        else:
            neighbors[0] = node.parent
            neighbors[1] = node.children[0]
            if node.num_children == 2:
                neighbors[2] = node.children[1]

            eta2 = <double[:self.dim:1]>self.mem.alloc(self.dim, sizeof(double))
            for i in range(node.num_children + 1):
                neighbor = neighbors[i]
                if neighbor == source:
                    continue
                neighbor_message = self.get_message(neighbor, node)
                eta1 += neighbor_message.eta1
                eta2 = axpy(1, neighbor_message.eta2, eta2, self.dim)

            if source == node.parent:
                factor = branch_length(node)
            else:
                factor = branch_length(source)
            factor_inv = 1. / factor
            inner_sigma = -0.5 * factor_inv + eta1
            inner_sigma_inv = 1. / inner_sigma
            message.eta1 = -0.25 * (factor_inv * inner_sigma_inv * factor_inv) - 0.5 * factor_inv
            message.eta2 = <double[:self.dim:1]>(<double*>self.mem.alloc(self.dim, sizeof(double)))
            v_ = -0.5 * factor_inv * inner_sigma_inv
            message.eta2 = axpy(v_, eta2, message.eta2, self.dim)
        print("Getting message from %s -> %s" % (print_node(node), print_node(source)))
        return message

    cdef double get_z(self, Node* node, Node* source):
        cdef:
            Node* neighbors[3]
            Node* neighbor
            double eta1 = 0
            double[::1] eta2 = <double[:self.dim:1]>(<double*>self.mem.alloc(self.dim, sizeof(double)))
            Message* message
            double z

        if node == NULL or node.is_leaf:
            return 0.0

        neighbors[0] = node.parent
        neighbors[1] = node.children[0]
        if node.num_children == 2:
            neighbors[2] = node.children[1]

        for i in range(node.num_children + 1):
            neighbor = neighbors[i]
            if neighbor == source:
                continue
            message = self.get_message(neighbor, node)
            eta1 += message.eta1
            eta2 = axpy(1, message.eta2, eta2, self.dim)

        inner_sigma_inv = 1. / eta1
        z = - 0.25 * inner_sigma_inv * dot(eta2, eta2, self.dim)
        return z

    cdef double likelihood(self, Node* node):
        cdef:
            Node* neighbors[3]
            Node* neighbor
            double likelihood

        if node.is_leaf:
            raise Exception()

        neighbors[0] = node.parent
        neighbors[1] = node.children[0]
        if node.num_children == 2:
            neighbors[2] = node.children[1]

        for i in range(node.num_children + 1):
            neighbor = neighbors[i]
            likelihood += self.get_z(neighbor)
        return likelihood

    def ll(self):
        return self.likelihood(self.index([]))

    cpdef message(self, node, source):
        cdef Message* message
        message = self.get_message(self.index(node), self.index(source))
        return (message.eta1, np.asarray(message.eta2))

    # cpdef get_message(self, list idx):
        # cdef:
            # double[::1] data
            # double param0
            # double* param1_
            # int incx = 1
            # double vs = 1
            # double[::1] param1
            # double[::1] message2
            # Node* neighbor
            # Node* neighbors[3]
        # node = self.index(idx)
        # neighbors[0] = node.parent
        # neighbors[1] = node.children[0]
        # if node.num_children == 2:
            # neighbors[2] = node.children[1]
        # param1_ = <double*>self.mem.alloc(self.dim, sizeof(double))
        # param0, param1 = 0, <double[:self.dim]>param1_
        # for i in range(node.num_children + 1):
            # neighbor = neighbors[i]
            # neighbor_message = self.compute_message(neighbor, node)
            # param0 += neighbor_message['message'][0]
            # message2 = neighbor_message['message'][1]
            # blas.daxpy(&self.dim, &vs, &param1[0], &incx, &message2[0], &incx)
            # print(np.asarray(param1), np.asarray(message2))
        # return (param0, np.asarray(param1))

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.cdivision(True)
    # cdef dict compute_message(self, Node* node, Node* source):
        # cdef:
            # double[::1] data
            # int incx = 1
            # double v, s, vs, z, param0
            # double[::1] y, param1
            # double y_y, factor
            # double* y_
            # double message1 
            # double[::1] message2
            # Node* neighbor
            # Node* neighbors[3]
        # if node == NULL:
            # v = self.var
            # y_ = <double*>self.mem.alloc(self.dim, sizeof(double))
            # y = <double[:self.dim]>y_
            # return {
                # 'message' : (-0.5 / v, np.zeros(self.dim)),
                # 'z': -0.5 / v - 0.5 * self.dim * log(v)
            # }
        # elif node.is_leaf:
            # data = self.data[node.index]
            # y_y = blas.ddot(&self.dim, &data[0], &incx, &data[0], &incx)
            # v, s = self.var, branch_length(node)
            # vs = 1 / (v * s)
            # y_ = <double*>self.mem.alloc(self.dim, sizeof(double))
            # blas.daxpy(&self.dim, &vs, &data[0], &incx, y_, &incx)
            # y = <double[:self.dim]>y_
            # return {
                # 'message' : (-0.5 * vs, y),
                # 'z': -0.5 * vs * y_y - 0.5 * self.dim * (log(v) + log(s))
            # }
        # else:
            # neighbors[0] = node.parent
            # neighbors[1] = node.children[0]
            # if node.num_children == 2:
                # neighbors[2] = node.children[1]
            # y_ = <double*>self.mem.alloc(self.dim, sizeof(double))
            # param0, param1 = 0, <double[:self.dim]>y_
            # z = 0
            # for i in range(node.num_children + 1):
                # neighbor = neighbors[i]
                # if neighbor == source:
                    # continue
                # neighbor_message = self.compute_message(neighbor, node)
                # param0 += neighbor_message['message'][0]
                # message2 = neighbor_message['message'][1]
                # blas.daxpy(&self.dim, &vs, &param1[0], &incx, &message2[0], &incx)
                # z += neighbor_message['z']
            # factor = self.get_factor(node, source)
            # factor_inv = 1 / factor
            # inner_sigma = -0.5 * factor_inv + param0
            # inner_sigma_inv = 1 / inner_sigma
            # y_ = <double*>self.mem.alloc(self.dim, sizeof(double))
            # vs = -0.5 * factor_inv * inner_sigma_inv
            # blas.daxpy(&self.dim, &vs, &param1[0], &incx, y_, &incx)
            # y = <double[:self.dim]>y_
            # marginal = (
                # -0.25 * (factor_inv * inner_sigma_inv * factor_inv) - 0.5 * factor_inv,
                # y
            # )
                # # - 0.25 * np.einsum('a,ab,b->', joint[1], inner_sigma_inv, joint[1])
                # # - 0.5 * np.linalg.slogdet(-2 * inner_sigma)[1]
                # # - 0.5 * np.linalg.slogdet(factor)[1]),
            # return {
                # 'message': marginal,
                # 'z': z
            # }

    def __str__(self):
        return print_node(self.root)
