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

    # def __cinit__(self):
    def __cinit__(self, np.ndarray[DTYPE_t, ndim=2] data, double var=1.0):
        self.mem = None
        self.root = NULL
        self.data = None

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

    def __str__(self):
        return print_node(self.root)
