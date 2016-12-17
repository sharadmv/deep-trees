import sys
import random
import numpy as np
from cpython cimport bool
from deepx import T

from .util import harmonic

cdef class DDT(object):

    cdef Node root
    cdef int root_index

    def __init__(self, Node root, int root_index = 0):
        self.root = root
        self.root_index = root_index

    cpdef public Node get_root(self):
        return self.root 

    def get_segments(self, node=None):
        if node is None:
            node = self.root

        for child in node.get_children():
            yield (node, child)
            for segment in self.get_segments(node=child):
                yield segment

    def get_states(self):
        return self.root.get_states()

    def induced_subtree(self, leaves):
        return DDT(self.root.induced_subtree(leaves, leave_node=True))

    def nodes(self, post=False, ignore_leaves=False, ignore_root=False):
        if ignore_root:
            return self.root.get_children()[0].nodes(post=post, ignore_leaves=ignore_leaves)
        return self.root.nodes(post=post, ignore_leaves=ignore_leaves)

    cpdef public int get_leaf_parent(self, int leaf_index):
        return self.root.get_leaf_parent(leaf_index)

    def select_node(self):
        chosen = None
        for i, node in enumerate(self.nodes(ignore_leaves=False, ignore_root=True)):
            if random.random() < 1.0 / (i + 1.0):
                chosen = node
        return chosen

    def detach_node(self):
        pass

    def pp(self):
        return self.root.pp(0)

    def __str__(self):
        return str(self.root)

    def __repr__(self):
        return "DDT(%s)" % repr(self.root)

    def copy(self):
        return DDT(self.root.copy(), root_index=self.root_index)

node_counter = 0

cdef class Node(object):

    cdef list children
    cdef int node_id
    cdef int D
    cdef Node parent
    cdef set leaf_cache
    cdef bool is_root
    cpdef state

    def __init__(self, list children, int D, int node_id = -1, bool is_root = False, state=None):
        global node_counter
        self.children = children
        self.D = D
        self.parent = None
        self.is_root = is_root
        self.leaf_cache = None
        if node_id is -1:
            self.node_id = node_counter
            node_counter += 1
        else:
            self.node_id = node_id
        if state is not None:
            self.state = state
        elif not self.is_leaf():
            if self.is_root:
                initial_value = np.zeros(D).astype(np.float32)
                initial_time = -np.inf
            else:
                initial_value = np.random.normal(size=D).astype(np.float32)
                initial_time = 0.0
            self.state = (
                T.variable(initial_time, trainable=not is_root, name="time-%u" % self.get_node_id()),
                T.variable(initial_value, trainable=not is_root, name="value-%u" % self.get_node_id()),
            )

    cpdef public Node copy(self):
        node = Node([c.copy() for c in self.children], self.D, node_id=self.node_id,
                    is_root=self.is_root,
                    state=self.state)
        for c in node.get_children():
            c.set_parent(node)
        return node

    def get_states(self, parent_time=0):
        time, value = self.state
        my_time = parent_time + (1 - parent_time) * T.sigmoid(time)
        my_value = value
        child_times, child_values = {}, {}
        for c in self.children:
            t, v = c.get_states(my_time)
            child_times.update(t)
            child_values.update(v)
        return (
            { self.get_node_id(): my_time, **child_times }, 
            { self.get_node_id(): my_value , **child_values }, 
        )

    cpdef public bool is_leaf(self):
        return False

    cpdef public bool is_node(self):
        return True

    cpdef public Node get_parent(self):
        return self.parent

    cpdef public void set_parent(self, node):
        self.parent = node

    cpdef public list get_children(self):
        return self.children

    cpdef public list get_node_stats(self):
        my_count = len(self.leaves())
        children_counts = [len(c.leaves()) for c in self.children]
        log_fac = np.sum([np.sum(np.log(range(1, c))) for  c in children_counts]) - np.sum(np.log(range(1, my_count + 1)))
        return [my_count, log_fac]

    cpdef public float log_fac(self):
        my_count = len(self.leaves())
        children_counts = [len(c.leaves()) for c in self.children]
        log_fac = np.sum([np.sum(np.log(range(1, c))) for  c in children_counts]) - np.sum(np.log(range(1, my_count + 1)))
        return log_fac

    cpdef public int get_leaf_parent(self, int leaf_index):
        leaves = self.leaves()
        for child in self.get_children():
            if leaf_index in child.leaves():
                if child.is_leaf():
                    return self.get_node_id()
                return child.get_leaf_parent(leaf_index)

    cpdef public void clear_leaf_cache(self):
        if self.parent is not None:
            self.parent.clear_leaf_cache()
        self.leaf_cache = None

    cpdef public set get_leaf_cache(self):
        return self.leaf_cache

    cpdef public int get_node_id(self):
        return self.node_id

    cpdef public float get_harmonic(self):
        return harmonic(len(self.leaves()) - 1)

    def nodes(self, post=False, ignore_leaves=False):
        if not post:
            yield self
        for child in self.children:
            for node in child.nodes(post=post, ignore_leaves=ignore_leaves):
                yield node
        if post:
            yield self

    def leaves(self):
        if self.leaf_cache is None:
            self.leaf_cache = set()
            for child in self.children:
                self.leaf_cache |= child.leaves()
        return self.leaf_cache

    def induced_subtree(self, leaves, leave_node=False):
        induced_children = []
        for child in self.children:
            child = child.induced_subtree(leaves)
            if child.leaves().intersection(leaves):
                induced_children.append(child)
        if len(induced_children) == 1 and not leave_node:
            return induced_children[0]
        return Node(induced_children, self.D, node_id=self.node_id, is_root=self.is_root, state=self.state)

    def pp(self, depth):
        pp = ""
        pp += "\t" * depth
        pp += "Node<%u>([\n" % self.node_id
        delimiter = ""
        for child in self.children:
            pp += delimiter
            pp += child.pp(depth + 1)
            delimiter = ",\n"
        pp += "\n"
        pp += "\t" * depth
        pp += "])"
        return pp


    def __str__(self):
        return "(%s)" % (" ".join(str(c) for c in self.children))

    def __repr__(self):
        return "Node(%u)" % (self.node_id)

cdef class Leaf(Node):

    cdef int index

    def __init__(self, int index):
        super(Leaf, self).__init__([], -1, node_id=-2)
        self.index = index

    cpdef public Node copy(self):
        return Leaf(self.index)

    cpdef public bool is_leaf(self):
        return True

    cpdef public bool is_node(self):
        return False

    cpdef public int get_index(self):
        return self.index

    cpdef public list get_node_stats(self):
        return [1, 0]

    def nodes(self, post=False, ignore_leaves=False):
        if not ignore_leaves:
            yield self

    def get_states(self, parent_time=0):
        return {}, {}

    def leaves(self):
        return {self.index}

    def induced_subtree(self, leaves):
        return Leaf(self.index)

    def pp(self, depth):
        return "\t" * depth + "Leaf(%u)" % (self.index)

    def __str__(self):
        return "(%u)" % self.index

    def __repr__(self):
        return "Leaf[%u]" % self.index
