import sys
import numpy as np
from cpython cimport bool

cdef class DDT(object):

    cdef Node root

    def __init__(self, Node root):
        self.root = root

    cpdef public Node get_root(self):
        return self.root

    def get_segments(self, node=None):
        if node is None:
            node = self.root

        for child in node.get_children():
            yield (node, child)
            for segment in self.get_segments(node=child):
                yield segment

    def induced_subtree(self, leaves):
        return DDT(self.root.induced_subtree(leaves, leave_node=True))

    def nodes(self, post=False, ignore_leaves=False):
        return self.root.nodes(post=post, ignore_leaves=ignore_leaves)

    def pp(self):
        return self.root.pp(0)

    def __str__(self):
        return str(self.root)

    def __repr__(self):
        return "DDT(%s)" % repr(self.root)

node_counter = 0

cdef class Node(object):

    cdef list children
    cdef int node_id
    cdef set leaf_cache

    def __init__(self, list children, int node_id = -1):
        global node_counter
        self.children = children
        self.leaf_cache = None
        if node_id is -1:
            self.node_id = node_counter
            node_counter += 1
        else:
            self.node_id = node_id

    cpdef public bool is_leaf(self):
        return False

    cpdef public bool is_node(self):
        return True

    cpdef public list get_children(self):
        return self.children

    cpdef public list get_node_stats(self):
        my_count = len(self.leaves())
        children_counts = [len(c.leaves()) for c in self.children]
        log_fac = np.sum([np.sum(np.log(range(1, c))) for  c in children_counts]) - np.sum(np.log(range(1, my_count + 1)))
        return [my_count, log_fac]

    cpdef public set get_leaf_cache(self):
        return self.leaf_cache

    cpdef public int get_node_id(self):
        return self.node_id

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
        return Node(induced_children, node_id=self.node_id)

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
        super(Leaf, self).__init__([], node_id=-2)
        self.index = index

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
