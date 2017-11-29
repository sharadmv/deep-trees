from cymem.cymem cimport Pool
cimport numpy as np
from .tree cimport add_child, make_node, make_leaf, Node

cdef Node* init_tree(Pool mem, int num_leaves):
    return tree_helper(mem, list(range(num_leaves)))

cdef Node* tree_helper(Pool mem, list leaves):
    if len(leaves) == 1:
        return make_leaf(mem, leaves[0])
    node = make_node(mem, 0.5)
    add_child(node, tree_helper(mem, leaves[:len(leaves)//2]))
    add_child(node, tree_helper(mem, leaves[len(leaves)//2:]))
    return node
