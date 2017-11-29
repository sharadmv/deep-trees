from cymem.cymem cimport Pool
from .tree cimport Node

cdef Node* init_tree(Pool mem, int num_leaves)
