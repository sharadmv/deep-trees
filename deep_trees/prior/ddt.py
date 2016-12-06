import numpy as np
from deepx import T

from ..util import create_tree

class DDTPrior(object):

    def __init__(self, num_data, embedding_size, c=2):
        self.embedding_size = embedding_size
        self.tree = create_tree(num_data)
        self.c = c
        self._initialize()

    def _initialize(self):
        nodes = list(self.tree.nodes(post=True, ignore_leaves=True))
        values = np.zeros((len(nodes), self.embedding_size))
        times = np.zeros(len(nodes))
        for node in nodes:
            values[node.get_node_id()] = np.random.normal(size=self.embedding_size)
            times[node.get_node_id()] = min(
                1.0 if c.is_leaf() else times[c.get_node_id()]
                for c in node.get_children()
            ) / 2.0
        times[self.tree.get_root().get_node_id()] = 0
        self.values = T.shared(values)
        self.times = T.shared(times)

    def log_prior(self, z, indices):
        subtree = self.tree.induced_subtree(indices)
        leaf_map = { index : i for i, index in enumerate(indices) }
        mat = T.concat(0, z, self.values)
