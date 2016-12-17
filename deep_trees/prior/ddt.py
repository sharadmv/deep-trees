import numpy as np
from deepx import T

from ..ddt import make_divergence, create_tree
from ..util import log_normal

class DDTPrior(object):

    def __init__(self, num_data, embedding_size, c=2, sigma0=1.0, tree=None):
        self.embedding_size = embedding_size
        if tree is None:
            self.tree = create_tree(num_data, embedding_size)
        else:
            self.tree = tree
        self.c = c
        self.a, self.A = make_divergence(self.c)
        self.sigma0 = sigma0
        self._initialize()

    def _initialize(self):
        self.calculate_state()
        self.left_segment = T.vector(dtype='int32')
        self.right_segment = T.vector(dtype='int32')
        self.log_fac = T.vector(dtype='float32')
        self.harmonic = T.vector(dtype='float32')
        self.parents = T.vector(dtype='int32')

    def calculate_state(self):
        nodes = list(self.tree.nodes(post=True, ignore_leaves=True))
        values = np.random.normal(size=(len(nodes), self.embedding_size))
        times, values = [], []
        times_dict, values_dict  = self.tree.get_states()
        for i in range(len(nodes)):
            times.append(times_dict[i])
            values.append(values_dict[i])
        self.values = T.concatenate(values, concat=True)
        self.times = T.concatenate(times, concat=True)

    def get_info(self, indices):
        subtree = self.tree.induced_subtree(indices)
        left, right, log_fac, harmonic = [], [], [], []
        for a, b in subtree.get_segments():
            if b.is_leaf():
                continue
            left.append(a.get_node_id())
            right.append(b.get_node_id())
            log_fac.append(b.log_fac())
            harmonic.append(b.get_harmonic())

        parents = []
        for i in indices:
            parents.append(subtree.get_leaf_parent(i))
        return {
            self.left_segment: left,
            self.right_segment: right,
            self.log_fac: log_fac,
            self.harmonic: harmonic,
            self.parents: parents
        }

    def hill_climb(self):
        tree = self.tree.copy()
        node = tree.select_node()
        while node.get_parent() == tree.get_root():
            node = tree.select_node()
        parent = node.get_parent()
        parent.clear_leaf_cache()
        node_index = parent.get_children().index(node)
        sibling = parent.get_children()[1 - node_index]
        parent.get_children().remove(sibling)
        grandparent = parent.get_parent()
        grandparent.get_children().remove(parent)
        grandparent.get_children().append(sibling)
        sibling.set_parent(grandparent)
        parent.set_parent(None)

        attach_node = tree.select_node()
        attach_parent = attach_node.get_parent()
        attach_parent.get_children().remove(attach_node)
        attach_parent.get_children().append(parent)
        parent.set_parent(attach_parent)
        parent.get_children().append(attach_node)
        attach_node.set_parent(parent)
        return tree

    def log_prior(self, leaf_values):
        left_times, right_times = T.gather(self.times, self.left_segment), T.gather(self.times, self.right_segment)
        left_values, right_values = T.gather(self.values, self.left_segment), T.gather(self.values, self.right_segment)
        term1 = T.log(self.a(right_times)) + T.exp((self.A(left_times) - self.A(right_times)) * self.harmonic)
        term2 = self.log_fac
        term3 = log_normal(right_values, left_values, (right_times - left_times) * self.sigma0, self.embedding_size, dim=1)

        parent_values = T.gather(self.values, self.parents)
        parent_times = T.gather(self.times, self.parents)
        term4 = log_normal(leaf_values, parent_values, self.sigma0 * (1 - parent_times), self.embedding_size, dim=1)
        return T.sum(term1 + term2 + term3) + T.sum(term4)

class SubtreeInfo(object):

    def __init__(self, left, right, log_fac, parents):
        pass

    def feed_dict(self):
        pass
