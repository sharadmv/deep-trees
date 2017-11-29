from scipy.optimize import minimize_scalar
from scipy.stats import beta
import numpy as np

from deep_trees.stats import Gaussian

cimport numpy as np
from cpython cimport bool

from .util import log_factorial, tree_factor

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "math.h":
    cpdef double log(double x)

cdef class TMC(object):

    cdef Node root
    cdef public np.ndarray data
    cdef int dim
    cdef bool subtree
    cdef dict cache
    cdef float alpha, beta

    def __init__(self, Node root, np.ndarray[DTYPE_t, ndim=2] data, bool subtree = False,
                 float alpha = 2.0, float beta = 2.0):
        self.root = root
        self.data = data
        self.dim = data.shape[1]
        self.subtree = subtree
        self.alpha = alpha
        self.beta = beta
        self.clear_cache()

    cpdef public Node get_root(self):
        return self.root

    cpdef public Node clear_cache(self):
        self.cache = {
            'num_nodes': {},
            'messages': {}
        }

    cpdef marginals(self):
        leaves = self.get_leaf_marginals()
        marginals = [None] * len(leaves)
        for index, marginal in leaves:
            marginals[index] = marginal
        return marginals

    cpdef get_leaf_marginals(self, Node node = None):
        if node is None:
            node = self.root
        if node.is_leaf():
            return [(node.index, self.marginal(node))]
        else:
            marginals = []
            for child in node.children:
                marginals.extend(self.get_leaf_marginals(child))
            return marginals

    cpdef public get_message(self, Node node, Node source):
        if (node, source) not in self.cache['messages']:
            if node is None:
                p = Gaussian([np.eye(self.dim), np.zeros(self.dim)])
                message = (
                    p.get_parameters('natural'), 
                    -(p.log_z() + p.log_h(np.zeros(self.dim))),
                    0,
                )
            elif node.is_leaf():
                branch_length = node.branch_length()
                sigma = np.eye(self.dim) * branch_length
                value = self.data[node.index]
                p = Gaussian([sigma, self.data[node.index]], 'regular')
                message = (
                    p.get_parameters('natural'), 
                    -(p.log_z() + p.log_h(value)),
                    0,
                )
            else:
                neighbors = (set(node.children) | {node.parent}) - {source}
                log_likelihood, joint = 0, np.zeros([self.dim + 2, self.dim + 2])
                time_prior = 0
                for neighbor in neighbors:
                    message = self.get_message(neighbor, node)
                    joint += message[0]
                    log_likelihood += message[1]
                    time_prior += message[2]
                joint = Gaussian.unpack(joint)
                factor = self.get_factor(source, node)
                factor_inv = np.linalg.inv(factor)
                inner_sigma = -0.5 * factor_inv + joint[0]
                inner_sigma_inv = np.linalg.inv(inner_sigma)
                branch_length = node.branch_length()
                marginal = Gaussian(Gaussian.pack([
                    -0.25 * np.einsum('ab,bc,cd->ad', factor_inv, inner_sigma_inv, factor_inv) - 0.5 * factor_inv,
                    -0.5 * np.einsum('ab,bc,c->a', factor_inv, inner_sigma_inv, joint[1])
                ]), 'natural')
                time_factor = 0
                if not (node is self.root):
                    time_factor = time_prior + beta(self.alpha, self.beta).logpdf(branch_length / (1 - node.parent.time))
                message = (
                    marginal.get_parameters('natural'), 
                    (log_likelihood
                    - 0.25 * np.einsum('a,ab,b->', joint[1], inner_sigma_inv, joint[1])
                    - 0.5 * np.linalg.slogdet(-2 * inner_sigma)[1]
                    - 0.5 * np.linalg.slogdet(factor)[1]),
                    time_factor
                )
            # print("Computing message from %s -> %s" % (node, source))
            self.cache['messages'][node, source] = message
        message = self.cache['messages'][node, source]
        return message

    cpdef num_nodes(self, node):
        if node not in self.cache['num_nodes']:
            if node.is_leaf():
                return 0, 0.0
            num_nodes = 1
            prod_nodes = 0
            for child in node.children:
                message = self.num_nodes(child)
                num_nodes += message[0]
                prod_nodes += message[1]
            prod_nodes += log(num_nodes)
            self.cache['num_nodes'][node] = (num_nodes, prod_nodes)
        return self.cache['num_nodes'][node]

    cpdef get_factor(self, node, source):
        if source is node.parent:
            return np.eye(self.dim) * node.branch_length()
        else:
            return np.eye(self.dim) * source.branch_length()

    cpdef marginal(self, node):
        neighbors = (set(node.children) | {node.parent})
        param = np.zeros([self.dim + 2, self.dim + 2])
        for neighbor in neighbors:
            message = self.get_message(neighbor, node)
            param += message[0]
        return Gaussian(param, 'natural')

    cpdef public detach(self, node):
        assert not (node is self.root or node.parent is self.root)
        parent = node.parent
        grandparent = parent.parent
        sibling = list(set(parent.children) - {node})[0]
        assert parent in grandparent.children
        assert node in parent.children
        grandparent.remove_child(parent)
        parent.remove_child(sibling)
        grandparent.add_child(sibling)
        return parent

    cpdef public attach(self, subtree, node, time):
        parent = node.parent
        assert node.time >= time >= parent.time, (node.time, time, parent.time)
        subtree.child(0).fix_time(time)
        subtree.time = time
        parent.remove_child(node)
        parent.add_child(subtree)
        subtree.add_child(node)
        return self

    cpdef tree_prior(self):
        num_nodes, prod_nodes = self.num_nodes(self.root)
        return log_factorial(num_nodes) - prod_nodes + tree_factor(num_nodes + 1)

    def find_interval(self, subtree, node, min_likelihood, min_width=0.1, delta=2):
        detach_node = subtree.child(0)
        bounds = node.parent.time, node.time
        def subtree_likelihood(t):
            likelihood = self.attach(subtree, node, t).likelihood(subtree)
            self.detach(detach_node)
            return likelihood
        result = minimize_scalar(lambda t: -subtree_likelihood(t), bounds=bounds, method='bounded')
        argmax, likelihood = result.x, -result.fun
        interval = None
        if likelihood == -np.inf:
            return bounds
        if likelihood > min_likelihood:
            left, right = min_width, min_width
            l = float('inf')
            while argmax + right < bounds[1] and l > min_likelihood:
                l = subtree_likelihood(argmax + right)
                right *= delta
            right = min(right, bounds[1] - argmax)
            l = float('inf')
            while argmax - left > bounds[0] and l > min_likelihood:
                l = subtree_likelihood(argmax - left)
                left *= delta
            left = min(left, argmax - bounds[0])
            interval = (argmax - left, argmax + right)
        return interval

    cpdef candidate_nodes(self, Node node = None, detach=False):
        if node is None:
            node = self.root
        if node == self.root:
            return [c for child in node.children for c in self.candidate_nodes(child, detach=detach)]
        elif node.is_leaf():
            if detach and node.parent == self.root:
                return []
            return [node]
        else:
            if detach and node.parent == self.root:
                return [c for child in node.children for c in self.candidate_nodes(child, detach=detach)]
            return [node] + [c for child in node.children for c in self.candidate_nodes(child)]

    cpdef likelihood(self, Node node = None):
        if node is None:
            node = self.root
        neighbors = (set(node.children) | {node.parent})
        param = np.zeros([self.dim + 2, self.dim + 2])
        likelihood = 0
        for neighbor in neighbors:
            message = self.get_message(neighbor, node)
            param += message[0]
            likelihood += message[1]
            likelihood += message[2]
        time_factor = 0
        branch_length = node.branch_length()
        if not (node is self.root):
            time_factor = beta(self.alpha, self.beta).logpdf(branch_length / (1 - node.parent.time))
        marginal = Gaussian(param, 'natural')
        return marginal.log_z() + marginal.log_h(np.zeros([self.dim])) + likelihood + time_factor

cdef class Node(object):

    cdef public list children
    cdef public float time
    cdef public Node parent

    def __init__(self, list children, float time, Node parent = None):
        self.children = children
        self.time = time
        self.parent = parent

    cpdef public branch_length(self):
        if self.parent is not None:
            return self.time - self.parent.time

    cpdef public add_child(self, Node child):
        child.set_parent(self)
        self.children.append(child)
        assert child.parent == self
        return self

    cpdef public remove_child(self, Node child):
        assert child in self.children
        child.set_parent(None)
        self.children.remove(child)
        return self

    def add_children(self, *children):
        if len(children) == 0:
            return self
        result = self.add_child(children[0]).add_children(*children[1:])
        for child in self.children:
            assert child.parent == self
        return self

    cpdef fix_time(self, time):
        fraction = (self.time - self.parent.time) / (1 - self.parent.time)
        new_time = time + fraction * (1 - time)
        for child in self.children:
            child.fix_time(new_time)
            assert new_time < child.time
        self.time = new_time

    cpdef set_parent(self, Node parent):
        self.parent = parent
        return self

    cpdef public child(self, i):
        return self.children[i]

    cpdef public bool is_leaf(self):
        return False

    cpdef public bool is_node(self):
        return True

    cpdef public list get_children(self):
        return self.children

    def __hash__(self):
        return hash((self.time, tuple([c for c in self.children])))

    cpdef public copy(self):
        children = []
        new_node = Node([], self.time, parent=self.parent)
        for child in self.children:
            new_child = child.copy()
            new_node.add_child(new_child)
        return new_node

    def __repr__(self):
        return "Node(%s)" % (self.children)

cdef class Leaf(Node):

    cdef public int index 

    def __init__(self, int index, Node parent = None):
        super(Leaf, self).__init__([], 1.0, parent = parent)
        self.index = index

    cpdef public bool is_leaf(self):
        return True

    cpdef public bool is_node(self):
        return False

    cpdef public int get_index(self):
        return self.index

    def __hash__(self):
        return hash(self.index)

    cpdef fix_time(self, time):
        pass

    cpdef public copy(self):
        return Leaf(self.index, parent=self.parent)

    def __repr__(self):
        return "Leaf(%u)" % self.index
