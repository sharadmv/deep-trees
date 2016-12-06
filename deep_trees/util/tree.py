import random

from ..ddt import DDT, Node, Leaf

def create_tree(n):
    leaves = [Leaf(i) for i in range(n)]

    def merge():
        nonlocal leaves
        random.shuffle(leaves)
        a, b = leaves[:2]
        leaves = [Node([a, b])] + leaves[2:]

    while len(leaves) > 1:
        merge()
    return DDT(Node([leaves[0]]))
