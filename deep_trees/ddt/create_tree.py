import random
from ..ddt import DDT, Node, Leaf

def create_tree(n, D):
    leaves = [Leaf(i) for i in range(n)]

    def merge():
        nonlocal leaves
        random.shuffle(leaves)
        a, b = leaves[:2]
        parent = Node([a, b], D)
        a.set_parent(parent)
        b.set_parent(parent)
        leaves = [parent] + leaves[2:]

    while len(leaves) > 1:
        merge()
    root = Node([leaves[0]], D, is_root=True)
    leaves[0].set_parent(root)
    return DDT(root)
