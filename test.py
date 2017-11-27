import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from deep_trees.tmc import Node, Leaf, TMC, slice_sample
from deep_trees.util import plot_tree, init_tree
from deep_trees.stats import Gaussian

data = np.array([
    [-0.5, 0.5],
    [0.5, -0.5],
    [-0.3, 0.0],
    [0.3, 0.0],
    [-0.5, -0.5],
    [0.5, 0.5],
]).astype(np.float64) * 20

data = np.random.normal(size=(10, 2)) * 20

tree = init_tree(data)
# TMC(Node([], 0.0).add_children(
    # Node([], 0.25).add_children(
        # Node([], 0.5).add_children(
            # Leaf(0),
            # Leaf(1),
        # ),
        # Node([], 0.5).add_children(
            # Leaf(2),
            # Leaf(3),
        # ),
    # ),
    # Node([], 0.5).add_children(
        # Leaf(4),
        # Leaf(5),
    # ),
# ), data)

def iter():
    slice_sample(tree)
    plt.cla()
    plot_tree(tree)
    plt.draw()
    plt.pause(0.001)

plt.ion()
plt.figure()
plot_tree(tree)
plt.show()
[iter() for _ in range(100)]
