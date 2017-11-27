import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

def plot_tree(tree, ax=None):
    def plot_tree_node(node, size=1):
        if node != tree.get_root():
            if node.is_leaf():
                mean = tree.data[node.index]
            else:
                mean = tree.marginal(node).get_parameters('regular')[1]
            parent_mean = tree.marginal(node.parent).get_parameters('regular')[1]
            assert len(mean) == 2
            plt.plot([mean[0], parent_mean[0]], [mean[1], parent_mean[1]], color='blue')
            if node.is_leaf():
                plt.scatter([mean[0]], [mean[1]])
                return
        for child in node.children:
            plot_tree_node(child)

    plot_tree_node(tree.get_root())
