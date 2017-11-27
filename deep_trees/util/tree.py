from ..tmc import TMC, Node, Leaf

def init_tree(data, **kwargs):
    def create_tree(idx, t):
        if len(idx) == 1:
            return Leaf(idx[0])
        else:
            mid = len(idx) // 2
            return Node([], t).add_children(
                create_tree(idx[:mid], t + (1 - t) / 2),
                create_tree(idx[mid:], t + (1 - t) / 2),
            )
    return TMC(create_tree(range(data.shape[0]), 0), data, **kwargs)
