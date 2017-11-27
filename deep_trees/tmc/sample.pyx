import numpy as np
import random

def slice_sample(tree, slide_delta=1.1, slice_width=0.1):
    return slice(tree, slide_delta, slice_width)

cdef slice(tree, slide_delta, slice_width):
    detach_node = random.choice(tree.candidate_nodes(detach=True))
    sibling = list(set(detach_node.parent.children) - {detach_node})[0]
    likelihood = tree.likelihood(detach_node.parent)
    min_likelihood = np.log(np.random.uniform(0, 1)) + likelihood

    subtree = tree.detach(detach_node)
    subtree_time = subtree.time

    candidates = tree.candidate_nodes()
    intervals = [tree.find_interval(subtree, candidate, min_likelihood, delta=slide_delta, min_width=slice_width) for candidate in candidates]
    if all([i is None for i in intervals]):
        tree.attach(subtree, sibling, subtree_time)
        return
    candidates, intervals = zip(*[(a, b) for a, b in zip(candidates, intervals) if b is not None])
    weights = np.array([(i[1] - i[0]) if i is not None else 0.0 for i in intervals])
    weights /= weights.sum()
    u = np.random.random()
    choice = None
    prev = 0
    for i, weight in enumerate(np.cumsum(weights)):
        if u > weight:
            prev = weight
            continue
        choice = i
        p = (u - prev) / (weight - prev)
        break
    regraft_choice = candidates[choice]
    print(intervals[choice], p)
    time = intervals[choice][0] + p * (intervals[choice][1] - intervals[choice][0])
    print("Attaching subtree", time, regraft_choice, regraft_choice.time)
    tree.attach(subtree, regraft_choice, time)
