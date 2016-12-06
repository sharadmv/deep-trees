from argparse import ArgumentParser
from deepx import T

from deep_trees.prior import DDTPrior
from deep_trees.util import load_mnist

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--subset_size', type=int, default=100)
    argparser.add_argument('--embedding_size', type=int, default=10)
    argparser.add_argument('--default_device', default='/cpu:0')
    return argparser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    T.set_default_device(args.default_device)

    X = load_mnist(args.subset_size)
    N, D = X.shape

    tree_prior = DDTPrior(N, args.embedding_size)
