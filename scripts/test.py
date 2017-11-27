import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from argparse import ArgumentParser
import tensorflow as tf
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

from deepx import T
from deepx.nn import *

from deep_trees.prior import DDTPrior, GaussianPrior
from deep_trees.util import load_mnist, sample_minibatch
from deep_trees.likelihood import VAE, BernoulliLikelihoodModel

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--subset_size', type=int, default=1000)
    argparser.add_argument('--embedding_size', type=int, default=20)
    argparser.add_argument('--default_device', default='/cpu:0')
    argparser.add_argument('--no-tree', action='store_true')
    argparser.add_argument('--batch_size', type=int, default=100)
    return argparser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    T.set_default_device(args.default_device)

    # X, y = load_mnist(args.subset_size, one_hot=False)
    X, y = np.eye(args.subset_size).astype(np.float32), np.ones(args.subset_size)
    N, D = X.shape

    p_network = Tanh(5)
    q_network = Tanh(5)

    print("Creating tree prior...")
    if args.no_tree:
        tree_prior = GaussianPrior(N, args.embedding_size)
    else:
        tree_prior = DDTPrior(N, args.embedding_size, c=0.1, sigma0=1)
    print("Creating likelihood model...")
    likelihood_model = BernoulliLikelihoodModel(args.embedding_size, D, p_network)
    vae = VAE(D, args.embedding_size, q_network, likelihood_model)

    batch_indices = T.vector(dtype='int32')
    batch = T.matrix()
    batch_noise = T.matrix()

    z_sample = vae.sample_z(batch, batch_noise)
    encodings = vae.encode(batch)
    log_likelihood = likelihood_model.log_likelihood(batch, z_sample)
    log_likelihood_z = vae.log_likelihood(z_sample, batch)
    log_prior = tree_prior.log_prior(z_sample)
    lower_bound = log_prior + log_likelihood - log_likelihood_z


    all_encodings = vae.encode(X)
    tree_likelihood = tree_prior.log_prior(all_encodings)

    bound_summary = tf.merge_summary([
        tf.summary.scalar("ELBO", lower_bound),
        tf.summary.scalar("pz", log_prior),
        tf.summary.scalar("px_z", log_likelihood),
        tf.summary.scalar("qz_x", log_likelihood_z),
    ])

    trainable = tf.trainable_variables()
    # for parameter in vae.q_network.get_graph_parameters():# + likelihood_model.p_network.get_graph_parameters():
        # trainable.remove(parameter)

    # trainable = [v for v in trainable if 'time' not in v.name]
    # time_vars = [v for v in tf.trainable_variables() if 'time' in v.name]

    train_op = tf.train.AdamOptimizer(0.01).minimize(-lower_bound, var_list=trainable)
    # time_op = tf.train.AdamOptimizer(0.01).minimize(-lower_bound, var_list=time_vars)

    def plot_node(ax, node, root=True, size=40.0):
        if node.is_leaf():
            value = es[node.get_index()]
            ax.scatter(value[0], value[1], s=size, color='red')
            ax.text(value[0], value[1], str(node.get_index()))
        else:
            value = pca_values[node.get_node_id()]
            if root:
                ax.scatter(value[0], value[1], s=size, color='purple')
            else:
                ax.scatter(value[0], value[1], s=size)
            ax.text(value[0], value[1], "%.3f" % times[node.get_node_id()])
        for child in node.get_children():
            plot_node(ax, child, root=False, size=size - 1)
            if child.is_leaf():
                child_value = es[child.get_index()]
            else:
                child_value = pca_values[child.get_node_id()]
            ax.plot(*zip(value, child_value), color='g', alpha=0.2)


    fig, ax = plt.subplots(1, 2)
    plt.ion()
    plt.show()

    with T.session() as sess:
        writer = tf.train.SummaryWriter("./logs", sess.graph)
        full_info = tree_prior.get_info(range(N))
        for i in range(40000):
            batch_idx = sample_minibatch(N, args.batch_size)
            batch_ = X[batch_idx]
            batch_noise_ = np.random.normal(size=(args.batch_size, args.embedding_size))
            info = tree_prior.get_info(batch_idx)
            loss = sess.run([train_op, lower_bound, bound_summary, encodings], feed_dict={
                batch: batch_,
                batch_noise: batch_noise_,
                **info
            })
            writer.add_summary(loss[2], i + 1)
            tree_ll = sess.run(tree_likelihood, feed_dict={
                **full_info
            })
            print("Loss[%u]: %.3f" % (i + 1, loss[1]))
            # candidate_tree = tree_prior.hill_climb()
            # print(candidate_tree)

            if i % 500 == 0:
                embeds = sess.run(encodings, feed_dict={batch: X})
                if args.embedding_size > 2:
                    pca = PCA(2).fit(embeds)
                    es = pca.transform(embeds)
                else:
                    es = embeds
                if not args.no_tree:
                    ax[0].cla()
                    values = T.get_value(tree_prior.values)
                    times = T.get_value(tree_prior.times)
                    if args.embedding_size > 2:
                        pca_values = pca.transform(values)
                    else:
                        pca_values = values
                    plot_node(ax[0], tree_prior.tree.get_root())

                ax[1].cla()
                ax[1].scatter(es[:, 0], es[:, 1], c=y, cmap=ListedColormap(sns.color_palette("hls", 10)), alpha=0.8)
                # ax[1].scatter(es[:, 0], es[:, 1])
                plt.legend(loc='best')
                plt.pause(0.001)
                plt.draw()
