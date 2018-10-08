"""

SteinNS: 8Gaussian_KSD.py

Created on 10/7/18 8:40 PM

@author: Hanxi Sun

"""

import tensorflow as tf
import numpy as np
from scipy.stats import multivariate_normal
import math
import matplotlib.pyplot as plt


########################################################################################################################
# target

X_dim = 2  # dimension of the target distribution
r = 15.  # radius of the circle where modes on (equally spaced)
n_comp = 8  # number of mixture component (number of modes)
sd = 1.  # variance of each component

mu = np.array([[r * math.cos((i*2*math.pi)/n_comp), r * math.sin((i*2*math.pi)/n_comp)] for i in range(n_comp)])
Sigma = np.tile(np.array([[sd, 0], [0, sd]]), [n_comp, 1]).reshape(n_comp, X_dim, X_dim)
Sigma_inv = np.linalg.inv(Sigma)
Sigma_det = np.linalg.det(Sigma)
p = np.ones(n_comp) / n_comp  # equal weights

# true sample
show_size = 2000  # number of samples showed each time
label = np.random.choice(n_comp, size=show_size, p=p)[:, np.newaxis]
true_sample = np.sum(np.stack([np.random.multivariate_normal(mu[i], Sigma[i], show_size) * (label == i)
                               for i in range(n_comp)]), 0)


# mmd evaluation with rbf kernel
def mmd_eval(x, y=true_sample):

    def rbf(x, y, h=1.):
        # x, y of shape (n, d)
        xy = np.matmul(x, y.T)
        x2 = np.sum(x ** 2, 1).reshape(-1, 1)
        y2 = np.sum(y ** 2, 1).reshape(1, -1)
        pdist = (x2 + y2) - 2 * xy
        return np.exp(- pdist / h ** 2 / 2.0)  # kernel matrix

    nx, ny = x.shape[0], y.shape[0]
    kxx = rbf(x, x)
    kxy = rbf(x, y)
    kyy = rbf(y, y)
    return np.sum(kxx) / nx / (nx - 1) + np.sum(kyy) / ny / (ny - 1) - 2 * np.sum(kxy) / nx / ny


########################################################################################################################
# model parameters

lr = 2e-4  # learning rate
h_dim_g = 200  # number of hidden neurons per layer of the generator
z_dim = 5  # noise dimension

mb_size = 500  # mini-batch size
n_iter = 20000

optimizer = tf.train.RMSPropOptimizer

kernel = "rbf"  # "rbf" or "imq" kernel


########################################
# plot

iter_display = 1000  # display the sample every iter_display iterations
nsd_x = nsd_y = 8  # number of sd's to be shown on x & y axises
delta = 0.025  # grid

# grids & densities
xlim = [-r - nsd_x*sd, r + nsd_x*sd]
ylim = [-r - nsd_y*sd, r + nsd_y*sd]
X_range, Y_range = np.meshgrid(np.arange(xlim[0], xlim[1], delta),
                               np.arange(ylim[0], ylim[1], delta))
den2d = sum([p[i] * multivariate_normal.pdf(np.stack((X_range, Y_range), 2),
                                            mu[i], Sigma[i])
             for i in range(n_comp)])


def show_plot(sample, method="", is_true_sample=False, it=None, loss=None, mmd=None, title=False,
              fname=None, show_axis=True, show_true=False, show_contour=True, fix_window=True, equal=True,
              c_contour="Red", alpha=0.2, col=['m', 'g']):
    """
    Show samples in plt.
    :param sample: the sample
    :param method: method name
    :param is_true_sample: if the ploted sample is the true sample
    :param it: iteration
    :param loss: current loss
    :param mmd: current mmd with the true sample
    :param title: if the title will be included in the figure
    :param fname: the output file name, if none, then will show the plot directly
    :param show_axis: whether the axises will be shown
    :param show_true: whether the true sample will also be shown
    :param show_contour: whether the contour will also be shown
    :param fix_window: whether the window is fixed to be within xlim & ylim
    :param equal: whether the axises are in the same scale
    :param c_contour: contour (and mode) color
    :param alpha: transparency
    :param col: [true_sample color, fake_sample color]
    :return: None
    """

    x, y = sample[:, 0] * 1., sample[:, 1] * 1.
    adjust = 0.1  # adjust for fixed windows
    lvls = np.linspace(1e-3, np.max(den2d), 7)  # levels for contour plot

    if is_true_sample:
        ttl = "One sample from the target distribution"
        c = col[0]
    else:
        c = col[1]
        ttl = method
        if show_true or show_contour:
            ttl += " vs true"
        if it is not None:
            ttl += " at iter {}".format(it)
        if loss is not None:
            ttl += ", loss={:.03f}".format(loss)
        if mmd is not None:
            ttl += ", mmd={:.03f}".format(mmd)
    if fix_window:
        show = (x < xlim[0]) | (x > xlim[1]) | (y < ylim[0]) | (y > ylim[1])
        x[x < xlim[0]] = xlim[0] + adjust
        x[x > xlim[1]] = xlim[1] - adjust
        y[y < ylim[0]] = ylim[0] + adjust
        y[y > ylim[1]] = ylim[1] - adjust
        ttl += ", {:.02f}% showed".format((1 - (sum(show)/sample.shape[0]))*100)

    if title:
        plt.title(ttl)
    if show_contour:
        plt.contour(X_range, Y_range, den2d, levels=lvls, colors=c_contour, linewidths=.5)
    if show_true:
        plt.scatter(true_sample[:, 0], true_sample[:, 1], color=col[0], alpha=alpha, s=10, label="True Sample")

    plt.scatter(x, y, color=c, alpha=alpha, s=10, label="Fake Sample")

    if show_true:
        plt.legend(loc="upper left")

    plt.scatter(mu[:, 0], mu[:, 1], s=10, color=c_contour)

    if fix_window:
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.axhline(ylim[0], color='black', alpha=.1)
        plt.axhline(ylim[1], color='black', alpha=.1)
        plt.axvline(xlim[0], color='black', alpha=.1)
        plt.axvline(xlim[1], color='black', alpha=.1)

    if not show_axis:
        ax = plt.axes()
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)

    if equal:
        ax = plt.axes()
        ax.set_aspect('equal')

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, format="pdf", bbox_inches='tight', pad_inches=0)

    plt.close()


show_plot(true_sample, is_true_sample=True, title=True, fix_window=False)


########################################################################################################################
# network parameters
tf.reset_default_graph()

initializer = tf.contrib.layers.xavier_initializer()

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

mu_tf = tf.convert_to_tensor(mu, dtype=tf.float32)
Sigma_inv_tf = tf.convert_to_tensor(Sigma_inv, dtype=tf.float32)
p_tf = tf.reshape(tf.convert_to_tensor(p, dtype=tf.float32), shape=[n_comp, 1])


G_W1 = tf.get_variable('g_w1', [z_dim, h_dim_g], dtype=tf.float32, initializer=initializer)
G_b1 = tf.get_variable('g_b1', [h_dim_g], initializer=initializer)
G_W2 = tf.get_variable('g_w2', [h_dim_g, h_dim_g], dtype=tf.float32, initializer=initializer)
G_b2 = tf.get_variable('g_b2', [h_dim_g], initializer=initializer)
G_W3 = tf.get_variable('g_w3', [h_dim_g, X_dim], dtype=tf.float32, initializer=initializer)
G_b3 = tf.get_variable('g_b3', [X_dim], initializer=initializer)
G_scale = tf.get_variable('g_scale', [1, X_dim], initializer=tf.constant_initializer(10.))
G_location = tf.get_variable('g_location', [1, X_dim], initializer=tf.constant_initializer(30.))

theta_G = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3, G_scale, G_location]


########################################################################################################################
# functions & structures

def sample_z(m, n, std=10.):
    s1 = np.random.normal(0, std, size=[m, n])
    # s1 = np.random.uniform(-sd, sd, size=[m, n])
    return s1


def log_densities(xs):

    log_den0 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[0], Sigma_inv_tf[0]),
                                        tf.transpose(xs - mu_tf[0]))) / 2 - np.log(Sigma_det[0]) / 2
    log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[1], Sigma_inv_tf[1]),
                                        tf.transpose(xs - mu_tf[1]))) / 2 - np.log(Sigma_det[1]) / 2
    log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[2], Sigma_inv_tf[2]),
                                        tf.transpose(xs - mu_tf[2]))) / 2 - np.log(Sigma_det[2]) / 2
    log_den3 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[3], Sigma_inv_tf[3]),
                                        tf.transpose(xs - mu_tf[3]))) / 2 - np.log(Sigma_det[3]) / 2
    log_den4 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[4], Sigma_inv_tf[4]),
                                        tf.transpose(xs - mu_tf[4]))) / 2 - np.log(Sigma_det[4]) / 2
    log_den5 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[5], Sigma_inv_tf[5]),
                                        tf.transpose(xs - mu_tf[5]))) / 2 - np.log(Sigma_det[5]) / 2
    log_den6 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[6], Sigma_inv_tf[6]),
                                        tf.transpose(xs - mu_tf[6]))) / 2 - np.log(Sigma_det[6]) / 2
    log_den7 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[7], Sigma_inv_tf[7]),
                                        tf.transpose(xs - mu_tf[7]))) / 2 - np.log(Sigma_det[7]) / 2
    ld = tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p[0]) + log_den0, np.log(p[1]) + log_den1,
                                                      np.log(p[2]) + log_den2, np.log(p[3]) + log_den3,
                                                      np.log(p[4]) + log_den4, np.log(p[5]) + log_den5,
                                                      np.log(p[6]) + log_den6, np.log(p[7]) + log_den7], 0), 0), 1)
    return ld


def S_q(xs):
    return tf.gradients(log_densities(xs), xs)[0]


def generator(z):
    G_h1 = tf.nn.tanh(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.matmul(G_h2, G_W3) + G_b3
    out = tf.multiply(G_h3, G_scale) + G_location
    return out


def rbf_kernel(x, dim=X_dim, h=1.):
    # Reference 1: https://github.com/ChunyuanLI/SVGD/blob/master/demo_svgd.ipynb
    # Reference 2: https://github.com/yc14600/svgd/blob/master/svgd.py
    XY = tf.matmul(x, tf.transpose(x))
    X2_ = tf.reshape(tf.reduce_sum(tf.square(x), axis=1), shape=[tf.shape(x)[0], 1])
    X2 = tf.tile(X2_, [1, tf.shape(x)[0]])
    pdist = tf.subtract(tf.add(X2, tf.transpose(X2)), 2 * XY)  # pairwise distance matrix

    kxy = tf.exp(- pdist / h ** 2 / 2.0)  # kernel matrix

    sum_kxy = tf.expand_dims(tf.reduce_sum(kxy, axis=1), 1)
    dxkxy = tf.add(-tf.matmul(kxy, x), tf.multiply(x, sum_kxy)) / (h ** 2)  # sum_y dk(x, y)/dx

    dxykxy_tr = tf.multiply((dim * (h**2) - pdist), kxy) / (h**4)  # tr( dk(x, y)/dxdy )

    return kxy, dxkxy, dxykxy_tr


def imq_kernel(x, dim=X_dim, beta=-.5, c=1.):
    # IMQ kernel
    XY = tf.matmul(x, tf.transpose(x))
    X2_ = tf.reshape(tf.reduce_sum(tf.square(x), axis=1), shape=[tf.shape(x)[0], 1])
    X2 = tf.tile(X2_, [1, tf.shape(x)[0]])
    pdist = tf.subtract(tf.add(X2, tf.transpose(X2)), 2 * XY)  # pairwise distance matrix

    kxy = (c + pdist) ** beta

    coeff = 2 * beta * (c + pdist) ** (beta-1)
    dxkxy = tf.matmul(coeff, x) - tf.multiply(x, tf.expand_dims(tf.reduce_sum(coeff, axis=1), 1))

    dxykxy_tr = tf.multiply((c + pdist) ** (beta - 2),
                            - 2 * dim * c * beta + (- 4 * beta ** 2 + (4 - 2 * dim) * beta) * pdist)

    return kxy, dxkxy, dxykxy_tr


kernels = {"rbf": rbf_kernel,
           "imq": imq_kernel}

Kernel = kernels[kernel]


def ksd_emp(x, dim=X_dim):
    sq = S_q(x)
    kxy, dxkxy, dxykxy_tr = Kernel(x, dim)
    t13 = tf.multiply(tf.matmul(sq, tf.transpose(sq)), kxy) + dxykxy_tr
    t2 = 2 * tf.trace(tf.matmul(sq, tf.transpose(dxkxy)))
    n = tf.cast(tf.shape(x)[0], tf.float32)

    # ksd = (tf.reduce_sum(t13) - tf.trace(t13) + t2) / (n * (n-1))
    ksd = (tf.reduce_sum(t13) + t2) / (n ** 2)

    return ksd


G_sample = generator(z)

ksd = ksd_emp(G_sample)
solver_KSD = optimizer(learning_rate=lr).minimize(ksd, var_list=theta_G)


#######################################################################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses = np.zeros(n_iter)
mmds = np.zeros(1 + (n_iter // iter_display))
for it in range(n_iter):

    _, loss_curr = sess.run([solver_KSD, ksd],
                            feed_dict={z: sample_z(mb_size, z_dim)})
    losses[it] = loss_curr

    if it % iter_display == 0:
        print("iter:", it, ", {:.04f},".format(loss_curr))
        samples = sess.run(G_sample, feed_dict={z: sample_z(show_size, z_dim)})
        mmd_curr = mmd_eval(samples)
        mmds[it // iter_display] = mmd_curr
        show_plot(samples, "KSD", it=it, loss=loss_curr, mmd=mmd_curr, fname=None, title=True, fix_window=True)

plt.plot(losses)
plt.title("loss (min={} at iter {})".format(np.min(losses), np.argmin(losses)))
plt.show()
plt.close()

plt.plot(np.arange(len(mmds)) * iter_display, mmds)
plt.title("mmd (min={} at iter {})".format(np.min(mmds), np.argmin(mmds) * iter_display))
plt.show()
plt.close()




