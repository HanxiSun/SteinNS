"""

SteinNS: BayesianLogisticRegression_KSD.py

Created on 10/9/18 6:25 PM

@author: Hanxi Sun

"""

import tensorflow as tf
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


########################################################################################################################
# Data

data = scipy.io.loadmat("data/covertype.mat")
X_input = data['covtype'][:, 1:]
y_input = data['covtype'][:, 0]
y_input[y_input == 2] = -1

N_all = X_input.shape[0]
X_input = np.hstack([X_input, np.ones([N_all, 1])])
d = X_input.shape[1]
X_dim = d + 1  # dimension of the target distribution

# split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=21)
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float64)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float64)
y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float64)

N = X_train.shape[0]


########################################################################################################################
# model parameters

lr = 4e-4  # learning rate
kernel = "rbf"  # "rbf" or "imq" kernel

z_dim = 100
h_dim_g = 200

mb_size_x = 100  # date mini-batch size
mb_size = 100  # sample mini-batch size
n_iter = 200000
iter_eval = 1000

optimizer = tf.train.RMSPropOptimizer


########################################################################################################################
# network
tf.reset_default_graph()

initializer = tf.contrib.layers.xavier_initializer()

Xs = tf.placeholder(tf.float64, shape=[None, d])
ys = tf.placeholder(tf.float64, shape=[None])
z = tf.placeholder(tf.float64, shape=[None, z_dim])

G_W1 = tf.get_variable('g_w1', [z_dim, h_dim_g], dtype=tf.float64, initializer=initializer)
G_b1 = tf.get_variable('g_b1', [h_dim_g], dtype=tf.float64, initializer=initializer)
G_W2 = tf.get_variable('g_w2', [h_dim_g, h_dim_g], dtype=tf.float64, initializer=initializer)
G_b2 = tf.get_variable('g_b2', [h_dim_g], dtype=tf.float64, initializer=initializer)
G_W3 = tf.get_variable('g_w3', [h_dim_g, X_dim], dtype=tf.float64, initializer=initializer)
G_b3 = tf.get_variable('g_b3', [X_dim], dtype=tf.float64, initializer=initializer)

theta_G = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]


########################################################################################################################
# functions & structures

def sample_z(m, n, sd=10.):
    return np.random.normal(0, sd, size=[m, n])


def S_q(theta, a0=1, b0=0.01):
    # Reference:
    # https://github.com/DartML/Stein-Variational-Gradient-Descent/blob/master/python/bayesian_logistic_regression.py

    w = theta[:, :-1]  # (m, d)
    s = tf.reshape(theta[:, -1], shape=[-1, 1])  # (m, 1); alpha = s**2

    y_hat = 1. / (1. + tf.exp(- tf.matmul(Xs, tf.transpose(w))))  # (mx, m); shape(Xs) = (mx, d)
    y = tf.reshape((ys + 1.) / 2., shape=[-1, 1])  # (mx, 1)

    dw_data = tf.matmul(tf.transpose(y - y_hat), Xs)  # (m, d)
    dw_prior = - s**2 * w  # (m, d)
    dw = dw_data * N / mb_size_x + dw_prior  # (m, d)

    w2 = tf.reshape(tf.reduce_sum(tf.square(w), axis=1), shape=[-1, 1])  # (m, 1); = wtw
    ds = (2. * a0 - 2 + d) / s - tf.multiply(w2 + 2. * b0, s)  # (m, 1)

    return tf.concat([dw, ds], axis=1)


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
    XY = tf.matmul(x, tf.transpose(x))
    X2_ = tf.reshape(tf.reduce_sum(tf.square(x), axis=1), shape=[tf.shape(x)[0], 1])
    X2 = tf.tile(X2_, [1, tf.shape(x)[0]])
    pdist = tf.subtract(tf.add(X2, tf.transpose(X2)), 2 * XY)  # pairwise distance matrix

    kxy = (c + pdist) ** beta

    coeff = 2 * beta * ((c + pdist) ** (beta-1))
    dxkxy = tf.matmul(coeff, x) - tf.multiply(x, tf.expand_dims(tf.reduce_sum(coeff, axis=1), 1))

    dxykxy_tr = tf.multiply((c + pdist) ** (beta - 2),
                            - 2 * dim * c * beta + (- 4 * beta ** 2 + (4 - 2 * dim) * beta) * pdist)

    return kxy, dxkxy, dxykxy_tr


kernels = {"rbf": rbf_kernel,
           "imq": imq_kernel}

Kernel = kernels[kernel]


def ksd_emp(x, ap=1, dim=X_dim):
    sq = S_q(x, ap)
    kxy, dxkxy, dxykxy_tr = Kernel(x, dim)
    t13 = tf.multiply(tf.matmul(sq, tf.transpose(sq)), kxy) + dxykxy_tr
    t2 = 2 * tf.trace(tf.matmul(sq, tf.transpose(dxkxy)))
    n = tf.cast(tf.shape(x)[0], tf.float64)

    # ksd = (tf.reduce_sum(t13) - tf.trace(t13) + t2) / (n * (n-1))
    ksd = (tf.reduce_sum(t13) + t2) / (n ** 2)

    return ksd


def generator(z):
    G_h1 = tf.nn.tanh(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
    out = 10. * tf.matmul(G_h2, G_W3) + G_b3
    return out


def evaluation(theta, X_t=X_test, y_t=y_test):
    w = theta[:, :-1]
    y = y_t.reshape([-1, 1])
    coff = - np.matmul(y * X_t, w.T)
    prob = np.mean(1. / (1 + np.exp(coff)), axis=1)
    acc = np.mean(prob > .5)
    llh = np.mean(np.log(prob))
    return acc, llh


G_sample = generator(z)

ksd = ksd_emp(G_sample)
solver_KSD = optimizer(learning_rate=lr).minimize(ksd, var_list=theta_G)


#######################################################################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

ksd_loss = np.zeros(n_iter)
acc = np.zeros(1 + (n_iter // iter_eval))
loglik = np.zeros(1 + (n_iter // iter_eval))

for it in range(n_iter):
    batch = [i % N for i in range(it * mb_size_x, (it + 1) * mb_size_x)]

    X_b = X_train[batch, :]
    y_b = y_train[batch]

    _, loss_curr = sess.run([solver_KSD, ksd], feed_dict={Xs: X_b, ys: y_b, z: sample_z(mb_size, z_dim)})

    ksd_loss[it] = loss_curr

    if it % iter_eval == 0:
        post = sess.run(G_sample, feed_dict={z: sample_z(mb_size, z_dim)})
        post_eval = evaluation(post)
        acc[it // iter_eval] = post_eval[0]
        loglik[it // iter_eval] = post_eval[1]

plt.plot(ksd)
plt.axvline(np.argmin(ksd_loss), color="r")
plt.title("KSD loss (min={:.04f} at iter {})".format(np.min(ksd_loss), np.argmin(ksd_loss)))
plt.show()
plt.close()


plt.plot(np.arange(len(acc)) * iter_eval, acc)
plt.ylim(top=0.8)
plt.axhline(0.75, color="g")
plt.title("Accuracy (max={:0.4f} at iter {})".format(np.max(acc), np.argmax(acc)*iter_eval))
plt.show()
plt.close()



