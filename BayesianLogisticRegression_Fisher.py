"""

SteinNS: BayesianLogisticRegression_Fisher.py

Created on 10/11/18 10:25 AM

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
d = X_input.shape[1]  # dimension of w
X_dim = d + 1  # dimension of the target distribution (w and alpha or s, alpha = s ** 2)

# split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=21)
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float64)
X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float64)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float64)
y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float64)

N = X_train.shape[0]


########################################################################################################################
# model parameters

lr = 1e-4  # learning rate
n_D = 2  # number of updates of the discriminator
lbd = 100.  # lambda for the L2 regularization
blow_up = 5  # blow_up (number of replicates of the target)
ini_var_w = 5.0  # initial variance of w
ini_var_s = 0.25  # initial variance of s (alpha = s**2)

z_dim = h_dim_g = h_dim_d = out_dim = X_dim * blow_up

mb_size = 100  # sample mini-batch size
mb_size_x = mb_size * blow_up  # date mini-batch size

n_iter = 200000
iter_eval = 1000

optimizer = tf.train.RMSPropOptimizer

# initial variances
G_scale_init = np.zeros((blow_up, X_dim), dtype=np.float64)
G_scale_init[:, :d] = ini_var_w
G_scale_init[:, -1] = ini_var_s
G_scale_init = G_scale_init.reshape((-1, X_dim * blow_up))


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
G_scale = tf.get_variable('g_scale', initializer=G_scale_init, dtype=tf.float64)

theta_G = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3, G_scale]


D_W1 = tf.get_variable('SD_w1', [X_dim, h_dim_d], dtype=tf.float64, initializer=initializer)
D_b1 = tf.get_variable('SD_b1', [h_dim_d], dtype=tf.float64, initializer=initializer)
D_W2 = tf.get_variable('SD_w2', [h_dim_d, h_dim_d], dtype=tf.float64, initializer=initializer)
D_b2 = tf.get_variable('SD_b2', [h_dim_d], dtype=tf.float64, initializer=initializer)
D_W3 = tf.get_variable('SD_w3', [h_dim_d, X_dim], dtype=tf.float64, initializer=initializer)
D_b3 = tf.get_variable('SD_b3', [X_dim], dtype=tf.float64, initializer=initializer)

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]


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


def generator(z):
    G_h1 = tf.nn.tanh(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
    out = 10. * tf.matmul(G_h2, G_W3) + G_b3
    return out


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    out = (tf.matmul(D_h2, D_W3) + D_b3)
    return out


def diag_gradient(y, x):
    dg = tf.stack([tf.gradients(y[:, i], x)[0][:, i] for i in range(X_dim)], axis=0)
    return tf.transpose(dg)


def evaluation(theta, X_t=X_test, y_t=y_test):
    w = theta[:, :-1]
    y = y_t.reshape([-1, 1])
    coff = - np.matmul(y * X_t, w.T)
    prob = np.mean(1. / (1 + np.exp(coff)), axis=1)
    acc = np.mean(prob > .5)
    llh = np.mean(np.log(prob))
    return acc, llh


G_sample = tf.reshape(generator(z), shape=[-1, X_dim])
D_fake = discriminator(G_sample)

loss1 = tf.expand_dims(tf.reduce_sum(tf.multiply(S_q(G_sample), D_fake), 1), 1)
loss2 = tf.expand_dims(tf.reduce_sum(diag_gradient(D_fake, G_sample), axis=1), 1)
loss = tf.abs(tf.reduce_mean(loss1 + loss2)) - (lbd * tf.reduce_mean(tf.square(D_fake)))

solver_D = optimizer(learning_rate=lr).minimize(-loss, var_list=theta_D)
solver_G = optimizer(learning_rate=lr).minimize(loss, var_list=theta_G)


#######################################################################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

D_loss = np.zeros(n_iter)
G_loss = np.zeros(n_iter)
acc = np.zeros(1 + (n_iter // iter_eval))
loglik = np.zeros(1 + (n_iter // iter_eval))

loss_curr = None
for it in range(n_iter):
    batch = [i % N for i in range(it * mb_size_x, (it + 1) * mb_size_x)]

    X_b = X_train[batch, :]
    y_b = y_train[batch]

    for _ in range(n_D):
        _, loss_curr = sess.run([solver_D, loss], feed_dict={Xs: X_b, ys: y_b, z: sample_z(mb_size, z_dim)})

    D_loss[it] = loss_curr

    _, loss_curr = sess.run([solver_G, loss], feed_dict={Xs: X_b, ys: y_b, z: sample_z(mb_size, z_dim)})

    if it % iter_eval == 0:
        post = sess.run(G_sample, feed_dict={z: sample_z(mb_size, z_dim)})
        post_eval = evaluation(post)
        acc[it // iter_eval] = post_eval[0]
        loglik[it // iter_eval] = post_eval[1]

plt.plot(D_loss)
plt.axvline(np.argmin(D_loss), color="r")
plt.title("D_loss (min={:.04f} at iter {})".format(np.min(D_loss), np.argmin(D_loss)))
plt.show()
plt.close()

plt.plot(G_loss)
plt.axvline(np.argmin(G_loss), color="r")
plt.title("G_loss (min={:.04f} at iter {})".format(np.min(G_loss), np.argmin(G_loss)))
plt.show()
plt.close()

plt.plot(np.arange(len(acc)) * iter_eval, acc)
plt.ylim(top=0.8)
plt.axhline(0.75, color="g")
plt.title("Accuracy (max={:0.4f} at iter {})".format(np.max(acc), np.argmax(acc)*iter_eval))
plt.show()
plt.close()


