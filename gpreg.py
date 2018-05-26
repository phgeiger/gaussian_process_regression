"""
This is a preliminary implementation of Gaussian process (GP) regression.

Notes:
- This is a preliminary version, only tested in a limited amount of cases. Not meant for public usage.
- Concerning error "not positive definite": sometimes this error occurs (from the cholesky decomposition function).
I checked, for the linear kernel, that there is no bug in the kernel function (by comparing it to simply the product
X X^T). So it seems that this is a numerical issue, possibly occurring due to similar datapoints (-> similar rows).
Adding more additive noise helps usually.

Background:
- The problem this method addresses is the "regression problem": given training pairs (X, Y), and one (or several) test 
  X, predict the corresponding Y from them.
- It is based on Carl Rasmussen's "Gaussian Processes for Machine Learning", eq. 2.22-24 and Algorithm 2.1
"""


import numpy as np
import scipy as sp
import scipy.sparse.linalg as sp_sparse_linalg


__author__ = 'pgeiger'


# Prior means, kernels and their eval. on samples
# -----------------------------------------------


class kernels:
    """
    Collection of some standard prior kernels k(u, v).
    """

    @staticmethod
    def rbf(u, v, sigma=1):
        return np.exp( - float(1)/(2*np.square(sigma)) * np.square(np.linalg.norm(u - v)) )

    @staticmethod
    def l2(u, v):
        return np.dot(u, v)

    @staticmethod
    def l2_const(u, v):
        return np.dot(u, v) + 1


def zero_mean_fun(u):
    return 0


def eval_mean_fun(mean_fun, u):
    """
    Evaluate mean function on finite set of data points.
    
    Args:
    :param mean_fun: mean function
    :param u: data points
    :return: (mean_fun(u_i))_i
    """

    length = len(u)

    res = np.zeros(length)

    for i in range(length):
        res[i] = mean_fun(u[i])

    return res


def eval_kernel(kernel, *args):
    """
    Evaluate kernel on finite set of data points.
    
    Args:
    :param args: column vector X, optionally a column vector Y
    :return: kernel matrix (k(x_i,x_j))_{ij} or (k(x_i,y_j))_{ij}, respectively
    """

    if len(args) == 1:
        X_1 = args[0]
        X_2 = args[0]
    elif len(args) == 2:
        X_1 = args[0]
        X_2 = args[1]

    K = np.zeros((len(X_1), len(X_2)))
    for i, x_1 in enumerate(X_1):
        for j, x_2 in enumerate(X_2):
            #print kernel(x_1, x_2)
            K[i][j] = kernel(x_1, x_2)
    assert isinstance(K, object)
    return K


# "gp"-object
# -----------


class gp:
    """
    This class pools all parts of our (Bayesian) model/method: prior, likelihood, measurements,
    and calculation of posterior. This object-oriented structure is inspired by machine learning libraries 
    such as GPy from U Sheffield.

    Underlying model from which it is derived:
    Y_i = f(X_i) + N_i, f ~ GP(prior_mean_fun, prior_kernel), N_i ~ Gauss(0, additive_noise_sigma)
    """

    def __init__(self, prior_mean_fun=zero_mean_fun, prior_kernel=kernels.rbf, additive_noise_sigma=0.1):
        """        
        Args:
        :param prior_mean_fun: prior mean
        :param prior_kernel: prior kernel (without additive noise)
        :param additive_noise_sigma: variance of the additive noise N_i
        """

        self.prior_mean_fun = prior_mean_fun
        # Note that this is gives the prior kernel for noise-free measurements. The probabilistic model that includes
        # the data actually contains additional noise for the measurements. This is accounted for by in the calculation
        # of the posterior below, by adding a noise covariance to the prior kernel for the training points.
        self.prior_kernel = prior_kernel
        self.additive_noise_sigma = additive_noise_sigma

        # Remaining internal attributes and parameters:

        self.X_training = np.array([])
        self.Y_training = np.array([])
        self.X_test = np.array([])

        self.posterior_mean = None
        self.posterior_cov = None

        self.matrix_inversion_method = 'cholesky'
        self.conjugate_gradients_maxiter = None

    def eval_prior(self, X_test):
        return eval_mean_fun(self.prior_mean_fun, X_test), eval_kernel(self.prior_kernel, X_test)

    def sample_from_prior(self, X_test):
        return np.random.multivariate_normal(*self.eval_prior(X_test))

    def predictive_posterior(self, X_training=None, Y_training=None, X_test=None, matrix_inversion_method=None):
        """
        Use training data and test points given as arguments to calculate posterior mean and covariance on test points.

        Notes:
        - This is based on the 2013/2014 "Intelligent Systems I" lecture script by Philipp Hennig as well as
        Carl Rasmussen's GPML, eq. 2.22-24 and Algorithm 2.1, for the formula
        (this is also where I took the variable names "L", "alpha" and "v" from).
        
        Args:
        :param X_training: X-coordinates of training points
        :param Y_training: Y-coordinates of training points
        :param X_test: where to predict
        :param matrix_inversion_method: 'cholesky' (standard) or 'conjugate_gradients' (not well-tested yet)
        :return: (posterior mean, posterior covariance)
        """

        X_training = X_training if X_training is not None else self.X_training
        Y_training = Y_training if Y_training is not None else self.Y_training
        training_length = len(X_training)
        X_test = X_test if X_test is not None else self.X_test
        test_length = len(X_test)
        matrix_inversion_method = matrix_inversion_method if matrix_inversion_method is not None \
            else self.matrix_inversion_method

        self.X_training = X_training
        self.Y_training = Y_training
        self.X_test = X_test

        k_tr = eval_kernel(self.prior_kernel, X_training)
        k_te = eval_kernel(self.prior_kernel, X_test)
        k_tr_te = eval_kernel(self.prior_kernel, X_training, X_test)
        k_te_tr = k_tr_te.T
        M = k_tr + (self.additive_noise_sigma ** 2) * np.identity(training_length)  # Include additive noise in kernel

        if matrix_inversion_method == 'cholesky':

            L = np.linalg.cholesky(M)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_training - eval_mean_fun(self.prior_mean_fun, X_training)))
            v = np.linalg.solve(L, k_tr_te)
            self.posterior_mean = eval_mean_fun(self.prior_mean_fun, X_test) + np.dot(k_te_tr, alpha)
            self.posterior_cov = k_te - np.dot(v.T, v)

        elif matrix_inversion_method == 'conjugate_gradients':

            if len(X_test) > 1:
                raise ValueError('If matrix inversion method is conjugate_gradients, '
                                 'then only individual test points can be predicted.')

            else:
                alpha = sp_sparse_linalg.cg(M, Y_training)[0] if self.conjugate_gradients_maxiter is None \
                    else sp_sparse_linalg.cg(M, Y_training, maxiter=self.conjugate_gradients_maxiter)[0]
                # TODO: ask philipp hennig how to do CGS for many training samples
                w = sp_sparse_linalg.cg(M, k_tr_te)[0] if self.conjugate_gradients_maxiter is None \
                    else sp_sparse_linalg.cg(M, k_tr_te, maxiter=self.conjugate_gradients_maxiter)[0]
                self.posterior_mean = eval_mean_fun(self.prior_mean_fun, X_test) + np.dot(k_te_tr, alpha)
                self.posterior_cov = k_te - np.dot(k_tr_te.T, w)

        return self.posterior_mean, self.posterior_cov
