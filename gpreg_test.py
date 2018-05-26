"""
This is a simple visual test for gpreg.py on synthetic data sampled from a Gaussian process (GP).
"""


import numpy as np
import gpreg
import matplotlib.pyplot as plt
import unittest


__author__ = 'pgeiger'


class test_gpreg(unittest.TestCase):

    def setUp(self):

        # Instantiate gp and specify prior:

        prior_mean_fun = gpreg.zero_mean_fun
        prior_kernel = lambda u, v: gpreg.kernels.rbf(u, v, sigma=1)
        additive_noise_sigma = 0.1

        self.gp = gpreg.gp(prior_mean_fun, prior_kernel, additive_noise_sigma)
        self.gp.matrix_inversion_method = 'cholesky'

        # Draw sample (="ground truth"):

        self.grid_size = 101
        self.grid = np.linspace(-10, 10, self.grid_size)
        self.mean, self.cov = self.gp.eval_prior(self.grid)

        self.sample = np.random.multivariate_normal(self.mean, self.cov)  # Sample Y-coordinates from GP prior

        # Select training set:

        self.training_sample_size = 10
        self.training_indices = np.random.choice(range(self.grid_size), size=self.training_sample_size, replace=False)
        # training_indices = np.random.choice(range(50), size=training_sample_size, replace=False)

        self.X_train = self.grid[self.training_indices]
        self.Y_train = self.sample[self.training_indices]

        self.X_test = self.grid
        # self.X_test = np.array([u for u in range(grid_size) if not u in training_indices ])

    def test_predictive_posterior(self):

        # Fit GP to subsample and calculate posterior over test sample:

        posterior_mean, posterior_cov = self.gp.predictive_posterior(self.X_train, self.Y_train, self.X_test)

        # Plot posterior mean and variance:

        plt.clf()
        plt.plot(self.grid, self.sample, 'blue')  # complete ground truth
        plt.plot(self.X_train, self.Y_train, 'bo')  # training points
        plt.plot(self.X_test, posterior_mean, 'red')  # test point prediction
        plt.fill_between(self.X_test, posterior_mean - np.diagonal(posterior_cov), posterior_mean + np.diagonal(posterior_cov),
                         color='lightgray')
        plt.show()
        print('Posterior mean (solid red) should be close to "ground truth" (solid blue) and have low variance ' 
              '(light gray area), at least around training samples (o).')


if __name__ == '__main__':
    unittest.main()
