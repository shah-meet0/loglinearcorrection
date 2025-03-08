###############################################################################################
# This module generates various simulated data for the purpose of testing.
# Each function generates a different type of data.
# Author: Meet Shah
###############################################################################################

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats

###############################################################################################
# Error Generators\
# TODO: Replace with scipy.stats implementation
###############################################################################################


class _ErrorGenerator:

    def __init__(self):
        self.dist = None

    def generate(self, n):
        return self.dist.rvs(size=n)

    def draw(self):
        x = np.linspace(self.dist.ppf(0.01), self.dist.ppf(0.99), 100)
        y = self.dist.pdf(x)
        fig, ax = plt.subplots(1,1)
        sns.kdeplot(x=x, y=y, ax=ax)
        return fig, ax

    def __call__(self, n):
        return self.generate(n)


class NormalErrorGenerator(_ErrorGenerator):

    def __init__(self, mean=0, sd=1):
        super().__init__()
        self.dist = stats.norm(loc=mean, scale=sd)
        self.mean = mean
        self.sd = sd


class UniformErrorGenerator(_ErrorGenerator):

    def __init__(self, low=0, high=1):
        super().__init__()
        self.dist = stats.uniform(low=low, high=high)
        self.low = low
        self.high = high


class HeteroscedasticErrorGenerator(_ErrorGenerator):

    def __init__(self, mean=np.array([0]), cov=np.array([[1]])):
        super().__init__()
        self.dist = stats.multivariate_normal(mean=mean, cov=cov)
        self.mean = mean
        self.cov = cov

    def generate(self, n=1):
        return super().generate(n)

    def draw(self):
        data = self.generate(1000)
        fig, ax = plt.subplots(1,1)
        n = len(self.mean)
        for i in range(n):
            sns.kdeplot(data[:, i], ax=ax)

    def __call__(self, n=1):
        return self.generate(n)



###############################################################################################
# Random Data Generators
###############################################################################################

class _RandomDataGenerator:

    def generate(self, n):
        pass


class MVNDataGenerator(_RandomDataGenerator):

    def __init__(self, means, sigma):
        self.means = means
        self.sigma = sigma

        if len(means) != sigma.shape[0] or len(means) != sigma.shape[1]:
            raise ValueError("Variance Covariance Matrix should be square and of size equal to number of means")

    def generate(self, n):
        return np.random.multivariate_normal(mean=self.means, cov=self.sigma, size=n)

    def get_feature_space_size(self):
        return len(self.means)


class BinaryDataGenerator(_RandomDataGenerator):

    def __init__(self, n_features, p):
        self.n_features = n_features
        self.p = p

    def generate(self, n):
        return np.random.binomial(1, self.p, size=(n, self.n_features))

    def get_feature_space_size(self):
        return self.n_features


class DependentBinaryDataGenerator(_RandomDataGenerator):
    # To be implemented if needed
    pass


###############################################################################################
# Data Generating Processes
###############################################################################################

# TODO: REVAMP THIS CLASS
class RCT:

    def __init__(self, treatment_effect, n_regressors,  intercept, means, sigma, betas, p_treated= 0.5):
        """
        :param treatment_effect: Value of treatment effect
        :param n_regressors: Number of regressors
        :param intercept: Value of intercept
        :param means: means of regressors (array of size n_regressors)
        :param sigma: variance covariance matrix of regressors (n_regressors x n_regressors)
        :param betas: coefficients of regressors (array of size n_regressors)
        :param p_treated: Probability of getting treated
        """
        self.te = treatment_effect
        self.n_regressors = n_regressors
        self.p = p_treated
        self.c = intercept
        self.means = means
        self.sigma = sigma
        self.betas = betas

    def generate_data(self, n, error_generator= NormalErrorGenerator(0,1)):
        if self.n_regressors == 0:
            X = np.array([])
        else:
            if self.n_regressors != len(self.means):
                raise ValueError("Number of means should be equal to number of regressors")
            if self.n_regressors != self.sigma.shape[0]:
                raise ValueError("Number of rows of sigma should be equal to number of regressors")
            if self.n_regressors != self.sigma.shape[1]:
                raise ValueError("Number of columns of sigma should be equal to number of regressors")

            X = np.random.multivariate_normal(mean=self.means, cov=self.sigma, size=n)

