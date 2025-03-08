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

###############################################################################################
# Error Generators
###############################################################################################


class _ErrorGenerator:

    def generate(self, n):
        raise NotImplementedError(f"{self.__class__} must implement generate")

    def draw(self):
        data= self.generate(10000)
        fig, ax = plt.subplots(1,1)
        sns.kdeplot(data, ax=ax)
        return fig, ax

    def __call__(self, n):
        return self.generate(n)


class NormalErrorGenerator(_ErrorGenerator):

    def __init__(self, mean=0, sd=1):
        self.mean = mean
        self.sd = sd

    def generate(self, n):
        return np.random.normal(self.mean, self.sd, n)


class UniformErrorGenerator(_ErrorGenerator):

    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high

    def generate(self, n):
        return np.random.uniform(self.low, self.high, n)


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


if __name__ == '__main__':
    print("Testing Error Generators")
    print("Testing Normal Error Generator")
    neg = NormalErrorGenerator(0,1)
    fig, ax = neg.draw()