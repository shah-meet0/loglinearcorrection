#################################################################################
# This file contains functions that are used to generate dependence functions.
# Author: Meet Shah
#################################################################################



import numpy as np


def constant_mean(c):
    def f(x):
        return c * np.ones(len(x))

    return f


def constant_variance(c):
    def f(x):
        return c * np.eye(len(x))
    return f

def independent_absolute(X):
    return np.abs(X[:, 0])


def independent_squared(X):
    return X[:, 0] ** 2