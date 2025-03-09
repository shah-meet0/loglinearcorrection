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
    assert len(X[0]) == 1

    return np.diag(np.abs(X[:, 0]))