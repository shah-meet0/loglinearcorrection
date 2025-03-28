###############################################################################################
# This module generates various simulated data for the purpose of testing.
# Each function generates a different type of data.
# Author: Meet Shah
###############################################################################################

import numpy as np
import scipy.stats as stats
from .dependence_funcs import constant_mean, constant_variance

###############################################################################################
# Error Generators
###############################################################################################


class _ErrorGenerator:
    def __init__(self):
        pass

    def generate(self, x):
        pass

    def __call__(self, x):
        return self.generate(x)

    def expected_value(self, x, func, trials=10000):
        draws = np.array([func(self.generate(x)) for _ in range(trials)])
        return np.mean(draws, axis=0)


class NormalErrorGenerator(_ErrorGenerator):

    def __init__(self, mean_fn=constant_mean(0), cov_fn=constant_variance(1)):
        """
        :param mean_fn: Converts data into a mean function of length n
        :param cov_fn: Converts data into a variance covariance matrix of size nxn
        """
        super().__init__()

        self.mean_fn = mean_fn
        self.cov_fn = cov_fn

    def generate(self, x):
        n = len(x)
        means = self.mean_fn(x)
        cov = self.cov_fn(x)

        if len(means) != n:
            raise ValueError(f"Mean function returns vector of length {len(means)} but expected {n}")

        if len(cov) != n or len(cov[0]) != n:
            raise ValueError(f"Covariance matrix has shape {np.shape(cov)} but expected ({n}, {n})")

        rng = np.random.default_rng()
        return rng.multivariate_normal(means, cov, method='cholesky')


class IndependentNormErrorGenerator(_ErrorGenerator):

    def __init__(self, mean_fn=constant_mean(0), var_fn=constant_variance(1)):
        super().__init__()
        self.mean_fn = mean_fn
        self.var_fn = var_fn

    def generate(self, x):
        n = len(x)
        means = self.mean_fn(x)
        var = self.var_fn(x)

        if len(means) != n:
            raise ValueError(f"Mean function returns vector of length {len(means)} but expected {n}")

        if len(var) != n:
            raise ValueError(f"Variance vector has shape {np.shape(var)} but expected ({n})")

        return means + np.sqrt(var) * np.random.randn(n)

class IndependentLogNormErrorGenerator(_ErrorGenerator):

    def __init__(self, mean_fn=constant_mean(0), var_fn=constant_variance(1)):
        super().__init__()
        self.mean_fn = mean_fn
        self.var_fn = var_fn

    def generate(self, x):
        n = len(x)
        means = self.mean_fn(x)
        var = np.diagonal(self.var_fn(x))

        if len(means) != n:
            raise ValueError(f"Mean function returns vector of length {len(means)} but expected {n}")

        if len(var) != n:
            raise ValueError(f"Variance matrix has shape {np.shape(var)} but expected ({n})")

        return np.random.lognormal(means, np.sqrt(var))


###############################################################################################
# Random Data Generators
###############################################################################################

# TODO: Add more methods here, for example visualization

class _RandomDataGenerator:

    def __init__(self):
        self.dist = None

    def generate(self, n):
        pass

    def __call__(self, n):
        return self.generate(n)


class MVNDataGenerator(_RandomDataGenerator):

    def __init__(self, means, sigma):
        super().__init__()
        self.means = means
        self.sigma = sigma

        if len(means) != len(sigma) or len(means) != len(sigma[0]):
            raise ValueError(f"Covariance matrix has shape {np.shape(sigma)} but expected ({len(means)}, {len(means)})")

        self.dist = stats.multivariate_normal(mean=means, cov=sigma)

    def generate(self, n):
        return self.dist.rvs(n).reshape(-1, len(self.means))

    def get_feature_space_size(self):
        return len(self.means)


class BinaryDataGenerator(_RandomDataGenerator):

    def __init__(self, n_features, p):
        super().__init__()
        if n_features == 1 and isinstance(p, float):
            p = [p]
        if len(p) != n_features:
            raise ValueError("Probability vector should be of size equal to number of features")
        self.n_features = n_features
        self.p = p
        self.dist = [stats.bernoulli(p[i]) for i in range(n_features)]

    def generate(self, n):
        x = np.zeros((n, self.n_features))
        for i in range(self.n_features):
            x[:, i] = self.dist[i].rvs(n)
        return x.reshape(-1, self.n_features)

    def get_feature_space_size(self):
        return self.n_features


class ConstantGenerator(_RandomDataGenerator):

    def __init__(self, c):
        super().__init__()
        self.c = c

    def generate(self, n):
        return np.array([self.c for _ in range(n)]).reshape(-1, 1)

    def get_feature_space_size(self):
        return 1


class DependentBinaryDataGenerator(_RandomDataGenerator):
    # To be implemented if needed
    pass


class CombinedDataGenerator(_RandomDataGenerator):

    def __init__(self, data_generators):
        super().__init__()
        self.data_generators = data_generators

    def generate(self, n):
        data = []
        for generator in self.data_generators:
            data.append(generator(n))
        return np.concatenate(data, axis=1)

    def get_feature_space_size(self):
        return sum([gen.get_feature_space_size() for gen in self.data_generators])


###############################################################################################
# Data Generating Processes
###############################################################################################

# Could just have a general dgp with a set of data generators and betas, and allow combinations of them
# Then, subclasses could just call the original class with the appropriate data generators and betas

class DGP:

    def __init__(self, data_generator, betas, error_generator=NormalErrorGenerator(), exponential=False):
        self.data_generator = data_generator
        self.betas = betas
        self.error_generator = error_generator
        self.exponential = exponential

    def generate(self, n):
        """
        Generates data from the DGP
        :param n: Number of data points
        :return: x: Regressor Matrix, y: Response Vector, u: Error Vector
        """
        x = self.data_generator(n)
        u = self.error_generator(x)
        y = x @ self.betas + u

        if self.exponential:
            y = np.exp(y)
        return y, x, u

# TODO: Add handling of datagenerator len being different from betas
class RCT(DGP):

    def __init__(self, treatment_effect, p_treated, data_generator=ConstantGenerator, betas=None,
                 error_generator=NormalErrorGenerator(), exponential=False):

        if data_generator is None:
            if betas is not None:
                raise ValueError('Betas provided without data generator')

        if data_generator is not None:
            if betas is None:
                betas = np.array([1])

        treatment_generator = BinaryDataGenerator(1, p_treated)

        if data_generator is None and betas is None:
            data_generator = treatment_generator
            betas = np.array([treatment_effect])

        else:
            data_generator = CombinedDataGenerator([treatment_generator, data_generator])
            betas = np.concatenate(([treatment_effect], betas))

        super().__init__(data_generator, betas, error_generator, exponential)
