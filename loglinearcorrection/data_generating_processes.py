###############################################################################################
# This module generates various simulated data for the purpose of testing.
# Each function generates a different type of data.
###############################################################################################

import numpy as np
import scipy.stats as stats
from .dependence_funcs import constant_mean, constant_variance, constant_variance_ind

###############################################################################################
# Error Generators
###############################################################################################

# TODO: Implement expectation tracking.
class _ErrorGenerator:
    """Base class for error generators.
    
    This abstract class defines the interface for all error generators. Error generators
    create random noise/errors that are added to simulated data.
    """
    def __init__(self):
        """Initialize the error generator."""
        pass

    def generate(self, x):
        """Generate errors based on input data.
        
        :param x: Input data for which to generate errors
        :type x: numpy.ndarray
        :return: Generated errors
        :rtype: numpy.ndarray
        """
        pass

    def __call__(self, x):
        """Make the error generator callable.
        
        :param x: Input data for which to generate errors
        :type x: numpy.ndarray
        :return: Generated errors by calling the generate method
        :rtype: numpy.ndarray
        """
        return self.generate(x)

    def expected_value(self, x, func, trials=10000):
        """Calculate the expected value of a function of the generated errors.
        
        :param x: Input data
        :type x: numpy.ndarray
        :param func: Function to apply to generated errors
        :type func: callable
        :param trials: Number of Monte Carlo trials, defaults to 10000
        :type trials: int, optional
        :return: Expected value of the function applied to the errors
        :rtype: numpy.ndarray
        """
        draws = np.array([func(self.generate(x)) for _ in range(trials)])
        return np.mean(draws, axis=0)


class NormalErrorGenerator(_ErrorGenerator):
    """Generates multivariate normal errors with configurable mean and covariance.
    
    This generator creates correlated normal errors based on provided mean and
    covariance functions that can depend on the input data.
    """

    def __init__(self, mean_fn=constant_mean(0), cov_fn=constant_variance(1)):
        """Initialize the normal error generator.
        
        :param mean_fn: Function that converts input data into a mean vector, defaults to constant_mean(0)
        :type mean_fn: callable, optional
        :param cov_fn: Function that converts input data into a covariance matrix, defaults to constant_variance(1)
        :type cov_fn: callable, optional
        """
        super().__init__()

        self.mean_fn = mean_fn
        self.cov_fn = cov_fn

    def generate(self, x):
        """Generate multivariate normal errors based on input data.
        
        :param x: Input data of shape (n, d) where n is number of samples
        :type x: numpy.ndarray
        :return: Multivariate normal errors of shape (n,)
        :rtype: numpy.ndarray
        :raises ValueError: If mean function returns vector of wrong length or
            covariance matrix has wrong shape
        """
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
    """Generates independent normal errors with configurable mean and variance.
    
    This generator creates uncorrelated normal errors where each error term has its
    own mean and variance that can depend on the input data.
    """

    def __init__(self, mean_fn=constant_mean(0), var_fn=constant_variance_ind(1)):
        """Initialize the independent normal error generator.
        
        :param mean_fn: Function that converts input data into a mean vector, defaults to constant_mean(0)
        :type mean_fn: callable, optional
        :param var_fn: Function that converts input data into a variance vector, defaults to constant_variance_ind(1)
        :type var_fn: callable, optional
        """
        super().__init__()
        self.mean_fn = mean_fn
        self.var_fn = var_fn

    def generate(self, x):
        """Generate independent normal errors based on input data.
        
        :param x: Input data of shape (n, d) where n is number of samples
        :type x: numpy.ndarray
        :return: Independent normal errors of shape (n,)
        :rtype: numpy.ndarray
        :raises ValueError: If mean function returns vector of wrong length or
            variance vector has wrong shape
        """
        n = len(x)
        means = self.mean_fn(x)
        var = self.var_fn(x)

        if len(means) != n:
            raise ValueError(f"Mean function returns vector of length {len(means)} but expected {n}")

        if len(var) != n:
            raise ValueError(f"Variance vector has shape {np.shape(var)} but expected ({n})")

        return means + np.sqrt(var) * np.random.randn(n)

class IndependentLogNormErrorGenerator(_ErrorGenerator):
    """Generates independent log-normal errors with configurable mean and variance.
    
    This generator creates uncorrelated log-normal errors where each error term is
    derived from a normal distribution and then exponentiated.
    """

    def __init__(self, mean_fn=constant_mean(0), var_fn=constant_variance(1)):
        """Initialize the independent log-normal error generator.
        
        :param mean_fn: Function that converts input data into a mean vector for the underlying normal distribution, defaults to constant_mean(0)
        :type mean_fn: callable, optional
        :param var_fn: Function that converts input data into a covariance matrix for the underlying normal distribution, defaults to constant_variance(1)
        :type var_fn: callable, optional
        """
        super().__init__()
        self.mean_fn = mean_fn
        self.var_fn = var_fn

    def generate(self, x):
        """Generate independent log-normal errors based on input data.
        
        :param x: Input data of shape (n, d) where n is number of samples
        :type x: numpy.ndarray
        :return: Independent log-normal errors of shape (n,)
        :rtype: numpy.ndarray
        :raises ValueError: If mean function returns vector of wrong length or
            variance matrix has wrong shape
        """
        n = len(x)
        means = self.mean_fn(x)
        var = self.var_fn(x)

        if len(means) != n:
            raise ValueError(f"Mean function returns vector of length {len(means)} but expected {n}")

        if len(var) != n:
            raise ValueError(f"Variance matrix has shape {np.shape(var)} but expected ({n})")

        return np.exp(means + np.sqrt(var) * np.random.randn(n))


###############################################################################################
# Random Data Generators
###############################################################################################

class _RandomDataGenerator:
    """Base class for random data generators.
    
    This abstract class defines the interface for all data generators, which
    create feature matrices for simulated data.
    """

    def __init__(self):
        """Initialize the random data generator.
        
        The dist attribute will hold the statistical distribution to be used (initialized as None).
        """
        self.dist = None

    def generate(self, n):
        """Generate random data.
        
        :param n: Number of samples to generate
        :type n: int
        :return: Generated data
        :rtype: numpy.ndarray
        """
        pass

    def __call__(self, n):
        """Make the data generator callable.
        
        :param n: Number of samples to generate
        :type n: int
        :return: Generated data by calling the generate method
        :rtype: numpy.ndarray
        """
        return self.generate(n)


class MVNDataGenerator(_RandomDataGenerator):
    """Generates multivariate normal data with specified mean and covariance.
    
    This generator creates correlated features following a multivariate normal distribution.
    """

    def __init__(self, means, sigma):
        """Initialize the multivariate normal data generator.
        
        :param means: Mean vector for the multivariate normal distribution
        :type means: numpy.ndarray
        :param sigma: Covariance matrix for the multivariate normal distribution
        :type sigma: numpy.ndarray
        :raises ValueError: If the dimensions of means and sigma don't match
        """
        super().__init__()
        self.means = means
        self.sigma = sigma

        if len(means) != len(sigma) or len(means) != len(sigma[0]):
            raise ValueError(f"Covariance matrix has shape {np.shape(sigma)} but expected ({len(means)}, {len(means)})")

        self.dist = stats.multivariate_normal(mean=means, cov=sigma)

    def generate(self, n):
        """Generate multivariate normal data.
        
        :param n: Number of samples to generate
        :type n: int
        :return: Generated data of shape (n, d) where d is the number of features
        :rtype: numpy.ndarray
        """
        return self.dist.rvs(n).reshape(-1, len(self.means))

    def get_feature_space_size(self):
        """Get the number of features in the generated data.
        
        :return: Number of features
        :rtype: int
        """
        return len(self.means)


class BinaryDataGenerator(_RandomDataGenerator):
    """Generates binary (0/1) data with specified probabilities.
    
    This generator creates binary features where each feature has a specified
    probability of being 1.
    """

    def __init__(self, n_features, p):
        """Initialize the binary data generator.
        
        :param n_features: Number of binary features to generate
        :type n_features: int
        :param p: Probability/probabilities of each feature being 1. If float, the same probability is used for all features. If list, must have length equal to n_features
        :type p: float or list
        :raises ValueError: If length of p doesn't match n_features
        """
        super().__init__()
        if n_features == 1 and isinstance(p, float):
            p = [p]
        if len(p) != n_features:
            raise ValueError("Probability vector should be of size equal to number of features")
        self.n_features = n_features
        self.p = p
        self.dist = [stats.bernoulli(p[i]) for i in range(n_features)]

    def generate(self, n):
        """Generate binary data.
        
        :param n: Number of samples to generate
        :type n: int
        :return: Generated binary data of shape (n, n_features)
        :rtype: numpy.ndarray
        """
        x = np.zeros((n, self.n_features))
        for i in range(self.n_features):
            x[:, i] = self.dist[i].rvs(n)
        return x.reshape(-1, self.n_features)

    def get_feature_space_size(self):
        """Get the number of features in the generated data.
        
        :return: Number of features
        :rtype: int
        """
        return self.n_features


class ConstantGenerator(_RandomDataGenerator):
    """Generates constant values.
    
    This generator creates a single feature column with the same constant value
    for all samples.
    """

    def __init__(self, c):
        """Initialize the constant generator.
        
        :param c: Constant value to generate
        :type c: float
        """
        super().__init__()
        self.c = c

    def generate(self, n):
        """Generate constant data.
        
        :param n: Number of samples to generate
        :type n: int
        :return: Generated constant data of shape (n, 1)
        :rtype: numpy.ndarray
        """
        return np.array([self.c for _ in range(n)]).reshape(-1, 1)

    def get_feature_space_size(self):
        """Get the number of features in the generated data.
        
        :return: Always 1 for constant generator
        :rtype: int
        """
        return 1


class DependentBinaryDataGenerator(_RandomDataGenerator):
    """Placeholder for a generator of binary data with dependencies between features.
    
    To be implemented if needed.
    """
    pass


class CombinedDataGenerator(_RandomDataGenerator):
    """Combines multiple data generators into one.
    
    This generator concatenates the outputs of multiple data generators to create
    a unified feature matrix.
    """

    def __init__(self, data_generators):
        """Initialize the combined data generator.
        
        :param data_generators: List of data generator instances to combine
        :type data_generators: list
        """
        super().__init__()
        self.data_generators = data_generators

    def generate(self, n):
        """Generate combined data from all generators.
        
        :param n: Number of samples to generate
        :type n: int
        :return: Combined generated data of shape (n, total_features)
        :rtype: numpy.ndarray
        """
        data = []
        for generator in self.data_generators:
            data.append(generator(n))
        return np.concatenate(data, axis=1)

    def get_feature_space_size(self):
        """Get the total number of features across all generators.
        
        :return: Sum of feature dimensions from all generators
        :rtype: int
        """
        return sum([gen.get_feature_space_size() for gen in self.data_generators])


###############################################################################################
# Data Generating Processes
###############################################################################################

class DGP:
    """Base Data Generating Process class.
    
    This class combines data generators with coefficients (betas) and error generators
    to create simulated response variables that depend on features.
    """

    def __init__(self, data_generator, betas, error_generator=NormalErrorGenerator(), exponential=False):
        """Initialize the Data Generating Process.
        
        :param data_generator: Generator that produces feature matrix X
        :type data_generator: callable
        :param betas: Coefficient vector for the linear model
        :type betas: numpy.ndarray
        :param error_generator: Generator for the error terms, defaults to NormalErrorGenerator()
        :type error_generator: _ErrorGenerator, optional
        :param exponential: Whether to exponentiate the final response, defaults to False
        :type exponential: bool, optional
        """
        self.data_generator = data_generator
        self.betas = betas
        self.error_generator = error_generator
        self.exponential = exponential

    def generate(self, n):
        """Generate data from the Data Generating Process.
        
        :param n: Number of data points to generate
        :type n: int
        :return: Tuple containing (y, x, u) where y is the response vector,
                 x is the regressor matrix, and u is the error vector
        :rtype: tuple
        """
        x = self.data_generator(n)
        u = self.error_generator(x)
        y = x @ self.betas + u

        if self.exponential:
            y = np.exp(y)
        return y, x, u

class RCT(DGP):
    """Randomized Controlled Trial data generating process.
    
    This class simulates data from a randomized controlled trial where some units
    receive a treatment with a specified effect.
    """

    def __init__(self, treatment_effect, p_treated, data_generator=ConstantGenerator(1), betas=None,
                 error_generator=NormalErrorGenerator(), exponential=False):
        """Initialize the RCT data generating process.
        
        :param treatment_effect: Effect of the treatment on the outcome
        :type treatment_effect: float
        :param p_treated: Probability of a unit being treated
        :type p_treated: float
        :param data_generator: Generator for covariates, defaults to ConstantGenerator
        :type data_generator: callable, optional
        :param betas: Coefficients for covariates, required if data_generator is provided
        :type betas: numpy.ndarray, optional
        :param error_generator: Generator for error terms, defaults to NormalErrorGenerator()
        :type error_generator: _ErrorGenerator, optional
        :param exponential: Whether to exponentiate the final response, defaults to False
        :type exponential: bool, optional
        :raises ValueError: If betas are provided without a data generator or vice versa
        """

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
