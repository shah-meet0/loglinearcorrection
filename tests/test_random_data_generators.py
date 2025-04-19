import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

# Import the generators to test
# Replace with your actual import path
# from your_package.data_generators import (
from loglinearcorrection.data_generating_processes import (
    MVNDataGenerator, BinaryDataGenerator, ConstantGenerator, 
    CombinedDataGenerator
)

# Constants for testing
MAX_SAMPLES = 1000
MAX_FEATURE_DIM = 10

# Helper strategies
@st.composite
def mean_and_cov_matrices(draw, max_dim=MAX_FEATURE_DIM):
    """Generate a mean vector and valid covariance matrix pair."""
    dim = draw(st.integers(min_value=1, max_value=max_dim))
    
    # Generate mean vector
    means = draw(arrays(
        dtype=np.float64,
        shape=dim,
        elements=st.floats(min_value=-10.0, max_value=10.0, width=32)
    ))
    
    # Generate a positive definite covariance matrix
    A = draw(arrays(
        dtype=np.float64,
        shape=(dim, dim),
        elements=st.floats(min_value=-2.0, max_value=2.0, width=32)
    ))
    
    # Make it positive definite
    cov = A @ A.T
    cov = cov + np.eye(dim) * 0.01  # Ensure positive definiteness
    
    return means, cov

@st.composite
def probability_vectors(draw, max_dim=MAX_FEATURE_DIM):
    """Generate a vector of probabilities."""
    dim = draw(st.integers(min_value=1, max_value=max_dim))
    
    return draw(arrays(
        dtype=np.float64,
        shape=dim,
        elements=st.floats(min_value=0.0, max_value=1.0, width=32)
    ))

class TestMVNDataGenerator:
    """Tests for the Multivariate Normal Data Generator."""
    
    def test_mvn_instantiation(self):
        """Test that MVNDataGenerator can be instantiated."""
        means = np.array([0, 0])
        cov = np.array([[1, 0.5], [0.5, 1]])
        
        gen = MVNDataGenerator(means, cov)
        assert gen is not None
        assert gen.get_feature_space_size() == 2
    
    def test_mvn_instantiation_invalid(self):
        """Test that MVNDataGenerator raises error for invalid inputs."""
        means = np.array([0, 0])
        cov = np.array([[1, 0.5]])  # Invalid shape
        
        with pytest.raises(ValueError):
            MVNDataGenerator(means, cov)
    
    @given(means_cov=mean_and_cov_matrices(), n=st.integers(1, MAX_SAMPLES))
    def test_mvn_generator_dimensions(self, means_cov, n):
        """Test dimensions of generated data."""
        means, cov = means_cov
        gen = MVNDataGenerator(means, cov)
        
        data = gen.generate(n)
        
        # Check shape
        assert data.shape == (n, len(means))
        
        # Check get_feature_space_size
        assert gen.get_feature_space_size() == len(means)
    
    def test_mvn_generator_distribution(self):
        """Test that MVNDataGenerator produces data with expected distribution properties."""
        means = np.array([1.0, -2.0])
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        
        gen = MVNDataGenerator(means, cov)
        n = 10000  # Large sample for stable estimates
        
        data = gen.generate(n)
        
        # Sample mean should be close to the true mean
        sample_mean = np.mean(data, axis=0)
        np.testing.assert_allclose(sample_mean, means, rtol=0.1)
        
        # Sample covariance should be close to the true covariance
        sample_cov = np.cov(data, rowvar=False)
        np.testing.assert_allclose(sample_cov, cov, rtol=0.2)

class TestBinaryDataGenerator:
    """Tests for the Binary Data Generator."""
    
    def test_binary_instantiation(self):
        """Test that BinaryDataGenerator can be instantiated."""
        # Single feature with probability 0.5
        gen = BinaryDataGenerator(1, 0.5)
        assert gen is not None
        assert gen.get_feature_space_size() == 1
        
        # Multiple features with different probabilities
        gen = BinaryDataGenerator(3, [0.3, 0.5, 0.7])
        assert gen is not None
        assert gen.get_feature_space_size() == 3
    
    def test_binary_instantiation_invalid(self):
        """Test that BinaryDataGenerator raises error for invalid inputs."""
        # Wrong length of probability vector
        with pytest.raises(ValueError):
            BinaryDataGenerator(3, [0.3, 0.5])
    
    @given(
        n_features=st.integers(1, MAX_FEATURE_DIM),
        p=probability_vectors(),
        n_samples=st.integers(1, MAX_SAMPLES)
    )
    def test_binary_generator_dimensions(self, n_features, p, n_samples):
        """Test dimensions of generated data."""
        assume(len(p) == n_features)
        
        gen = BinaryDataGenerator(n_features, p)
        data = gen.generate(n_samples)
        
        # Check shape
        assert data.shape == (n_samples, n_features)
        
        # Check get_feature_space_size
        assert gen.get_feature_space_size() == n_features
    
    @given(
        n_features=st.integers(1, MAX_FEATURE_DIM),
        p=probability_vectors(),
        n_samples=st.integers(100, MAX_SAMPLES)  # Larger samples for stable estimates
    )
    def test_binary_generator_distribution(self, n_features, p, n_samples):
        """Test that BinaryDataGenerator produces data with expected distribution properties."""
        assume(len(p) == n_features)
        
        gen = BinaryDataGenerator(n_features, p)
        data = gen.generate(n_samples)
        
        # Check that values are binary (0 or 1)
        assert np.all(np.logical_or(data == 0, data == 1))
        
        # Check that mean of each column is close to the corresponding probability
        sample_means = np.mean(data, axis=0)
        
        # Use higher tolerance for smaller sample sizes
        tol = max(0.2, 5.0 / np.sqrt(n_samples))
        np.testing.assert_allclose(sample_means, p, rtol=tol, atol=tol)

class TestConstantGenerator:
    """Tests for the Constant Generator."""
    
    def test_constant_instantiation(self):
        """Test that ConstantGenerator can be instantiated."""
        gen = ConstantGenerator(1.0)
        assert gen is not None
        assert gen.get_feature_space_size() == 1
    
    @given(c=st.floats(min_value=-100, max_value=100), n=st.integers(1, MAX_SAMPLES))
    def test_constant_generator_dimensions(self, c, n):
        """Test dimensions of generated data."""
        gen = ConstantGenerator(c)
        data = gen.generate(n)
        
        # Check shape
        assert data.shape == (n, 1)
        
        # Check get_feature_space_size
        assert gen.get_feature_space_size() == 1
    
    @given(c=st.floats(min_value=-100, max_value=100), n=st.integers(1, MAX_SAMPLES))
    def test_constant_generator_values(self, c, n):
        """Test that ConstantGenerator produces data with the expected constant value."""
        gen = ConstantGenerator(c)
        data = gen.generate(n)
        
        # All values should be equal to c
        assert np.all(data == c)

class TestCombinedDataGenerator:
    """Tests for the Combined Data Generator."""
    
    def test_combined_instantiation(self):
        """Test that CombinedDataGenerator can be instantiated."""
        gen1 = ConstantGenerator(1.0)
        gen2 = BinaryDataGenerator(2, [0.3, 0.7])
        
        combined_gen = CombinedDataGenerator([gen1, gen2])
        assert combined_gen is not None
        assert combined_gen.get_feature_space_size() == 3  # 1 + 2
    
    @pytest.mark.parametrize("generators,expected_size", [
        ([ConstantGenerator(1), ConstantGenerator(2)], 2),
        ([ConstantGenerator(1), BinaryDataGenerator(2, [0.5, 0.5])], 3),
        ([ConstantGenerator(1), BinaryDataGenerator(2, [0.3, 0.7]), 
          MVNDataGenerator(np.array([0, 0]), np.array([[1, 0.5], [0.5, 1]]))], 5)
    ])
    def test_combined_generator_size(self, generators, expected_size):
        """Test that CombinedDataGenerator has the expected feature space size."""
        combined_gen = CombinedDataGenerator(generators)
        assert combined_gen.get_feature_space_size() == expected_size
    
    @given(n=st.integers(1, MAX_SAMPLES))
    def test_combined_generator_dimensions(self, n):
        """Test dimensions of generated data."""
        gen1 = ConstantGenerator(1.0)
        gen2 = BinaryDataGenerator(2, [0.3, 0.7])
        gen3 = MVNDataGenerator(np.array([0, 0]), np.array([[1, 0.5], [0.5, 1]]))
        
        combined_gen = CombinedDataGenerator([gen1, gen2, gen3])
        expected_size = gen1.get_feature_space_size() + gen2.get_feature_space_size() + gen3.get_feature_space_size()
        
        data = combined_gen.generate(n)
        
        # Check shape
        assert data.shape == (n, expected_size)
    
    def test_combined_generator_values(self):
        """Test that CombinedDataGenerator correctly combines the values from each generator."""
        c = 5.0
        gen1 = ConstantGenerator(c)
        p = [0.3, 0.7]
        gen2 = BinaryDataGenerator(2, p)
        
        combined_gen = CombinedDataGenerator([gen1, gen2])
        n = 100
        data = combined_gen.generate(n)
        
        # First column should be constant c
        assert np.all(data[:, 0] == c)
        
        # Second and third columns should be binary (0 or 1)
        assert np.all(np.logical_or(data[:, 1:3] == 0, data[:, 1:3] == 1))
        
        # Check that mean of binary columns is close to the corresponding probability
        # (for a large enough sample)
        if n >= 50:
            sample_means = np.mean(data[:, 1:3], axis=0)
            np.testing.assert_allclose(sample_means, p, rtol=0.5)
