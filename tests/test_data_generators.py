import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from loglinearcorrection.data_generating_processes import (
    NormalErrorGenerator, IndependentNormErrorGenerator, 
    IndependentLogNormErrorGenerator, MVNDataGenerator, 
    BinaryDataGenerator, ConstantGenerator, CombinedDataGenerator,
    DGP, RCT
)
from loglinearcorrection.data_generating_processes import constant_mean, constant_variance, constant_variance_ind

# Constants for testing
MAX_ARRAY_SIZE = 100
MAX_FEATURE_DIM = 10
MAX_SAMPLES = 500

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
    # Create a random matrix A
    A = draw(arrays(
        dtype=np.float64,
        shape=(dim, dim),
        elements=st.floats(min_value=-2.0, max_value=2.0, width=32)
    ))
    
    # Make it positive definite by multiplying with its transpose
    # and adding a small value to the diagonal to ensure it's positive definite
    cov = A @ A.T
    cov = cov + np.eye(dim) * 0.01
    
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

###############################################################################################
# Error Generator Tests
###############################################################################################

@pytest.mark.parametrize("mean_value,var_value", [
    (0, 1),
    (5, 2),
    (-3, 0.5)
])
def test_normal_error_generator_deterministic(mean_value, var_value):
    """Test NormalErrorGenerator with deterministic parameters."""
    x = np.array([1, 2, 3, 4, 5])
    gen = NormalErrorGenerator(
        mean_fn=constant_mean(mean_value),
        cov_fn=constant_variance(var_value)
    )
    
    errors = gen.generate(x)
    
    # Check shape
    assert len(errors) == len(x)
    
    # Test expected value with identity function
    # This should approximately equal the mean
    expected = gen.expected_value(x, lambda x: x, trials=10000)
    np.testing.assert_allclose(expected, np.full_like(x, mean_value), rtol=0.1)

@given(
    x=arrays(dtype=np.float64, shape=st.integers(1, MAX_ARRAY_SIZE), 
             elements=st.floats(min_value=-100, max_value=100, width=32)),
    mean_value=st.floats(min_value=-10, max_value=10),
    var_value=st.floats(min_value=0.01, max_value=5)
)
def test_normal_error_generator_property(x, mean_value, var_value):
    """Property-based test for NormalErrorGenerator."""
    gen = NormalErrorGenerator(
        mean_fn=constant_mean(mean_value),
        cov_fn=constant_variance(var_value)
    )
    
    errors = gen.generate(x)
    
    # Check shape
    assert len(errors) == len(x)
    
    # For large enough samples, the mean should be close to mean_value
    # and variance close to var_value, but we don't test that here
    # as it would require multiple samples

@given(
    x=arrays(dtype=np.float64, shape=st.integers(1, MAX_ARRAY_SIZE), 
             elements=st.floats(min_value=-100, max_value=100, width=32)),
    mean_value=st.floats(min_value=-10, max_value=10),
    var_value=st.floats(min_value=0.01, max_value=5)
)
def test_independent_norm_error_generator_property(x, mean_value, var_value):
    """Property-based test for IndependentNormErrorGenerator."""
    gen = IndependentNormErrorGenerator(
        mean_fn=constant_mean(mean_value),
        var_fn=constant_variance_ind(var_value)
    )
    
    errors = gen.generate(x)
    
    # Check shape
    assert len(errors) == len(x)
    
    # Since we're using constant functions, all means should be the same
    expected_mean = gen.expected_value(x, lambda x: x, trials=5000)
    np.testing.assert_allclose(expected_mean, np.full_like(x, mean_value), rtol=0.2)

@given(
    x=arrays(dtype=np.float64, shape=st.integers(1, MAX_ARRAY_SIZE), 
             elements=st.floats(min_value=-10, max_value=10, width=32)),
    mean_value=st.floats(min_value=-3, max_value=3),
    var_value=st.floats(min_value=0.01, max_value=1)
)
def test_lognormal_error_generator_property(x, mean_value, var_value):
    """Property-based test for IndependentLogNormErrorGenerator."""
    gen = IndependentLogNormErrorGenerator(
        mean_fn=constant_mean(mean_value),
        var_fn=constant_variance(var_value)
    )
    
    errors = gen.generate(x)
    
    # Check shape
    assert len(errors) == len(x)
    
    # Log-normal values should be positive
    assert np.all(errors > 0)
    
    # We can also check that the log of the errors follows a normal distribution
    # but that's more complex for a property test

###############################################################################################
# Random Data Generator Tests
###############################################################################################

@given(means_cov=mean_and_cov_matrices(), n=st.integers(1, MAX_SAMPLES))
def test_mvn_data_generator(means_cov, n):
    """Test MVNDataGenerator."""
    means, cov = means_cov
    gen = MVNDataGenerator(means, cov)
    
    data = gen.generate(n)
    
    # Check shape
    assert data.shape == (n, len(means))
    
    # Check feature space size
    assert gen.get_feature_space_size() == len(means)

@given(
    n_features=st.integers(1, MAX_FEATURE_DIM),
    p=probability_vectors(),
    n_samples=st.integers(1, MAX_SAMPLES)
)
def test_binary_data_generator(n_features, p, n_samples):
    """Test BinaryDataGenerator."""
    assume(len(p) == n_features)
    
    gen = BinaryDataGenerator(n_features, p)
    data = gen.generate(n_samples)
    
    # Check shape
    assert data.shape == (n_samples, n_features)
    
    # Check that values are binary (0 or 1)
    assert np.all(np.logical_or(data == 0, data == 1))
    
    # Check feature space size
    assert gen.get_feature_space_size() == n_features

@given(c=st.floats(min_value=-100, max_value=100), n=st.integers(1, MAX_SAMPLES))
def test_constant_generator(c, n):
    """Test ConstantGenerator."""
    gen = ConstantGenerator(c)
    data = gen.generate(n)
    
    # Check shape
    assert data.shape == (n, 1)
    
    # Check that all values are equal to c
    assert np.all(data == c)
    
    # Check feature space size
    assert gen.get_feature_space_size() == 1

@pytest.mark.parametrize("generators,expected_size", [
    ([ConstantGenerator(1), ConstantGenerator(2)], 2),
    ([ConstantGenerator(1), BinaryDataGenerator(2, [0.5, 0.5])], 3),
])
def test_combined_data_generator_deterministic(generators, expected_size):
    """Test CombinedDataGenerator with deterministic examples."""
    gen = CombinedDataGenerator(generators)
    data = gen.generate(10)
    
    # Check shape
    assert data.shape == (10, expected_size)
    
    # Check feature space size
    assert gen.get_feature_space_size() == expected_size

@given(n=st.integers(1, MAX_SAMPLES))
def test_combined_data_generator_property(n):
    """Property-based test for CombinedDataGenerator."""
    # Create a fixed set of generators for this test
    gen1 = ConstantGenerator(1)
    gen2 = BinaryDataGenerator(2, [0.3, 0.7])
    gen3 = MVNDataGenerator(np.array([0, 0]), np.array([[1, 0.5], [0.5, 1]]))
    
    combined_gen = CombinedDataGenerator([gen1, gen2, gen3])
    data = combined_gen.generate(n)
    
    expected_size = gen1.get_feature_space_size() + gen2.get_feature_space_size() + gen3.get_feature_space_size()
    
    # Check shape
    assert data.shape == (n, expected_size)
    
    # Check feature space size
    assert combined_gen.get_feature_space_size() == expected_size

###############################################################################################
# Data Generating Process Tests
###############################################################################################

@given(n=st.integers(1, MAX_SAMPLES))
def test_dgp_basic(n):
    """Test basic functionality of DGP."""
    data_gen = MVNDataGenerator(np.array([0, 0]), np.array([[1, 0.5], [0.5, l]]))
    betas = np.array([1.0, 2.0])
    error_gen = NormalErrorGenerator()
    
    dgp = DGP(data_gen, betas, error_gen)
    y, x, u = dgp.generate(n)
    
    # Check shapes
    assert y.shape == (n,)
    assert x.shape == (n, 2)
    assert u.shape == (n,)
    
    # Check that y = x @ betas + u
    np.testing.assert_allclose(y, x @ betas + u)

@given(
    n=st.integers(1, MAX_SAMPLES),
    treatment_effect=st.floats(min_value=-10, max_value=10),
    p_treated=st.floats(min_value=0.1, max_value=0.9)
)
def test_rct_basic(n, treatment_effect, p_treated):
    """Test basic functionality of RCT."""
    # Simple RCT with just treatment effect
    rct = RCT(treatment_effect, p_treated)
    y, x, u = rct.generate(n)
    
    # Check shapes
    assert y.shape == (n,)
    assert x.shape == (n, 1)
    assert u.shape == (n,)
    
    # Treatment indicator should be binary
    assert np.all(np.logical_or(x == 0, x == 1))
    
    # Check average treatment effect
    treated = y[x[:, 0] == 1]
    untreated = y[x[:, 0] == 0]
    
    # Skip if any group is empty (can happen with small n and extreme p_treated)
    if len(treated) > 0 and len(untreated) > 0:
        # In this simple model, the difference in means should be close to the treatment effect
        # (assuming normal errors with zero mean)
        observed_effect = np.mean(treated) - np.mean(untreated)
        # This could fail sometimes due to randomness, so we use a high tolerance
        np.testing.assert_allclose(observed_effect, treatment_effect, rtol=0.5, atol=1.0)

def test_rct_with_covariates():
    """Test RCT with additional covariates."""
    n = 1000
    treatment_effect = 2.0
    p_treated = 0.5
    
    # Create a data generator for covariates
    covariate_gen = ConstantGenerator(1)  # Just an intercept
    
    # Create an RCT with treatment effect and intercept
    rct = RCT(treatment_effect, p_treated, covariate_gen, np.array([3.0]))
    y, x, u = rct.generate(n)
    
    # Check shapes - we should have 2 columns: treatment and intercept
    assert y.shape == (n,)
    assert x.shape == (n, 2)
    assert u.shape == (n,)
    
    # First column should be treatment indicator (binary)
    assert np.all(np.logical_or(x[:, 0] == 0, x[:, 0] == 1))
    
    # Second column should be constant 1
    assert np.all(x[:, 1] == 1)
    
    # The model is: y = treatment_effect * T + 3.0 * 1 + u
    # So untreated units should have mean close to 3.0
    untreated = y[x[:, 0] == 0]
    np.testing.assert_allclose(np.mean(untreated), 3.0, rtol=0.2)
    
    # Treated units should have mean close to treatment_effect + 3.0
    treated = y[x[:, 0] == 1]
    np.testing.assert_allclose(np.mean(treated), treatment_effect + 3.0, rtol=0.2)

def test_rct_exponential():
    """Test RCT with exponential transformation."""
    n = 1000
    treatment_effect = 0.5  # Be careful with large effects when exponentiating
    p_treated = 0.5
    
    # Create an RCT with exponential transformation
    rct = RCT(treatment_effect, p_treated, exponential=True)
    y, x, u = rct.generate(n)
    
    # Check shapes
    assert y.shape == (n,)
    assert x.shape == (n, 1)
    assert u.shape == (n,)
    
    # Values should be positive due to exponential
    assert np.all(y > 0)
    
    # In log scale, we expect: log(y) = treatment_effect * T + u
    # So log(y) for untreated should have mean close to 0
    log_y = np.log(y)
    untreated_log = log_y[x[:, 0] == 0]
    np.testing.assert_allclose(np.mean(untreated_log), 0, atol=0.2)
    
    # And log(y) for treated should have mean close to treatment_effect
    treated_log = log_y[x[:, 0] == 1]
    np.testing.assert_allclose(np.mean(treated_log), treatment_effect, atol=0.2)
