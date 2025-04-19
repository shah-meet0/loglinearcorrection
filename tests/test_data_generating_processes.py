import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

# Import the classes to test
# Replace with your actual import path
# from your_package.data_generators import (
from loglinearcorrection.data_generating_processes import (
    NormalErrorGenerator, ConstantGenerator, BinaryDataGenerator,
    MVNDataGenerator, CombinedDataGenerator, DGP, RCT
)
from loglinearcorrection.data_generating_processes import constant_mean, constant_variance

# Constants for testing
MAX_SAMPLES = 1000
MAX_FEATURES = 5

class TestDGP:
    """Tests for the Data Generating Process (DGP) class."""
    
    def test_dgp_instantiation(self):
        """Test that DGP can be instantiated."""
        data_gen = ConstantGenerator(1.0)
        betas = np.array([2.0])
        error_gen = NormalErrorGenerator()
        
        dgp = DGP(data_gen, betas, error_gen)
        assert dgp is not None
    
    @given(
        n=st.integers(1, MAX_SAMPLES),
        constant_value=st.floats(min_value=-10, max_value=10),
        beta=st.floats(min_value=-10, max_value=10)
    )
    def test_dgp_with_constant_generator(self, n, constant_value, beta):
        """Test DGP with ConstantGenerator."""
        data_gen = ConstantGenerator(constant_value)
        betas = np.array([beta])
        error_gen = NormalErrorGenerator(mean_fn=constant_mean(0), cov_fn=constant_variance(1))
        
        dgp = DGP(data_gen, betas, error_gen)
        y, x, u = dgp.generate(n)
        
        # Check shapes
        assert y.shape == (n,)
        assert x.shape == (n, 1)
        assert u.shape == (n,)
        
        # Check that y = x @ betas + u
        np.testing.assert_allclose(y, x @ betas + u)
        
        # Check that x has the constant value
        assert np.all(x == constant_value)
    
    @given(
        n=st.integers(1, MAX_SAMPLES),
        p=st.floats(min_value=0.1, max_value=0.9),
        beta=st.floats(min_value=-10, max_value=10)
    )
    def test_dgp_with_binary_generator(self, n, p, beta):
        """Test DGP with BinaryDataGenerator."""
        data_gen = BinaryDataGenerator(1, p)
        betas = np.array([beta])
        error_gen = NormalErrorGenerator()
        
        dgp = DGP(data_gen, betas, error_gen)
        y, x, u = dgp.generate(n)
        
        # Check shapes
        assert y.shape == (n,)
        assert x.shape == (n, 1)
        assert u.shape == (n,)
        
        # Check that y = x @ betas + u
        np.testing.assert_allclose(y, x @ betas + u)
        
        # Check that x is binary (0 or 1)
        assert np.all(np.logical_or(x == 0, x == 1))
    
    def test_dgp_with_mvn_generator(self):
        """Test DGP with MVNDataGenerator."""
        means = np.array([0, 0])
        cov = np.array([[1, 0.5], [0.5, 1]])
        data_gen = MVNDataGenerator(means, cov)
        betas = np.array([1.0, 2.0])
        error_gen = NormalErrorGenerator()
        
        n = 100
        dgp = DGP(data_gen, betas, error_gen)
        y, x, u = dgp.generate(n)
        
        # Check shapes
        assert y.shape == (n,)
        assert x.shape == (n, 2)
        assert u.shape == (n,)
        
        # Check that y = x @ betas + u
        np.testing.assert_allclose(y, x @ betas + u)
    
    def test_dgp_with_combined_generator(self):
        """Test DGP with CombinedDataGenerator."""
        gen1 = ConstantGenerator(1.0)
        gen2 = BinaryDataGenerator(1, 0.5)
        combined_gen = CombinedDataGenerator([gen1, gen2])
        betas = np.array([1.0, 2.0])
        error_gen = NormalErrorGenerator()
        
        n = 100
        dgp = DGP(combined_gen, betas, error_gen)
        y, x, u = dgp.generate(n)
        
        # Check shapes
        assert y.shape == (n,)
        assert x.shape == (n, 2)
        assert u.shape == (n,)
        
        # Check that y = x @ betas + u
        np.testing.assert_allclose(y, x @ betas + u)
        
        # Check that first column is constant 1.0
        assert np.all(x[:, 0] == 1.0)
        
        # Check that second column is binary (0 or 1)
        assert np.all(np.logical_or(x[:, 1] == 0, x[:, 1] == 1))
    
    def test_dgp_exponential(self):
        """Test DGP with exponential transformation."""
        data_gen = ConstantGenerator(1.0)
        betas = np.array([2.0])
        error_gen = NormalErrorGenerator(mean_fn=constant_mean(0), cov_fn=constant_variance(0.1))
        
        n = 100
        dgp = DGP(data_gen, betas, error_gen, exponential=True)
        y, x, u = dgp.generate(n)
        
        # Check shapes
        assert y.shape == (n,)
        assert x.shape == (n, 1)
        assert u.shape == (n,)
        
        # Check that y = exp(x @ betas + u)
        np.testing.assert_allclose(y, np.exp(x @ betas + u))
        
        # Values should be positive due to exponential
        assert np.all(y > 0)
        
        # Check that log(y) = x @ betas + u
        np.testing.assert_allclose(np.log(y), x @ betas + u)


class TestRCT:
    """Tests for the Randomized Controlled Trial (RCT) class."""
    
    def test_rct_instantiation(self):
        """Test that RCT can be instantiated."""
        rct = RCT(treatment_effect=2.0, p_treated=0.5)
        assert rct is not None
    
    @given(
        n=st.integers(5, MAX_SAMPLES),
        treatment_effect=st.floats(min_value=-5, max_value=5),
        p_treated=st.floats(min_value=0.1, max_value=0.9)
    )
    @settings(deadline=None)  # Disable deadline as this test can be slow
    def test_rct_basic_properties(self, n, treatment_effect, p_treated):
        """Test basic properties of RCT."""
        rct = RCT(treatment_effect, p_treated)
        y, x, u = rct.generate(n)
        
        # Check shapes
        assert y.shape == (n,)
        assert x.shape == (n, 1)
        assert u.shape == (n,)
        
        # Treatment indicator should be binary
        assert np.all(np.logical_or(x == 0, x == 1))
        
        # The proportion of treated units should be approximately p_treated
        prop_treated = np.mean(x)
        # This might fail sometimes due to randomness with small n
        if n >= 50:  # Only check for reasonably large n
            np.testing.assert_allclose(prop_treated, p_treated, rtol=0.3)
    
    @given(
        n=st.integers(50, MAX_SAMPLES),
        treatment_effect=st.floats(min_value=-5, max_value=5),
        p_treated=st.floats(min_value=0.2, max_value=0.8)
    )
    @settings(deadline=None)  # Disable deadline as this test can be slow
    def test_rct_treatment_effect(self, n, treatment_effect, p_treated):
        """Test that RCT produces the correct treatment effect."""
        # Use an error generator with small variance to make the test more reliable
        error_gen = NormalErrorGenerator(mean_fn=constant_mean(0), cov_fn=constant_variance(0.1))
        
        rct = RCT(treatment_effect, p_treated, error_generator=error_gen)
        y, x, u = rct.generate(n)
        
        # Extract treated and untreated units
        treated_idx = (x[:, 0] == 1)
        untreated_idx = (x[:, 0] == 0)
        
        # Ensure we have both treated and untreated units
        assume(np.any(treated_idx) and np.any(untreated_idx))
        
        treated_y = y[treated_idx]
        untreated_y = y[untreated_idx]
        
        # The difference in means should be approximately the treatment effect
        effect = np.mean(treated_y) - np.mean(untreated_y)
        np.testing.assert_allclose(effect, treatment_effect, rtol=0.5, atol=0.5)
    
    def test_rct_with_covariates(self):
        """Test RCT with additional covariates."""
        n = 500
        treatment_effect = 2.0
        p_treated = 0.5
        
        # Create a constant covariate (intercept)
        covariate_gen = ConstantGenerator(1.0)
        beta = np.array([3.0])  # Coefficient for the intercept
        
        # Use a small variance for more reliable tests
        error_gen = NormalErrorGenerator(mean_fn=constant_mean(0), cov_fn=constant_variance(0.1))
        
        rct = RCT(treatment_effect, p_treated, covariate_gen, beta, error_gen)
        y, x, u = rct.generate(n)
        
        # Check shapes - we should have 2 columns: treatment and intercept
        assert y.shape == (n,)
        assert x.shape == (n, 2)
        assert u.shape == (n,)
        
        # First column should be treatment indicator (binary)
        assert np.all(np.logical_or(x[:, 0] == 0, x[:, 0] == 1))
        
        # Second column should be constant 1.0
        assert np.all(x[:, 1] == 1.0)
        
        # Check that y = [treatment_effect, beta] @ [T, 1] + u
        expected_y = x @ np.concatenate(([treatment_effect], beta)) + u
        np.testing.assert_allclose(y, expected_y)
        
        # Extract treated and untreated units
        treated_idx = (x[:, 0] == 1)
        untreated_idx = (x[:, 0] == 0)
        
        treated_y = y[treated_idx]
        untreated_y = y[untreated_idx]
        
        # The difference in means should be approximately the treatment effect
        effect = np.mean(treated_y) - np.mean(untreated_y)
        np.testing.assert_allclose(effect, treatment_effect, rtol=0.3)
    
    def test_rct_no_data_generator(self):
        """Test RCT without specifying a data generator."""
        n = 500
        treatment_effect = 2.0
        p_treated = 0.5
        
        # Create an RCT with only treatment effect
        rct = RCT(treatment_effect, p_treated)
        y, x, u = rct.generate(n)
        
        # Check shapes
        assert y.shape == (n,)
        assert x.shape == (n, 1)
        assert u.shape == (n,)
        
        # Check that x is only the treatment indicator
        assert np.all(np.logical_or(x == 0, x == 1))
        
        # Check that y = treatment_effect * T + u
        expected_y = x @ np.array([treatment_effect]) + u
        np.testing.assert_allclose(y, expected_y)
    
    def test_rct_exponential(self):
        """Test RCT with exponential transformation."""
        n = 500
        treatment_effect = 0.5  # Use small effect for exponential
        p_treated = 0.5
        
        # Use small variance for more reliable tests
        error_gen = NormalErrorGenerator(mean_fn=constant_mean(0), cov_fn=constant_variance(0.1))
        
        rct = RCT(treatment_effect, p_treated, error_generator=error_gen, exponential=True)
        y, x, u = rct.generate(n)
        
        # Check shapes
        assert y.shape == (n,)
        assert x.shape == (n, 1)
        assert u.shape == (n,)
        
        # Values should be positive due to exponential
        assert np.all(y > 0)
        
        # Check that y = exp(treatment_effect * T + u)
        expected_y = np.exp(x @ np.array([treatment_effect]) + u)
        np.testing.assert_allclose(y, expected_y)
        
        # In log scale, the treatment effect should be more visible
        log_y = np.log(y)
        
        # Extract treated and untreated units
        treated_idx = (x[:, 0] == 1)
        untreated_idx = (x[:, 0] == 0)
        
        treated_log_y = log_y[treated_idx]
        untreated_log_y = log_y[untreated_idx]
        
        # The difference in means of log(y) should be approximately the treatment effect
        log_effect = np.mean(treated_log_y) - np.mean(untreated_log_y)
        np.testing.assert_allclose(log_effect, treatment_effect, rtol=0.3)
    
    @pytest.mark.parametrize("invalid_args", [
        # Test providing betas without data generator
        {"treatment_effect": 2.0, "p_treated": 0.5, "data_generator": None, "betas": np.array([1.0])},
    ])
    def test_rct_invalid_inputs(self, invalid_args):
        """Test that RCT raises appropriate errors for invalid inputs."""
        with pytest.raises(ValueError):
            RCT(**invalid_args)
