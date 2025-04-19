import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

from loglinearcorrection.data_generating_processes import (
    NormalErrorGenerator, IndependentNormErrorGenerator, 
    IndependentLogNormErrorGenerator
)
from loglinearcorrection.data_generating_processes import constant_mean, constant_variance, constant_variance_ind

# Constants for testing
MAX_ARRAY_SIZE = 100

class TestErrorGenerators:
    """Tests for the error generator classes."""
    
    @pytest.mark.parametrize("error_class,mean_fn,var_fn", [
        (NormalErrorGenerator, constant_mean(0), constant_variance(1)),
        (IndependentNormErrorGenerator, constant_mean(0), constant_variance_ind(1)),
        (IndependentLogNormErrorGenerator, constant_mean(0), constant_variance(1)),
    ])
    def test_error_generator_instantiation(self, error_class, mean_fn, var_fn):
        """Test that error generators can be instantiated."""
        if error_class == NormalErrorGenerator:
            gen = error_class(mean_fn=mean_fn, cov_fn=var_fn)
        else:
            gen = error_class(mean_fn=mean_fn, var_fn=var_fn)
        
        assert gen is not None
    
    @given(
        x=arrays(dtype=np.float64, shape=st.integers(1, MAX_ARRAY_SIZE), 
                elements=st.floats(min_value=-100, max_value=100, width=32))
    )
    def test_normal_error_generator_shape(self, x):
        """Test that NormalErrorGenerator produces errors with the right shape."""
        gen = NormalErrorGenerator()
        errors = gen.generate(x)
        
        assert len(errors) == len(x)
    
    @given(
        x=arrays(dtype=np.float64, shape=st.integers(1, MAX_ARRAY_SIZE), 
                elements=st.floats(min_value=-100, max_value=100, width=32))
    )
    def test_independent_norm_error_generator_shape(self, x):
        """Test that IndependentNormErrorGenerator produces errors with the right shape."""
        gen = IndependentNormErrorGenerator()
        errors = gen.generate(x)
        
        assert len(errors) == len(x)
    
    @given(
        x=arrays(dtype=np.float64, shape=st.integers(1, MAX_ARRAY_SIZE), 
                elements=st.floats(min_value=-100, max_value=100, width=32))
    )
    def test_lognormal_error_generator_shape(self, x):
        """Test that IndependentLogNormErrorGenerator produces errors with the right shape."""
        gen = IndependentLogNormErrorGenerator()
        errors = gen.generate(x)
        
        assert len(errors) == len(x)
        # Log-normal values should be positive
        assert np.all(errors > 0)
    
    def test_error_generator_expected_value(self):
        """Test the expected_value method of error generators."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test with NormalErrorGenerator with mean 5
        gen = NormalErrorGenerator(mean_fn=constant_mean(5), cov_fn=constant_variance(1))
        expected = gen.expected_value(x, lambda x: x, trials=10000)
        
        # Expected value should be close to the mean (5)
        np.testing.assert_allclose(expected, np.full_like(x, 5), rtol=0.1)
        
        # Test with IndependentNormErrorGenerator with mean -3
        gen = IndependentNormErrorGenerator(mean_fn=constant_mean(-3), var_fn=constant_variance_ind(1))
        expected = gen.expected_value(x, lambda x: x, trials=10000)
        
        # Expected value should be close to the mean (-3)
        np.testing.assert_allclose(expected, np.full_like(x, -3), rtol=0.1)
    
    def test_error_generator_call_method(self):
        """Test that the __call__ method works correctly."""
        x = np.array([1.0, 2.0, 3.0])
        
        gen = NormalErrorGenerator()
        
        # __call__ should call generate
        errors_from_call = gen(x)
        errors_from_generate = gen.generate(x)
        
        assert len(errors_from_call) == len(x)
        assert len(errors_from_generate) == len(x)
        # They won't be equal because they're random draws
    
    def test_error_generator_invalid_inputs(self):
        """Test that error generators raise appropriate errors for invalid inputs."""
        x = np.array([1.0, 2.0, 3.0])
        
        # Define a mean function that returns wrong size
        def wrong_size_mean(x):
            return np.zeros(len(x) + 1)
        
        # Define a covariance function that returns wrong size
        def wrong_size_cov(x):
            n = len(x)
            return np.zeros((n + 1, n + 1))
        
        # Test NormalErrorGenerator with invalid mean function
        gen = NormalErrorGenerator(mean_fn=wrong_size_mean)
        with pytest.raises(ValueError):
            gen.generate(x)
        
        # Test NormalErrorGenerator with invalid covariance function
        gen = NormalErrorGenerator(cov_fn=wrong_size_cov)
        with pytest.raises(ValueError):
            gen.generate(x)
        
        # Test IndependentNormErrorGenerator with invalid mean function
        gen = IndependentNormErrorGenerator(mean_fn=wrong_size_mean)
        with pytest.raises(ValueError):
            gen.generate(x)
        
        # Define a variance function that returns wrong size for IndependentNormErrorGenerator
        def wrong_size_var_ind(x):
            return np.zeros(len(x) + 1)
        
        # Test IndependentNormErrorGenerator with invalid variance function
        gen = IndependentNormErrorGenerator(var_fn=wrong_size_var_ind)
        with pytest.raises(ValueError):
            gen.generate(x)
