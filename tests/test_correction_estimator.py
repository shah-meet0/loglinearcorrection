import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator

# Mocking AssumptionTest for testing
class MockAssumptionTest:
    def __init__(self, model):
        self.model = model
    
    def test_direct(self):
        return {'test_statistic': 1.0, 'p_value': 0.5, 'conclusion': 'Test passed'}

# Replace actual import with mock
import sys
import types
mock_module = types.ModuleType('scripts.ppml_consistency')
mock_module.AssumptionTest = MockAssumptionTest
sys.modules['scripts.ppml_consistency'] = mock_module

# Strategies for data generation
@st.composite
def regression_data(draw, n_samples=100, n_features=3, min_value=0.1, max_value=10.0):
    """Generate random regression data."""
    X = draw(arrays(
        dtype=np.float64,
        shape=(n_samples, n_features),
        elements=st.floats(min_value=min_value, max_value=max_value)
    ))
    
    # Generate positive y values
    beta = draw(arrays(dtype=np.float64, shape=n_features, 
                      elements=st.floats(min_value=-2.0, max_value=2.0)))
    
    # Generate y with positive values to allow log transformation
    noise = draw(arrays(dtype=np.float64, shape=n_samples,
                       elements=st.floats(min_value=0.5, max_value=1.5)))
    
    y = np.exp(X @ beta) * noise
    
    # Ensure all y values are positive
    y = np.maximum(y, 0.1)
    
    return X, y, beta

class TestDoublyRobustElasticityEstimator:
    """Test class for DoublyRobustElasticityEstimator - pytest compatible."""
    
    def test_compute_corrections_binary(self):
        """Test binary regressor corrections using pytest class-based approach."""
        # Generate simple data with a binary regressor
        np.random.seed(0)
        n = 200
        x = np.random.binomial(1, 0.5, size=n)
        beta_0 = 0.7
        beta_1 = 0.9
        # u_i with some heteroskedasticity for realism
        u = np.random.normal(0, 0.2 + 0.1 * x, n)
        log_y = beta_0 + beta_1 * x + u
        y = np.exp(log_y)
        df = pd.DataFrame({'y': y, 'x': x})
        
        # Fit your estimator (use 'binary' for nonparametric correction)
        dre = DoublyRobustElasticityEstimator(
            endog=df['y'],
            exog=df[['x']],
            interest='x',
            estimator_type='binary',
            elasticity=False
        )
        results = dre.fit()
        
        # Pull out the values
        beta_hat = results.beta_hat[0]  # Only one regressor
        m0 = results.nonparam_model.predict(np.array([[0]]))[0]
        m1 = results.nonparam_model.predict(np.array([[1]]))[0]
        
        # Expected semi-elasticity for binary: e^{beta_hat} * m1/m0 - 1
        expected = np.exp(beta_hat) * (m1 / m0) - 1
        
        # The estimator_values for a binary variable should be constant at either value
        est_x0 = results.estimator_values[0][x == 0]
        est_x1 = results.estimator_values[0][x == 1]
        
        # Pytest assertions with helpful error messages
        assert np.allclose(est_x0, expected, atol=1e-2), (
            f"est_x0 values not close to expected:\n"
            f"  Mean est_x0: {np.mean(est_x0):.5f}\n"
            f"  Expected: {expected:.5f}\n"
            f"  Max difference: {np.max(np.abs(est_x0 - expected)):.5f}"
        )
        
        assert np.allclose(est_x1, expected, atol=1e-2), (
            f"est_x1 values not close to expected:\n"
            f"  Mean est_x1: {np.mean(est_x1):.5f}\n"
            f"  Expected: {expected:.5f}\n"
            f"  Max difference: {np.max(np.abs(est_x1 - expected)):.5f}"
        )
        
        print(f"Estimated semi-elasticity (binary): {est_x0[0]:.5f}")
        print(f"Expected value from formula: {expected:.5f}")
    
    def test_binary_corrections_consistency(self):
        """Additional test to verify consistency of binary corrections."""
        np.random.seed(42)
        n = 150
        x = np.random.binomial(1, 0.6, size=n)
        u = np.random.normal(0, 0.3, n)
        log_y = 1.2 + 0.8 * x + u
        y = np.exp(log_y)
        df = pd.DataFrame({'y': y, 'x': x})
        
        dre = DoublyRobustElasticityEstimator(
            endog=df['y'],
            exog=df[['x']],
            interest='x',
            estimator_type='binary',
            elasticity=False
        )
        results = dre.fit()
        
        # All estimator values for x=0 should be the same
        est_x0 = results.estimator_values[0][x == 0]
        est_x1 = results.estimator_values[0][x == 1]
        
        # Check consistency within groups
        assert np.allclose(est_x0, est_x0[0], atol=1e-10), (
            "All estimator values for x=0 should be identical"
        )
        assert np.allclose(est_x1, est_x1[0], atol=1e-10), (
            "All estimator values for x=1 should be identical"
        )
        
        # Check that both groups have the same correction value
        assert np.allclose(est_x0[0], est_x1[0], atol=1e-10), (
            f"Binary correction should be the same for both groups: "
            f"{est_x0[0]:.10f} vs {est_x1[0]:.10f}"
        )

    def test_bootstrap_only_supports_pairs_method(self):
        """Test that only 'pairs' bootstrap method is supported, others raise ValueError."""
        # Generate simple test data
        np.random.seed(42)
        n = 50
        x = np.random.normal(0, 1, n)
        y = np.exp(0.5 + 0.8 * x + np.random.normal(0, 0.2, n))
        df = pd.DataFrame({'y': y, 'x': x})
        
        # Create estimator
        dre = DoublyRobustElasticityEstimator(
            endog=df['y'],
            exog=df[['x']],
            interest='x',
            estimator_type='ols',
            elasticity=False
        )
        
        # Test that 'residuals' bootstrap method raises ValueError
        with pytest.raises(ValueError, match="Bootstrap method 'residuals' not supported"):
            dre.fit(bootstrap=True, bootstrap_reps=5, bootstrap_method='residuals')
        
        # Test that 'pairs' bootstrap method still works
        try:
            results = dre.fit(bootstrap=True, bootstrap_reps=3, bootstrap_method='pairs')
            assert results.bootstrap == True
            assert results.bootstrap_reps == 3
            print("✓ Bootstrap correctly supports only 'pairs' method")
        except Exception as e:
            pytest.fail(f"'pairs' bootstrap should work but failed with: {e}")
    
    def test_binary_variables_use_exponential_formula(self):
        """Test that binary variables use the correct e^{beta} * (m1/m0) - 1 formula."""
        # Generate data with a binary regressor
        np.random.seed(123)
        n = 200
        x_binary = np.random.binomial(1, 0.5, size=n)
        beta_0 = 0.7
        beta_1 = 1.2  # Coefficient for binary variable
        
        # Generate y with some heteroskedasticity
        u = np.random.normal(0, 0.15 + 0.05 * x_binary, n)
        log_y = beta_0 + beta_1 * x_binary + u
        y = np.exp(log_y)
        df = pd.DataFrame({'y': y, 'x': x_binary})
        
        # Fit the estimator using binary model
        dre = DoublyRobustElasticityEstimator(
            endog=df['y'],
            exog=df[['x']],
            interest='x',
            estimator_type='binary',
            elasticity=False
        )
        results = dre.fit()
        
        # Extract the components we need to verify the formula
        beta_hat = results.beta_hat[0]  # OLS estimate
        
        # Get m0 and m1 directly from the nonparametric model
        m0 = results.nonparam_model.predict(np.array([[0]]))[0]
        m1 = results.nonparam_model.predict(np.array([[1]]))[0]
        
        # Calculate expected semi-elasticity using the correct formula
        expected_semi_elasticity = np.exp(beta_hat) * (m1 / m0) - 1
        
        # Get actual estimates from the results
        actual_estimates_x0 = results.estimator_values[0][x_binary == 0]
        actual_estimates_x1 = results.estimator_values[0][x_binary == 1]
        
        # Test that all estimates equal the expected value (within tolerance)
        assert np.allclose(actual_estimates_x0, expected_semi_elasticity, atol=1e-10), (
            f"Binary estimates for x=0 don't match expected formula:\n"
            f"  Expected: {expected_semi_elasticity:.10f}\n"
            f"  Actual (mean): {np.mean(actual_estimates_x0):.10f}\n"
            f"  Max difference: {np.max(np.abs(actual_estimates_x0 - expected_semi_elasticity)):.2e}"
        )
        
        assert np.allclose(actual_estimates_x1, expected_semi_elasticity, atol=1e-10), (
            f"Binary estimates for x=1 don't match expected formula:\n"
            f"  Expected: {expected_semi_elasticity:.10f}\n"
            f"  Actual (mean): {np.mean(actual_estimates_x1):.10f}\n"
            f"  Max difference: {np.max(np.abs(actual_estimates_x1 - expected_semi_elasticity)):.2e}"
        )
        
        # Test estimate_at_point for binary variables
        point_0 = np.array([[0]])
        point_1 = np.array([[1]])
        
        est_at_0 = results.estimate_at_point(point_0, 0)
        est_at_1 = results.estimate_at_point(point_1, 0)
        
        assert np.isclose(est_at_0, expected_semi_elasticity, atol=1e-10), (
            f"estimate_at_point for x=0 doesn't match expected: {est_at_0:.10f} vs {expected_semi_elasticity:.10f}"
        )
        
        assert np.isclose(est_at_1, expected_semi_elasticity, atol=1e-10), (
            f"estimate_at_point for x=1 doesn't match expected: {est_at_1:.10f} vs {expected_semi_elasticity:.10f}"
        )
        
        # Verify it's NOT using the old incorrect formula (beta + correction)
        old_incorrect_formula = beta_hat + (m1 - m0) / ((m0 + m1) / 2)  # Approximate old logic
        assert not np.isclose(expected_semi_elasticity, old_incorrect_formula, atol=1e-3), (
            "The formula appears to still be using the old incorrect logic"
        )
        
        print(f"✓ Binary variables correctly use e^{{beta}} * (m1/m0) - 1 formula")
        print(f"  Beta_hat: {beta_hat:.4f}")
        print(f"  m0: {m0:.4f}, m1: {m1:.4f}")
        print(f"  Expected semi-elasticity: {expected_semi_elasticity:.6f}")
        print(f"  Actual estimate: {actual_estimates_x0[0]:.6f}")
    
    def test_binary_and_continuous_variables_handled_differently(self):
        """Test to ensure binary and continuous variables use different estimation approaches."""
        np.random.seed(456)
        n = 100
        
        # Create data with both binary and continuous variables
        x_binary = np.random.binomial(1, 0.6, size=n)
        x_continuous = np.random.normal(1, 0.5, n)
        
        u = np.random.normal(0, 0.2, n)
        log_y = 0.5 + 1.0 * x_binary + 0.8 * x_continuous + u
        y = np.exp(log_y)
        
        df = pd.DataFrame({'y': y, 'x_binary': x_binary, 'x_continuous': x_continuous})
        
        # Test binary variable
        dre_binary = DoublyRobustElasticityEstimator(
            endog=df['y'],
            exog=df[['x_binary', 'x_continuous']],
            interest='x_binary',
            estimator_type='binary',
            elasticity=False
        )
        results_binary = dre_binary.fit()
        
        # For binary variable, all estimates should be the same
        binary_estimates = results_binary.estimator_values[0]
        assert np.allclose(binary_estimates, binary_estimates[0], atol=1e-10), (
            "All binary estimates should be identical but they vary"
        )
        
        # Test continuous variable  
        dre_continuous = DoublyRobustElasticityEstimator(
            endog=df['y'],
            exog=df[['x_binary', 'x_continuous']],
            interest='x_continuous',
            estimator_type='ols',
            elasticity=False
        )
        results_continuous = dre_continuous.fit()
        
        # For continuous variable, estimates should vary across observations
        continuous_estimates = results_continuous.estimator_values[1]
        assert not np.allclose(continuous_estimates, continuous_estimates[0], atol=1e-5), (
            "Continuous estimates should vary across observations but they're all the same"
        )
        
        print("✓ Binary and continuous variables handled with different approaches as expected")
    
    def test_estimate_at_average_delegates_to_estimate_at_point(self):
        """Test that estimate_at_average uses estimate_at_point logic to avoid code duplication."""
        # Generate test data
        np.random.seed(789)
        n = 80
        x = np.random.normal(2, 0.8, n)
        y = np.exp(1.2 + 0.9 * x + np.random.normal(0, 0.3, n))
        df = pd.DataFrame({'y': y, 'x': x})
        
        # Create estimator
        dre = DoublyRobustElasticityEstimator(
            endog=df['y'],
            exog=df[['x']],
            interest='x',
            estimator_type='ols',
            elasticity=False
        )
        results = dre.fit()
        
        # Calculate expected result using estimate_at_point
        X_mean = np.mean(df[['x']].values, axis=0).reshape(1, -1)
        expected_result = results.estimate_at_point(X_mean, 0)
        
        # Calculate actual result using estimate_at_average
        actual_result = results.estimate_at_average(0)
        
        # They should be exactly equal (no numerical tolerance needed)
        assert actual_result == expected_result, (
            f"estimate_at_average should equal estimate_at_point at mean X:\n"
            f"  estimate_at_average: {actual_result:.10f}\n"
            f"  estimate_at_point at mean: {expected_result:.10f}\n"
            f"  Difference: {abs(actual_result - expected_result):.2e}"
        )
        
        # Test with variable name instead of index
        actual_result_by_name = results.estimate_at_average('x')
        assert actual_result_by_name == expected_result, (
            "estimate_at_average should work the same with variable name"
        )
        
        # Test with no argument (should use first variable of interest)
        actual_result_default = results.estimate_at_average()
        assert actual_result_default == expected_result, (
            "estimate_at_average should work with default argument"
        )
        
        print(f"✓ estimate_at_average correctly delegates to estimate_at_point logic")
        print(f"  Result: {actual_result:.6f}")
        print(f"  X_mean: {X_mean[0, 0]:.4f}")
    
    def test_estimate_at_average_consistent_with_binary_variables(self):
        """Test delegation consistency with binary variables."""
        np.random.seed(111)
        n = 60
        x_binary = np.random.binomial(1, 0.4, size=n)
        y = np.exp(0.8 + 1.5 * x_binary + np.random.normal(0, 0.25, n))
        df = pd.DataFrame({'y': y, 'x': x_binary})
        
        dre = DoublyRobustElasticityEstimator(
            endog=df['y'],
            exog=df[['x']],
            interest='x',
            estimator_type='binary',
            elasticity=False
        )
        results = dre.fit()
        
        # For binary variables, both methods should give the same result
        X_mean = np.mean(df[['x']].values, axis=0).reshape(1, -1)
        expected_result = results.estimate_at_point(X_mean, 0)
        actual_result = results.estimate_at_average(0)
        
        assert actual_result == expected_result, (
            f"estimate_at_average should equal estimate_at_point for binary variables:\n"
            f"  estimate_at_average: {actual_result:.10f}\n"
            f"  estimate_at_point: {expected_result:.10f}"
        )
        
        print(f"✓ estimate_at_average delegation works correctly with binary variables")
    
    def test_bootstrap_binary_formula_and_delegation_work_together(self):
        """Integration test to ensure bootstrap method restriction, binary formula, and delegation work together."""
        np.random.seed(999)
        n = 100
        
        # Create mixed data: binary and continuous
        x_binary = np.random.binomial(1, 0.5, size=n)
        x_continuous = np.random.normal(1.5, 0.6, n)
        
        u = np.random.normal(0, 0.2, n)
        log_y = 0.6 + 1.1 * x_binary + 0.7 * x_continuous + u
        y = np.exp(log_y)
        
        df = pd.DataFrame({'y': y, 'x_bin': x_binary, 'x_cont': x_continuous})
        
        # Test with binary variable
        dre_binary = DoublyRobustElasticityEstimator(
            endog=df['y'],
            exog=df[['x_bin', 'x_cont']],
            interest='x_bin',
            estimator_type='binary',
            elasticity=False
        )
        
        # Should work with pairs bootstrap (bootstrap method restriction)
        results_binary = dre_binary.fit(bootstrap=True, bootstrap_reps=3, bootstrap_method='pairs')
        
        # Should correctly handle binary formula (exponential formula)
        binary_estimates = results_binary.estimator_values[0]
        assert np.allclose(binary_estimates, binary_estimates[0], atol=1e-10), (
            "Binary estimates should be constant"
        )
        
        # estimate_at_average should use estimate_at_point (delegation)
        X_mean = np.mean(df[['x_bin', 'x_cont']].values, axis=0).reshape(1, -1)
        est_at_avg = results_binary.estimate_at_average('x_bin')
        est_at_point = results_binary.estimate_at_point(X_mean, 0)
        assert est_at_avg == est_at_point, "Delegation not working in integration test"
        
        # Should fail with residuals bootstrap (bootstrap method restriction)
        with pytest.raises(ValueError, match="Bootstrap method 'residuals' not supported"):
            dre_binary.fit(bootstrap=True, bootstrap_reps=2, bootstrap_method='residuals')
        
        print("✓ INTEGRATION: Bootstrap restriction, binary formula, and delegation all work together")


def test_binary_formula_mathematical_verification():
    """Verify that the binary formula is mathematically correct."""
    # Create a controlled scenario where we know the true values
    np.random.seed(12345)
    n = 1000  # Large sample for accuracy
    
    # Simple binary case: x is 0 or 1
    x = np.random.binomial(1, 0.5, size=n)
    
    # True parameters
    true_beta_0 = 1.0
    true_beta_1 = 0.8
    
    # Generate y with minimal noise for cleaner test
    u = np.random.normal(0, 0.05, n)  # Very small noise
    log_y = true_beta_0 + true_beta_1 * x + u
    y = np.exp(log_y)
    
    # Theoretical values
    # E[y|x=0] = exp(beta_0) * E[exp(u)|x=0]
    # E[y|x=1] = exp(beta_0 + beta_1) * E[exp(u)|x=1]
    # If E[exp(u)|x=0] ≈ E[exp(u)|x=1] ≈ c (due to small noise)
    # Then semi-elasticity ≈ exp(beta_1) - 1
    
    df = pd.DataFrame({'y': y, 'x': x})
    
    dre = DoublyRobustElasticityEstimator(
        endog=df['y'],
        exog=df[['x']],
        interest='x',
        estimator_type='binary',
        elasticity=False
    )
    results = dre.fit()
    
    # Get the estimated semi-elasticity
    estimated_semi_elasticity = results.estimator_values[0][0]
    
    # Compare with theoretical expectation
    # For small u, exp(u) ≈ 1 + u, so E[exp(u)] ≈ 1
    # This means the semi-elasticity should be approximately exp(beta_1) - 1
    theoretical_approx = np.exp(true_beta_1) - 1
    
    # Should be close to theoretical value
    assert abs(estimated_semi_elasticity - theoretical_approx) < 0.05, (
        f"Estimated semi-elasticity {estimated_semi_elasticity:.4f} too far from "
        f"theoretical {theoretical_approx:.4f}"
    )
    
    print(f"✓ Mathematical verification: Binary exponential formula is mathematically correct")
    print(f"  True beta_1: {true_beta_1}")
    print(f"  Theoretical semi-elasticity ≈ {theoretical_approx:.4f}")
    print(f"  Estimated semi-elasticity: {estimated_semi_elasticity:.4f}")


# Additional pytest utilities
@pytest.fixture
def sample_binary_data():
    """Fixture to provide sample binary regression data."""
    np.random.seed(123)
    n = 100
    x = np.random.binomial(1, 0.4, size=n)
    u = np.random.normal(0, 0.25, n)
    log_y = 0.5 + 1.2 * x + u
    y = np.exp(log_y)
    return pd.DataFrame({'y': y, 'x': x})

def test_with_fixture(sample_binary_data):
    """Example test using a pytest fixture."""
    df = sample_binary_data
    
    # Basic validation
    assert len(df) == 100
    assert set(df['x'].unique()) == {0, 1}
    assert all(df['y'] > 0)
    
    # Test the estimator
    dre = DoublyRobustElasticityEstimator(
        endog=df['y'],
        exog=df[['x']],
        interest='x',
        estimator_type='binary',
        elasticity=False
    )
    results = dre.fit()
    
    # Basic checks
    assert hasattr(results, 'beta_hat')
    assert hasattr(results, 'estimator_values')
    assert len(results.beta_hat) == 1  # Only one regressor
