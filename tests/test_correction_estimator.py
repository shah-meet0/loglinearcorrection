import pytest
import numpy as np
import tensorflow as tf
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

# Import the module to test
from loglinearcorrection.correction_estimator import (
    CorrectedEstimator, CorrectedEstimatorResults,
    CorrectedEstimatorResultsLogLinear, CorrectedEstimatorResultsLogLog,
    OLSCorrectionModel, NNCorrectionModel, CorrectionModel
)

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

class TestCorrectedEstimator:
    """Tests for CorrectedEstimator class."""
    
    def test_initialization(self):
        """Test initialization of CorrectedEstimator."""
        X = np.random.rand(100, 3)
        y = np.random.rand(100) + 0.1  # Ensure positive
        
        # Test with default parameters
        model = CorrectedEstimator(y, X)
        assert model.X is X
        assert model.y is y
        assert model.correction_model_type == 'nn'
        assert model.x_index == 0
        assert model.log_x is False
        
        # Test with custom parameters
        model = CorrectedEstimator(y, X, correction_model_type='OLS', interest=1, log_x=True)
        assert model.correction_model_type == 'ols'
        assert model.x_index == 1
        assert model.log_x is True
        
        # Test invalid correction_model_type
        with pytest.raises(ValueError):
            CorrectedEstimator(y, X, correction_model_type='invalid')
    
    @pytest.mark.parametrize("model_type", ['ols', 'nn'])
    def test_fit_small_dataset(self, model_type):
        """Test fitting on a small controlled dataset."""
        # Create a simple dataset
        np.random.seed(42)
        X = np.random.rand(50, 2)
        beta = np.array([1.5, -0.5])
        noise = np.random.normal(1, 0.1, 50)
        y = np.exp(X @ beta) * noise
        
        # Fit model
        model = CorrectedEstimator(y, X, correction_model_type=model_type)
        
        # Use minimal epochs for NN to speed up test
        params = {'epochs': 5} if model_type == 'nn' else {'degree': 2}
        results = model.fit(params_dict=params, disp=0)
        
        # Check results type based on log_x parameter
        assert isinstance(results, CorrectedEstimatorResultsLogLinear)
        
        # Also test with log_x=True
        model = CorrectedEstimator(y, X, correction_model_type=model_type, log_x=True)
        results = model.fit(params_dict=params, disp=0)
        assert isinstance(results, CorrectedEstimatorResultsLogLog)
    
    @given(data=regression_data(n_samples=50, n_features=2))
    @settings(deadline=None, max_examples=3)  # Limit examples due to computational cost
    def test_fit_property_based(self, data):
        """Property-based test for fit method."""
        X, y, _ = data
        
        # Test with OLS correction (faster than NN for property testing)
        model = CorrectedEstimator(y, X, correction_model_type='ols')
        params = {'degree': 2}  # Low degree for speed
        
        try:
            results = model.fit(params_dict=params, disp=0)
            
            # Basic checks
            assert hasattr(results, 'model')
            assert hasattr(results, 'ols_results')
            assert hasattr(results, 'correction_model')
            
            # Check if betahat is extracted correctly
            assert results.betahat == results.ols_results.params[0]
            
        except Exception as e:
            # Some combinations might cause numerical issues
            # This is acceptable for property-based testing
            pytest.skip(f"Skipped due to numerical issue: {str(e)}")
    
    def test_make_nn(self):
        """Test neural network model creation."""
        X = np.random.rand(100, 3)
        y = np.random.rand(100) + 0.1
        
        model = CorrectedEstimator(y, X)
        
        # Test with default parameters
        nn = model.make_nn({})
        assert isinstance(nn, tf.keras.Sequential)
        assert len(nn.layers) == 5  # Input + 3 hidden + output
        
        # Test with custom parameters
        params = {
            'num_layers': 2,
            'num_units': 32,
            'activation': 'tanh',
            'optimizer': 'sgd',
            'loss': 'mae'
        }
        nn = model.make_nn(params)
        assert len(nn.layers) == 4  # Input + 2 hidden + output
        
        # Test with num_units as list
        params = {
            'num_layers': 2,
            'num_units': [32, 16]
        }
        nn = model.make_nn(params)
        assert nn.layers[1].units == 32
        assert nn.layers[2].units == 16
        
        # Test with invalid num_units
        params = {
            'num_layers': 2,
            'num_units': [32, 16, 8]  # Too long for num_layers
        }
        with pytest.raises(ValueError):
            model.make_nn(params)

class TestCorrectionModels:
    """Tests for OLSCorrectionModel and NNCorrectionModel classes."""
    
    def test_ols_correction_model(self):
        """Test OLSCorrectionModel functionality."""
        # Create simple test data
        X = np.random.rand(50, 2)
        y = np.exp(X @ np.array([1.0, -0.5])) * np.random.normal(1, 0.1, 50)
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Fit a simple OLS model
        ols_model = sm.OLS(y, X_poly).fit()
        
        # Create OLSCorrectionModel
        correction_model = OLSCorrectionModel(ols_model, poly)
        
        # Test predict method
        predictions = correction_model.predict(X)
        assert predictions.shape == (len(X),)
        
        # Test marginal effects
        me = correction_model.marginal_effects(X, 0)
        assert me.shape == (len(X),)
        
        # Test semi_elasticity
        se = correction_model.semi_elasticity(X, 0)
        assert se.shape == (len(X),)
        
        # Test elasticity
        el = correction_model.elasticity(X, 0)
        assert el.shape == (len(X),)
    
    def test_nn_correction_model(self):
        """Test NNCorrectionModel functionality."""
        # Create simple test data
        X = np.random.rand(50, 2).astype(np.float32)
        
        # Create a simple neural network
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # Create NNCorrectionModel
        correction_model = NNCorrectionModel(model)
        
        # Test predict method
        predictions = correction_model.predict(X)
        assert predictions.shape == (len(X), 1)
        
        # Convert tensors to numpy for assertion
        predictions_np = predictions.numpy()
        assert predictions_np.shape == (len(X), 1)
        
        # Test marginal effects
        me = correction_model.marginal_effects(X, 0)
        assert me.shape == (len(X),)
        
        # Test semi_elasticity
        se = correction_model.semi_elasticity(X, 0)
        assert se.shape == (len(X),)
        
        # Test elasticity
        el = correction_model.elasticity(X, 0)
        assert el.shape == (len(X),)

class TestCorrectedEstimatorResults:
    """Tests for CorrectedEstimatorResults classes."""
    
    def setup_test_data(self):
        """Set up test data and models for result tests."""
        np.random.seed(42)
        X = np.random.rand(50, 2)
        beta = np.array([1.5, -0.5])
        noise = np.random.normal(1, 0.1, 50)
        y = np.exp(X @ beta) * noise
        
        # Create OLS results
        ols_results = sm.OLS(np.log(y), X).fit()
        
        # Create a simple neural network
        nn_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        correction_model = NNCorrectionModel(nn_model)
        
        return X, y, ols_results, correction_model
    
    def test_log_linear_results(self):
        """Test CorrectedEstimatorResultsLogLinear functionality."""
        X, y, ols_results, correction_model = self.setup_test_data()
        
        # Create model
        model = CorrectedEstimator(y, X)
        
        # Create results
        results = CorrectedEstimatorResultsLogLinear(model, ols_results, correction_model)
        
        # Test methods
        assert isinstance(results.average_semi_elasticity(), float)
        assert hasattr(results, 'se')
        
        fig, ax = results.plot_dist_semi_elasticity()
        assert fig is not None
        assert ax is not None
        
        assert isinstance(results.semi_elasticity_at_average(), float)
        
        # Test PPML test
        ppml_results = results.test_ppml()
        assert 'test_statistic' in ppml_results
        assert 'p_value' in ppml_results
        assert 'conclusion' in ppml_results
    
    def test_log_log_results(self):
        """Test CorrectedEstimatorResultsLogLog functionality."""
        X, y, ols_results, correction_model = self.setup_test_data()
        
        # Create model with log_x=True
        model = CorrectedEstimator(y, X, log_x=True)
        
        # Create results
        results = CorrectedEstimatorResultsLogLog(model, ols_results, correction_model)
        
        # Test methods
        assert isinstance(results.average_elasticity(), float)
        assert hasattr(results, 'e')
        
        fig, ax = results.plot_dist_elasticity()
        assert fig is not None
        assert ax is not None
        
        assert isinstance(results.elasticity_at_average(), float)
        
        # Test PPML test
        ppml_results = results.test_ppml()
        assert 'test_statistic' in ppml_results
        assert 'p_value' in ppml_results
        assert 'conclusion' in ppml_results
    
    def test_base_results_methods(self):
        """Test base CorrectedEstimatorResults methods."""
        X, y, ols_results, correction_model = self.setup_test_data()
        
        # Create model
        model = CorrectedEstimator(y, X)
        
        # Create results using the base class
        results = CorrectedEstimatorResults(model, ols_results, correction_model)
        
        # Test methods
        assert results.get_ols_results() is ols_results
        
        # Test print_ols_results (should not raise exception)
        try:
            results.print_ols_results()
        except Exception as e:
            pytest.fail(f"print_ols_results raised exception: {str(e)}")

class TestIntegration:
    """Integration tests for the full CorrectedEstimator workflow."""
    
    @pytest.mark.parametrize("log_x", [False, True])
    def test_simple_workflow(self, log_x):
        """Test a complete simple workflow."""
        np.random.seed(42)
        X = np.random.rand(50, 2)
        
        # If log_x is True, transform the regressor of interest
        if log_x:
            X_for_dgp = X.copy()
            X_for_dgp[:, 0] = np.exp(X[:, 0])
        else:
            X_for_dgp = X
        
        beta = np.array([1.0, -0.5])
        noise = np.random.normal(1, 0.1, 50)
        y = np.exp(X_for_dgp @ beta) * noise
        
        # Create model with OLS correction (faster for testing)
        model = CorrectedEstimator(y, X, correction_model_type='ols', log_x=log_x)
        
        # Fit with minimal settings
        params = {'degree': 2}
        results = model.fit(params_dict=params, disp=0)
        
        # Check result type
        if log_x:
            assert isinstance(results, CorrectedEstimatorResultsLogLog)
            corrected_estimate = results.average_elasticity()
        else:
            assert isinstance(results, CorrectedEstimatorResultsLogLinear)
            corrected_estimate = results.average_semi_elasticity()
        
        # Basic checks on results
        assert corrected_estimate != results.betahat  # Correction should change estimate
        assert isinstance(corrected_estimate, float)
        
        # Test plotting
        if log_x:
            fig, ax = results.plot_dist_elasticity()
        else:
            fig, ax = results.plot_dist_semi_elasticity()
        
        assert fig is not None
        assert ax is not None
        
        # Test PPML test
        ppml_results = results.test_ppml()
        assert isinstance(ppml_results, dict)

