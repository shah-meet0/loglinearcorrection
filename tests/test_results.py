"""
Tests for the DREEMR (DoublyRobustElasticityEstimatorModelResults) class.
"""
import numpy as np
import pytest
from unittest.mock import Mock, patch
from loglinearcorrection.results import DREEMR, DoublyRobustElasticityEstimatorModelResults


class TestDREEMRInit:
    """Tests for DREEMR initialization."""
    
    def test_basic_initialization(self):
        """Test basic initialization of DREEMR."""
        elasticities = {
            'x1': {'estimate': 0.5, 'type': 'continuous', 'index': 0},
            'x2': {'estimate': 0.3, 'type': 'binary', 'index': 1}
        }
        beta = np.array([0.2, 0.1])
        exog_names = ['x1', 'x2']
        endog_name = 'y'
        n_folds = 5
        nobs = 100
        variable_types = {0: 'continuous', 1: 'binary'}
        fold_diagnostics = {
            'ols_coefficients': [np.array([0.2, 0.1])],
            'moment_means': np.array([0.5, 0.3]),
            'm_results': [Mock()],
            'f_results': [Mock()],
            'alpha_x': [np.array([[0.1, 0.2]])],
            'theta_x': [np.array([[0.5, 0.3]])]
        }
        
        results = DREEMR(
            elasticities=elasticities,
            beta=beta,
            exog_names=exog_names,
            endog_name=endog_name,
            n_folds=n_folds,
            nobs=nobs,
            variable_types=variable_types,
            fold_diagnostics=fold_diagnostics
        )
        
        assert results.elasticities == elasticities
        assert np.array_equal(results.beta, beta)
        assert results.exog_names == exog_names
        assert results.endog_name == endog_name
        assert results.n_folds == n_folds
        assert results.nobs == nobs
        assert results.variable_types == variable_types
        assert results.fold_diagnostics == fold_diagnostics
    
    def test_alias_works(self):
        """Test that DREEMR alias works correctly."""
        assert DREEMR is DoublyRobustElasticityEstimatorModelResults


class TestDREEMRPredictions:
    """Tests for DREEMR prediction methods."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.elasticities = {
            'x1': {'estimate': 0.5, 'type': 'continuous', 'index': 0},
            'x2': {'estimate': 0.3, 'type': 'binary', 'index': 1}
        }
        self.beta = np.array([0.2, 0.1])
        
        # Create mock NPModelResults
        self.mock_m_results = []
        for i in range(3):  # 3 folds
            mock_m = Mock()
            mock_m.predict.return_value = np.array([1.0, 1.1])
            mock_m.predict_semi_elasticity.return_value = np.array([0.3, 0.35])
            self.mock_m_results.append(mock_m)
        
        self.fold_diagnostics = {
            'ols_coefficients': [self.beta] * 3,
            'moment_means': np.array([0.5, 0.3]),
            'm_results': self.mock_m_results,
            'f_results': [Mock()] * 3,
            'alpha_x': [np.array([[0.1, 0.2]])] * 3,
            'theta_x': [np.array([[0.5, 0.3]])] * 3
        }
        
        self.results = DREEMR(
            elasticities=self.elasticities,
            beta=self.beta,
            exog_names=['x1', 'x2'],
            endog_name='y',
            n_folds=3,
            nobs=100,
            variable_types={0: 'continuous', 1: 'binary'},
            fold_diagnostics=self.fold_diagnostics
        )
    
    def test_predict_log_y_scalar(self):
        """Test predict_log_y with scalar input."""
        x = np.array([1.0, 2.0])
        pred = self.results.predict_log_y(x)
        
        expected = np.dot(x, self.beta)  # 1.0*0.2 + 2.0*0.1 = 0.4
        assert np.isclose(pred, expected)
    
    def test_predict_log_y_matrix(self):
        """Test predict_log_y with matrix input."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        pred = self.results.predict_log_y(x)
        
        expected = x @ self.beta
        np.testing.assert_array_almost_equal(pred, expected)
    
    def test_predict_y(self):
        """Test predict_y method."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        pred = self.results.predict_y(x)
        
        # Should call predict on each m_results and average
        for mock_m in self.mock_m_results:
            mock_m.predict.assert_called_once()
            np.testing.assert_array_equal(mock_m.predict.call_args[0][0], x)
        
        # Expected: exp(β·x) * m(x)
        log_pred = self.results.predict_log_y(x)
        m_avg = np.mean([m.predict.return_value for m in self.mock_m_results], axis=0)
        expected = np.exp(log_pred) * m_avg
        
        np.testing.assert_array_almost_equal(pred, expected)
    
    def test_predict_semi_elasticity_continuous(self):
        """Test predict_semi_elasticity for continuous variable."""
        x = np.array([[1.0, 0.0], [2.0, 1.0]])
        var_index = 0  # Continuous variable
        
        pred = self.results.predict_semi_elasticity(x, var_index)
        
        # Should call predict_semi_elasticity on each m_results
        for mock_m in self.mock_m_results:
            mock_m.predict_semi_elasticity.assert_called_once_with(x, var_index)
        
        # Expected: β_k + average(m_k/m across folds)
        m_semi_avg = np.mean([m.predict_semi_elasticity.return_value 
                              for m in self.mock_m_results], axis=0)
        expected = self.beta[var_index] + m_semi_avg
        
        np.testing.assert_array_almost_equal(pred, expected)
    
    def test_predict_semi_elasticity_binary(self):
        """Test predict_semi_elasticity for binary variable."""
        x = np.array([[1.0, 0.0], [2.0, 1.0]])
        var_index = 1  # Binary variable
        
        pred = self.results.predict_semi_elasticity(x, var_index)
        
        # Should call predict for m(x) and m(x_shifted)
        for mock_m in self.mock_m_results:
            # Should be called twice: once for m(x), once for m(x_shifted)
            assert mock_m.predict.call_count == 2
        
        # Check the shifted x values
        first_call = self.mock_m_results[0].predict.call_args_list[0][0][0]
        second_call = self.mock_m_results[0].predict.call_args_list[1][0][0]
        
        # Second call should have flipped binary values
        np.testing.assert_array_equal(first_call[:, var_index], x[:, var_index])
        np.testing.assert_array_equal(second_call[:, var_index], 1 - x[:, var_index])
    
    def test_predict_semi_elasticity_ordinal(self):
        """Test predict_semi_elasticity for ordinal variable."""
        # Update variable types to include ordinal
        self.results.variable_types[2] = 'ordinal'
        
        x = np.array([[1.0, 0.0, 2.0], [2.0, 1.0, 3.0]])
        var_index = 2  # Ordinal variable
        
        # Update beta to have 3 components
        self.results.beta = np.array([0.2, 0.1, 0.15])
        
        pred = self.results.predict_semi_elasticity(x, var_index)
        
        # Should call predict for m(x) and m(x+1)
        for mock_m in self.mock_m_results:
            assert mock_m.predict.call_count == 2


class TestDREEMRVariance:
    """Tests for variance computation."""
    
    @patch('loglinearcorrection.results.VarianceModel')
    def test_standard_errors_computed_lazily(self, mock_variance_model_class):
        """Test that standard errors are computed lazily on first access."""
        # Create mock variance model and results
        mock_var_model = Mock()
        mock_variance_model_class.return_value = mock_var_model
        
        mock_var_results = Mock()
        mock_var_results.standard_errors = np.array([0.1, 0.2])
        mock_var_results.confidence_intervals = {
            'x1': {'lower': 0.304, 'upper': 0.696},
            'x2': {'lower': -0.092, 'upper': 0.692}
        }
        mock_var_model.fit.return_value = mock_var_results
        
        results = DREEMR(
            elasticities={
                'x1': {'estimate': 0.5, 'type': 'continuous', 'index': 0},
                'x2': {'estimate': 0.3, 'type': 'binary', 'index': 1}
            },
            beta=np.array([0.2, 0.1]),
            exog_names=['x1', 'x2'],
            endog_name='y',
            n_folds=3,
            nobs=100,
            variable_types={0: 'continuous', 1: 'binary'},
            fold_diagnostics={
                'ols_coefficients': [],
                'moment_means': np.array([0.5, 0.3]),
                'm_results': [],
                'f_results': [],
                'alpha_x': [np.array([[0.1, 0.2]])],
                'theta_x': [np.array([[0.5, 0.3]])]
            }
        )
        
        # Should not be computed yet
        assert results._standard_errors is None
        
        # Access standard_errors property
        se = results.standard_errors
        
        # Should create VarianceModel and call fit
        mock_variance_model_class.assert_called_once()
        mock_var_model.fit.assert_called_once()
        
        # Should use results from VarianceModelResults
        np.testing.assert_array_almost_equal(se, [0.1, 0.2])
        
        # Should be cached
        se2 = results.standard_errors
        assert se is se2
        mock_variance_model_class.assert_called_once()  # Still only once
    
    @patch('loglinearcorrection.results.VarianceModel')  
    def test_confidence_intervals_computed(self, mock_variance_model_class):
        """Test that confidence intervals are computed correctly."""
        # Create mock variance model and results
        mock_var_model = Mock()
        mock_variance_model_class.return_value = mock_var_model
        
        mock_var_results = Mock()
        mock_var_results.standard_errors = np.array([0.1, 0.2])
        mock_var_results.confidence_intervals = {
            'x1': {'lower': 0.304, 'upper': 0.696},
            'x2': {'lower': -0.092, 'upper': 0.692}
        }
        mock_var_model.fit.return_value = mock_var_results
        
        results = DREEMR(
            elasticities={
                'x1': {'estimate': 0.5, 'type': 'continuous', 'index': 0},
                'x2': {'estimate': 0.3, 'type': 'binary', 'index': 1}
            },
            beta=np.array([0.2, 0.1]),
            exog_names=['x1', 'x2'],
            endog_name='y',
            n_folds=3,
            nobs=100,
            variable_types={0: 'continuous', 1: 'binary'},
            fold_diagnostics={
                'ols_coefficients': [],
                'moment_means': np.array([0.5, 0.3]),
                'm_results': [],
                'f_results': [],
                'alpha_x': [np.array([[0.1, 0.2]])],
                'theta_x': [np.array([[0.5, 0.3]])]
            }
        )
        
        ci = results.confidence_intervals
        
        # Check structure
        assert 'x1' in ci
        assert 'x2' in ci
        
        # Check values from mock
        assert np.isclose(ci['x1'][0], 0.304)
        assert np.isclose(ci['x1'][1], 0.696)
        assert np.isclose(ci['x2'][0], -0.092)
        assert np.isclose(ci['x2'][1], 0.692)
    
    def test_variance_fallback_when_module_missing(self):
        """Test fallback behavior when variance module is missing."""
        results = DREEMR(
            elasticities={
                'x1': {'estimate': 0.5, 'type': 'continuous', 'index': 0}
            },
            beta=np.array([0.2]),
            exog_names=['x1'],
            endog_name='y',
            n_folds=3,
            nobs=100,
            variable_types={0: 'continuous'},
            fold_diagnostics={
                'ols_coefficients': [],
                'moment_means': np.array([0.5]),
                'm_results': [],
                'f_results': [],
                'alpha_x': [],
                'theta_x': []
            }
        )
        
        # Mock import error
        with patch('loglinearcorrection.results.VarianceModel', 
                   side_effect=ImportError):
            with pytest.warns(UserWarning, match="variance.py module not found"):
                se = results.standard_errors
            
            assert np.isnan(se[0])
            
            ci = results.confidence_intervals
            assert np.isnan(ci['x1'][0])
            assert np.isnan(ci['x1'][1])
    
    @patch('loglinearcorrection.results.VarianceModel')
    def test_compute_variance_method(self, mock_variance_model_class):
        """Test compute_variance method for recomputing with different params."""
        # Create mock variance models for different methods
        mock_var_model_influence = Mock()
        mock_var_model_gmm = Mock()
        
        # Setup different results for each method
        mock_results_influence = Mock()
        mock_results_influence.standard_errors = np.array([0.1])
        mock_var_model_influence.fit.return_value = mock_results_influence
        
        mock_results_gmm = Mock()
        mock_results_gmm.standard_errors = np.array([0.15])
        mock_var_model_gmm.fit.return_value = mock_results_gmm
        
        # Configure mock to return different models based on method
        def create_model(**kwargs):
            if kwargs.get('method') == 'gmm':
                return mock_var_model_gmm
            return mock_var_model_influence
        
        mock_variance_model_class.side_effect = create_model
        
        results = DREEMR(
            elasticities={
                'x1': {'estimate': 0.5, 'type': 'continuous', 'index': 0}
            },
            beta=np.array([0.2]),
            exog_names=['x1'],
            endog_name='y',
            n_folds=3,
            nobs=100,
            variable_types={0: 'continuous'},
            fold_diagnostics={
                'ols_coefficients': [],
                'moment_means': np.array([0.5]),
                'm_results': [],
                'f_results': [],
                'alpha_x': [],
                'theta_x': []
            }
        )
        
        # Compute with default (influence) method
        var_results1 = results.compute_variance()
        assert var_results1 == mock_results_influence
        
        # Recompute with GMM method
        var_results2 = results.compute_variance(method='gmm', robust=True)
        assert var_results2 == mock_results_gmm
        
        # Check that VarianceModel was created with correct params
        calls = mock_variance_model_class.call_args_list
        assert calls[-1][1]['method'] == 'gmm'
        assert calls[-1][1]['robust'] is True


class TestDREEMRMethods:
    """Tests for other DREEMR methods."""
    
    def test_get_elasticity(self):
        """Test get_elasticity method."""
        results = DREEMR(
            elasticities={
                'x1': {'estimate': 0.5, 'type': 'continuous', 'index': 0},
                'x2': {'estimate': 0.3, 'type': 'binary', 'index': 1}
            },
            beta=np.array([0.2, 0.1]),
            exog_names=['x1', 'x2'],
            endog_name='y',
            n_folds=3,
            nobs=100,
            variable_types={0: 'continuous', 1: 'binary'},
            fold_diagnostics={
                'ols_coefficients': [],
                'moment_means': np.array([0.5, 0.3]),
                'm_results': [],
                'f_results': [],
                'alpha_x': [],
                'theta_x': []
            }
        )
        
        # Mock standard errors
        with patch.object(results, 'standard_errors', np.array([0.1, 0.2])):
            est, se = results.get_elasticity('x1')
            assert est == 0.5
            assert se == 0.1
            
            est, se = results.get_elasticity('x2')
            assert est == 0.3
            assert se == 0.2
        
        # Test with missing variable
        with pytest.raises(KeyError, match="Variable 'x3' not found"):
            results.get_elasticity('x3')
    
    def test_summary(self, capsys):
        """Test summary method prints correctly."""
        results = DREEMR(
            elasticities={
                'x1': {'estimate': 0.5, 'type': 'continuous', 'index': 0}
            },
            beta=np.array([0.2]),
            exog_names=['x1'],
            endog_name='y',
            n_folds=3,
            nobs=100,
            variable_types={0: 'continuous'},
            fold_diagnostics={
                'ols_coefficients': [],
                'moment_means': np.array([0.5]),
                'm_results': [],
                'f_results': [],
                'alpha_x': [],
                'theta_x': []
            }
        )
        
        # Mock standard errors and CI
        with patch.object(results, '_standard_errors', np.array([0.1])):
            with patch.object(results, '_confidence_intervals', 
                            {'x1': (0.304, 0.696)}):
                results.summary()
        
        captured = capsys.readouterr()
        
        assert "Doubly Robust Elasticity Estimator Results" in captured.out
        assert "Dependent variable: y" in captured.out
        assert "Number of observations: 100" in captured.out
        assert "Number of folds: 3" in captured.out
        assert "x1" in captured.out
        assert "continuous" in captured.out
        assert "0.5000" in captured.out  # estimate
        assert "[0.3040, 0.6960]" in captured.out  # CI
        assert "OLS Coefficients" in captured.out
    
    def test_repr(self):
        """Test string representation."""
        results = DREEMR(
            elasticities={
                'x1': {'estimate': 0.5, 'type': 'continuous', 'index': 0},
                'x2': {'estimate': 0.3, 'type': 'binary', 'index': 1}
            },
            beta=np.array([0.2, 0.1]),
            exog_names=['x1', 'x2'],
            endog_name='y',
            n_folds=3,
            nobs=100,
            variable_types={0: 'continuous', 1: 'binary'},
            fold_diagnostics={}
        )
        
        repr_str = repr(results)
        assert "DoublyRobustElasticityEstimatorModelResults" in repr_str
        assert "n_vars=2" in repr_str
        assert "nobs=100" in repr_str
        assert "n_folds=3" in repr_str
