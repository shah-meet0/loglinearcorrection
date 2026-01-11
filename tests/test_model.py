import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock, call
import statsmodels.api as sm

from loglinearcorrection.model import DoublyRobustElasticityEstimatorModel


class TestDoublyRobustElasticityEstimatorModelInit:
    """Tests for DoublyRobustElasticityEstimatorModel initialization."""

    def test_basic_initialization_numpy(self):
        """Test basic initialization with numpy arrays."""
        endog = np.array([1.0, 2.0, 3.0, 4.0])
        exog = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])

        model = DoublyRobustElasticityEstimatorModel(endog, exog)

        assert model.nobs == 4
        assert model.endog.shape == (4,)
        assert model.exog.shape == (4, 2)
        assert model.weights is None
        assert model.exog_names == ["x1", "x2"]  # Default names
        assert model.endog_names == "y"  # Default name

    def test_initialization_pandas_dataframe(self):
        """Test initialization with pandas DataFrame preserves names."""
        endog = pd.Series([1.0, 2.0, 3.0, 4.0], name="y")
        exog = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [2.0, 3.0, 4.0, 5.0],
            "fe": [0, 0, 1, 1]
        })

        model = DoublyRobustElasticityEstimatorModel(endog, exog)

        assert model.endog_names == "y"
        assert model.exog_names == ["x1", "x2", "fe"]
        assert model.nobs == 4

    def test_initialization_pandas_series(self):
        """Test initialization with pandas Series for exog."""
        endog = pd.Series([1.0, 2.0, 3.0], name="outcome")
        exog = pd.Series([1.0, 2.0, 3.0], name="predictor")

        model = DoublyRobustElasticityEstimatorModel(endog, exog)

        assert model.endog_names == "outcome"
        assert model.exog_names == ["predictor"]
        assert model.exog.shape == (3, 1)

    def test_initialization_with_weights(self):
        """Test initialization with observation weights."""
        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([[1.0], [2.0], [3.0]])
        weights = np.array([1.0, 0.5, 2.0])

        model = DoublyRobustElasticityEstimatorModel(endog, exog, weights=weights)

        assert model.weights is not None
        assert np.array_equal(model.weights, weights)

    def test_fixed_effects_and_interest_stored(self):
        """Test fixed_effects, interest, and ordinal parameters are stored."""
        endog = np.array([1.0, 2.0, 3.0, 4.0])
        exog = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0],
            "fe": [0, 0, 1, 1],
            "x2": [2.0, 3.0, 4.0, 5.0],
            "x3": [1, 2, 3, 2]
        })

        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, 
            fixed_effects=["fe"], 
            interest=["x1"],
            ordinal=["x3"]
        )

        assert model.fixed_effects == ["fe"]
        assert model.interest == ["x1"]
        assert model.ordinal == ["x3"]

    def test_non_numeric_endog_raises_error(self):
        """Test that non-numeric endog raises ValueError."""
        endog = np.array(["a", "b", "c"])
        exog = np.array([[1.0], [2.0], [3.0]])

        with pytest.raises(ValueError, match="endog must contain only numeric data"):
            DoublyRobustElasticityEstimatorModel(endog, exog)

    def test_non_numeric_exog_raises_error(self):
        """Test that non-numeric exog raises ValueError."""
        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([["a"], ["b"], ["c"]])

        with pytest.raises(ValueError, match="exog must contain only numeric data"):
            DoublyRobustElasticityEstimatorModel(endog, exog)

    def test_non_numeric_weights_raises_error(self):
        """Test that non-numeric weights raises ValueError."""
        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([[1.0], [2.0], [3.0]])
        weights = np.array(["a", "b", "c"])

        with pytest.raises(ValueError, match="weights must contain only numeric data"):
            DoublyRobustElasticityEstimatorModel(endog, exog, weights=weights)


class TestDoublyRobustElasticityEstimatorModelFit:
    """Tests for the fit() method of DoublyRobustElasticityEstimatorModel."""
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NPModel')
    def test_basic_fit_continuous_variable(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test basic fit with a single continuous variable."""
        # Setup data
        np.random.seed(42)
        n = 100
        endog = np.exp(np.random.randn(n))  # Log-linear model appropriate data
        exog = np.random.randn(n, 1)
        
        # Create mock models and results
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Create mock NPModelResults
        mock_npm_results = Mock()
        mock_npm_results.predict.return_value = np.ones(10)  # m(x)
        mock_npm_results.predict_semi_elasticity.return_value = 0.1 * np.ones(10)  # m_k(x)/m(x)
        mock_npm.fit.return_value = mock_npm_results
        
        # Create mock DensityModelResults
        mock_dm_results = Mock()
        mock_dm_results.predict.return_value = 0.5 * np.ones(10)  # f(x)
        mock_dm_results.predict_semi_elasticity.return_value = 0.05 * np.ones(10)  # f_k(x)/f(x)
        mock_dm.fit.return_value = mock_dm_results
        
        # Mock DREEMR results
        mock_results = Mock()
        mock_results_class.return_value = mock_results
        
        # Fit model
        model = DoublyRobustElasticityEstimatorModel(endog, exog)
        result = model.fit(n_folds=2, random_state=42)
        
        # Verify NPModel was instantiated and called
        assert mock_npm_class.call_count == 2  # Once per fold
        assert mock_npm.fit.call_count == 2
        
        # Verify DensityModel was instantiated and called
        assert mock_dm_class.call_count == 2  # Once per fold
        assert mock_dm.fit.call_count == 2
        
        # Verify results object was created
        mock_results_class.assert_called_once()
        assert result == mock_results
        
        # Check that fold_diagnostics contains Results objects
        call_args = mock_results_class.call_args[1]
        fold_diagnostics = call_args['fold_diagnostics']
        assert 'm_results' in fold_diagnostics
        assert 'f_results' in fold_diagnostics
        assert len(fold_diagnostics['m_results']) == 2  # One per fold
        assert len(fold_diagnostics['f_results']) == 2  # One per fold
        assert fold_diagnostics['m_results'][0] == mock_npm_results
        assert fold_diagnostics['f_results'][0] == mock_dm_results
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NPModel')
    def test_fit_with_binary_variable(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test fit with a binary variable."""
        # Setup data with binary variable
        np.random.seed(42)
        n = 100
        endog = np.exp(np.random.randn(n))
        exog = np.random.choice([0, 1], size=(n, 1))
        
        # Create mock models and results
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Create mock NPModelResults
        mock_npm_results = Mock()
        mock_npm_results.predict.return_value = np.ones(50)
        mock_npm.fit.return_value = mock_npm_results
        
        # Create mock DensityModelResults
        mock_dm_results = Mock()
        mock_dm_results.predict.return_value = 0.5 * np.ones(50)
        mock_dm.fit.return_value = mock_dm_results
        
        # Mock variable type detection to return binary
        with patch('loglinearcorrection.model._detect_variable_types') as mock_detect:
            mock_detect.return_value = {0: "binary"}
            
            model = DoublyRobustElasticityEstimatorModel(endog, exog)
            result = model.fit(n_folds=2, random_state=42)
        
        # For binary variables, should call predict with shifted values
        # Check that predict was called multiple times (for m(x) and m(x+Δ))
        assert mock_npm_results.predict.call_count >= 4  # At least 2 per fold (m(x) and m(x+Δ))
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NPModel')
    def test_fit_with_ordinal_variable(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test fit with an ordinal variable."""
        # Setup data
        np.random.seed(42)
        n = 100
        endog = np.exp(np.random.randn(n))
        exog = np.random.choice([1, 2, 3, 4], size=(n, 1))
        
        # Create mock models and results
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Create mock NPModelResults
        mock_npm_results = Mock()
        mock_npm_results.predict.return_value = np.ones(25)  # For test set of each fold
        mock_npm.fit.return_value = mock_npm_results
        
        # Create mock DensityModelResults  
        mock_dm_results = Mock()
        mock_dm_results.predict.return_value = 0.5 * np.ones(25)
        mock_dm.fit.return_value = mock_dm_results
        
        model = DoublyRobustElasticityEstimatorModel(endog, exog, ordinal=[0])
        result = model.fit(n_folds=4, random_state=42)
        
        # For ordinal, should call predict with x+Δ, x-Δ
        assert mock_npm_results.predict.call_count >= 12  # m(x), m(x+Δ), m(x-Δ) per fold
        assert mock_dm_results.predict.call_count >= 8  # f(x), f(x-Δ) per fold
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NPModel')
    def test_results_structure(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test that results object has correct structure."""
        # Setup data
        np.random.seed(42)
        n = 50
        endog = np.exp(np.random.randn(n))
        exog = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.choice([0, 1], n)
        })
        
        # Create mock models and results
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Create mock NPModelResults
        mock_npm_results = Mock()
        mock_npm_results.predict.return_value = np.ones(25)
        mock_npm_results.predict_semi_elasticity.return_value = 0.1 * np.ones(25)
        mock_npm.fit.return_value = mock_npm_results
        
        # Create mock DensityModelResults
        mock_dm_results = Mock()
        mock_dm_results.predict.return_value = 0.5 * np.ones(25)
        mock_dm_results.predict_semi_elasticity.return_value = 0.05 * np.ones(25)
        mock_dm.fit.return_value = mock_dm_results
        
        model = DoublyRobustElasticityEstimatorModel(endog, exog)
        result = model.fit(n_folds=2, random_state=42)
        
        # Check DREEMR was called with correct arguments
        call_args = mock_results_class.call_args[1]
        
        assert 'elasticities' in call_args
        assert 'beta' in call_args
        assert 'exog_names' in call_args
        assert 'endog_name' in call_args
        assert 'n_folds' in call_args
        assert 'nobs' in call_args
        assert 'variable_types' in call_args
        assert 'fold_diagnostics' in call_args
        
        # Check elasticities structure
        elasticities = call_args['elasticities']
        assert isinstance(elasticities, dict)
        for var_name, var_info in elasticities.items():
            assert 'estimate' in var_info
            assert 'type' in var_info
            assert 'index' in var_info
        
        # Check fold diagnostics now contains Results objects
        diagnostics = call_args['fold_diagnostics']
        assert 'ols_coefficients' in diagnostics
        assert 'moment_means' in diagnostics
        assert 'm_results' in diagnostics  # NPModelResults objects
        assert 'f_results' in diagnostics  # DensityModelResults objects
        assert 'alpha_x' in diagnostics  # Alpha arrays
        assert 'theta_x' in diagnostics  # Theta arrays
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NPModel')
    def test_npm_initialization_with_params(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test that NPModel is initialized with variable_types and params."""
        # Setup data
        np.random.seed(42)
        n = 50
        endog = np.exp(np.random.randn(n))
        exog = np.random.randn(n, 2)
        
        # Create mock models and results
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Create mock Results
        mock_npm_results = Mock()
        mock_npm_results.predict.return_value = np.ones(25)
        mock_npm_results.predict_semi_elasticity.return_value = 0.1 * np.ones(25)
        mock_npm.fit.return_value = mock_npm_results
        
        mock_dm_results = Mock()
        mock_dm_results.predict.return_value = 0.5 * np.ones(25)
        mock_dm_results.predict_semi_elasticity.return_value = 0.05 * np.ones(25)
        mock_dm.fit.return_value = mock_dm_results
        
        fit_params = {'max_depth': 5, 'n_estimators': 100}
        
        model = DoublyRobustElasticityEstimatorModel(endog, exog)
        result = model.fit(n_folds=2, random_state=42, fit_params=fit_params)
        
        # Check NPModel was initialized with correct arguments
        for call in mock_npm_class.call_args_list:
            call_kwargs = call[1]
            assert 'variable_types' in call_kwargs
            assert 'params' in call_kwargs
            assert call_kwargs['params'] == fit_params
    
    def test_fit_without_mocks_errors_appropriately(self):
        """Test that fit raises appropriate error when models aren't available."""
        # Setup data
        np.random.seed(42)
        n = 50
        endog = np.exp(np.random.randn(n))
        exog = np.random.randn(n, 1)
        
        model = DoublyRobustElasticityEstimatorModel(endog, exog)
        
        # Should raise NameError since DensityModel isn't imported/defined
        with pytest.raises(NameError):
            model.fit(n_folds=2)
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NPModel')
    def test_fold_results_storage(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test that Results objects are stored for each fold."""
        # Setup data
        np.random.seed(42)
        n = 100
        endog = np.exp(np.random.randn(n))
        exog = np.random.randn(n, 1)
        n_folds = 5
        
        # Create mock models and unique results for each fold
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Create unique results for each fold to track them
        npm_results_list = []
        dm_results_list = []
        
        for i in range(n_folds):
            npm_result = Mock()
            npm_result.predict.return_value = np.ones(20)  # 100/5 for 5 folds
            npm_result.predict_semi_elasticity.return_value = 0.1 * np.ones(20)
            npm_result.fold_id = i  # Add identifier for testing
            npm_results_list.append(npm_result)
            
            dm_result = Mock()
            dm_result.predict.return_value = 0.5 * np.ones(20)
            dm_result.predict_semi_elasticity.return_value = 0.05 * np.ones(20)
            dm_result.fold_id = i  # Add identifier for testing
            dm_results_list.append(dm_result)
        
        # Set up side effects to return different results for each fold
        mock_npm.fit.side_effect = npm_results_list
        mock_dm.fit.side_effect = dm_results_list
        
        model = DoublyRobustElasticityEstimatorModel(endog, exog)
        result = model.fit(n_folds=n_folds, random_state=42)
        
        # Check that all Results objects were stored
        call_args = mock_results_class.call_args[1]
        fold_diagnostics = call_args['fold_diagnostics']
        
        assert len(fold_diagnostics['m_results']) == n_folds
        assert len(fold_diagnostics['f_results']) == n_folds
        
        # Verify each fold's results are stored correctly
        for i in range(n_folds):
            assert fold_diagnostics['m_results'][i].fold_id == i
            assert fold_diagnostics['f_results'][i].fold_id == i
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NPModel')
    def test_alpha_theta_arrays_stored(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test that alpha and theta arrays are stored for each fold."""
        # Setup data
        np.random.seed(42)
        n = 60
        endog = np.exp(np.random.randn(n))
        exog = np.random.randn(n, 2)
        n_folds = 3
        
        # Create mock models and results
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Create mock Results
        mock_npm_results = Mock()
        mock_npm_results.predict.return_value = np.ones(20)  # 60/3
        mock_npm_results.predict_semi_elasticity.return_value = 0.1 * np.ones(20)
        mock_npm.fit.return_value = mock_npm_results
        
        mock_dm_results = Mock()
        mock_dm_results.predict.return_value = 0.5 * np.ones(20)
        mock_dm_results.predict_semi_elasticity.return_value = 0.05 * np.ones(20)
        mock_dm.fit.return_value = mock_dm_results
        
        model = DoublyRobustElasticityEstimatorModel(endog, exog)
        result = model.fit(n_folds=n_folds, random_state=42)
        
        # Check that alpha and theta arrays were stored
        call_args = mock_results_class.call_args[1]
        fold_diagnostics = call_args['fold_diagnostics']
        
        assert len(fold_diagnostics['alpha_x']) == n_folds
        assert len(fold_diagnostics['theta_x']) == n_folds
        
        # Each fold should have arrays with shape (n_test, n_interest_vars)
        for alpha, theta in zip(fold_diagnostics['alpha_x'], fold_diagnostics['theta_x']):
            assert isinstance(alpha, np.ndarray)
            assert isinstance(theta, np.ndarray)
            # Should have 2 columns (one per variable)
            assert alpha.shape[1] == 2 if alpha.size > 0 else True
            assert theta.shape[1] == 2 if theta.size > 0 else True
