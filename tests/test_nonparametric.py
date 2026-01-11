import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch

from loglinearcorrection.nonparametric import (
    NPModel, NPModelResults, NNModel, NNModelResults,
    NNModelScore, NNModelScoreResults, NNModelNuisance
)


class TestNPModelResults:
    """Tests for NPModelResults class."""
    
    def test_initialization(self):
        """Test basic initialization of NPModelResults."""
        mock_model = Mock(spec=NPModel)
        mock_model.model = Mock()
        mock_model.variable_types = {'continuous': [0, 1], 'binary': [2]}
        
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])
        
        results = NPModelResults(mock_model, x, y)
        
        assert results.model == mock_model.model
        assert results.parent_model == mock_model
        assert np.array_equal(results.x, x)
        assert np.array_equal(results.y, y)
    
    def test_predict_semi_elasticity_continuous(self):
        """Test predict_semi_elasticity for continuous variables."""
        mock_model = Mock(spec=NPModel)
        mock_model.model = Mock()
        mock_model.variable_types = {0: 'continuous', 1: 'continuous', 2: 'binary'}
        
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])
        
        results = NPModelResults(mock_model, x, y)
        
        # Mock predict and derivative methods
        results.predict = Mock(return_value=np.array([10.0, 20.0]))
        results.derivative = Mock(return_value=np.array([2.0, 4.0]))
        
        # Test for continuous variable
        semi_elast = results.predict_semi_elasticity(x, var_index=0)
        
        # Should compute derivative / prediction = [2/10, 4/20] = [0.2, 0.2]
        np.testing.assert_array_almost_equal(semi_elast, [0.2, 0.2])
        results.predict.assert_called_once_with(x)
        results.derivative.assert_called_once_with(x, 0)
    
    def test_predict_semi_elasticity_binary_raises_error(self):
        """Test that predict_semi_elasticity raises error for binary variables."""
        mock_model = Mock(spec=NPModel)
        mock_model.model = Mock()
        mock_model.variable_types = {0: 'continuous', 1: 'binary'}
        
        x = np.array([[1, 0], [2, 1]])
        y = np.array([1, 2])
        
        results = NPModelResults(mock_model, x, y)
        results.predict = Mock(return_value=np.array([10.0, 20.0]))
        results.derivative = Mock(return_value=np.array([2.0, 4.0]))
        
        # Should raise ValueError for binary variable
        with pytest.raises(ValueError, match="predict_semi_elasticity called on binary variable"):
            results.predict_semi_elasticity(x, var_index=1)
    
    def test_predict_semi_elasticity_ordinal_raises_error(self):
        """Test that predict_semi_elasticity raises error for ordinal variables."""
        mock_model = Mock(spec=NPModel)
        mock_model.model = Mock()
        mock_model.variable_types = {0: 'continuous', 1: 'ordinal'}
        
        x = np.array([[1, 2], [2, 3]])
        y = np.array([1, 2])
        
        results = NPModelResults(mock_model, x, y)
        results.predict = Mock(return_value=np.array([10.0, 20.0]))
        results.derivative = Mock(return_value=np.array([2.0, 4.0]))
        
        # Should raise ValueError for ordinal variable
        with pytest.raises(ValueError, match="predict_semi_elasticity called on ordinal variable"):
            results.predict_semi_elasticity(x, var_index=1)
    
    def test_predict_semi_elasticity_zero_division(self):
        """Test that predict_semi_elasticity raises ZeroDivisionError when m(x) = 0."""
        mock_model = Mock(spec=NPModel)
        mock_model.model = Mock()
        mock_model.variable_types = {0: 'continuous'}
        
        x = np.array([[1], [2]])
        y = np.array([1, 2])
        
        results = NPModelResults(mock_model, x, y)
        
        # Mock predict to return zero for one observation
        results.predict = Mock(return_value=np.array([0.0, 20.0]))
        results.derivative = Mock(return_value=np.array([2.0, 4.0]))
        
        # Should raise ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            results.predict_semi_elasticity(x, var_index=0)
    
    def test_predict_semi_elasticity_without_variable_types(self):
        """Test predict_semi_elasticity when variable_types is not available."""
        mock_model = Mock(spec=NPModel)
        mock_model.model = Mock()
        # Don't set variable_types
        
        x = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        
        results = NPModelResults(mock_model, x, y)
        
        # Should still work without variable_types check
        results.predict = Mock(return_value=np.array([10.0, 20.0]))
        results.derivative = Mock(return_value=np.array([5.0, 10.0]))
        
        semi_elast = results.predict_semi_elasticity(x, var_index=0)
        
        # Should compute derivative / prediction = [5/10, 10/20] = [0.5, 0.5]
        np.testing.assert_array_almost_equal(semi_elast, [0.5, 0.5])
    
    def test_predict_semi_elasticity_default_continuous(self):
        """Test that unlisted variables default to continuous."""
        mock_model = Mock(spec=NPModel)
        mock_model.model = Mock()
        mock_model.variable_types = {0: 'binary'}  # Only specify var 0
        
        x = np.array([[0, 2.5], [1, 3.5]])
        y = np.array([1, 2])
        
        results = NPModelResults(mock_model, x, y)
        results.predict = Mock(return_value=np.array([10.0, 20.0]))
        results.derivative = Mock(return_value=np.array([3.0, 6.0]))
        
        # Variable 1 not in variable_types, should default to continuous
        semi_elast = results.predict_semi_elasticity(x, var_index=1)
        
        np.testing.assert_array_almost_equal(semi_elast, [0.3, 0.3])


class TestNNModel:
    """Tests for NNModel class."""
    
    def test_initialization_requires_pytorch(self):
        """Test that NNModel initialization requires PyTorch."""
        variable_types = {0: 'continuous'}
        params = {
            'input_size': 10,
            'hidden_layers': [32, 16],
            'output_size': 1
        }
        
        # This will fail if PyTorch is not installed
        try:
            model = NNModel(variable_types, params)
            assert hasattr(model, 'device')
            assert hasattr(model, 'model')
        except ImportError:
            pytest.skip("PyTorch not installed")
    
    def test_parse_params_validation(self):
        """Test parameter validation in _parse_params."""
        variable_types = {0: 'continuous'}
        
        # Missing required parameters
        with pytest.raises(ValueError, match="Missing required parameter"):
            model = NNModel(variable_types, {})
            model._parse_params()
        
        # Invalid input_size
        params = {
            'input_size': -1,
            'hidden_layers': [32],
            'output_size': 1
        }
        with pytest.raises(ValueError, match="input_size must be a positive integer"):
            model = NNModel(variable_types, params)
            model._parse_params()
        
        # Invalid hidden_layers
        params = {
            'input_size': 10,
            'hidden_layers': 'invalid',
            'output_size': 1
        }
        with pytest.raises(ValueError, match="hidden_layers must be a list"):
            model = NNModel(variable_types, params)
            model._parse_params()
        
        # Invalid activation
        params = {
            'input_size': 10,
            'hidden_layers': [32],
            'output_size': 1,
            'activation': 'invalid'
        }
        with pytest.raises(ValueError, match="Unsupported activation function"):
            model = NNModel(variable_types, params)
            model._parse_params()
    
    def test_parse_params_defaults(self):
        """Test that _parse_params fills in defaults correctly."""
        variable_types = {0: 'continuous'}
        params = {
            'input_size': 10,
            'hidden_layers': [32, 16],
            'output_size': 1
        }
        
        try:
            model = NNModel(variable_types, params)
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        parsed = model._parse_params()
        
        # Check defaults were filled
        assert parsed['activation'] == 'relu'
        assert parsed['output_activation'] == 'identity'
        assert parsed['learning_rate'] == 1e-3
        assert parsed['dropout'] == 0.0
        assert parsed['bias'] is True
        assert parsed['weight_init'] == 'default'
        
        # Check required params are present
        assert parsed['input_size'] == 10
        assert parsed['hidden_layers'] == [32, 16]
        assert parsed['output_size'] == 1


class TestNNModelScore:
    """Tests for NNModelScore class."""
    
    def test_initialization_requires_matching_sizes(self):
        """Test that NNModelScore requires input_size == output_size."""
        variable_types = {0: 'continuous'}
        
        # Mismatched sizes
        params = {
            'input_size': 10,
            'hidden_layers': [32],
            'output_size': 5
        }
        
        with pytest.raises(ValueError, match="input_size must equal output_size"):
            NNModelScore(variable_types, params)
        
        # Matching sizes should work
        params['output_size'] = 10
        try:
            model = NNModelScore(variable_types, params)
            assert model.params['input_size'] == model.params['output_size']
        except ImportError:
            pytest.skip("PyTorch not installed")
    
    @patch('loglinearcorrection.nonparametric.torch')
    @patch('loglinearcorrection.nonparametric.DataLoader')
    @patch('loglinearcorrection.nonparametric.SlicedScoreMatchingLoss')
    def test_fit_score_matching(self, mock_loss_class, mock_dataloader, mock_torch):
        """Test fit method for score matching."""
        # Skip if can't import
        try:
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        variable_types = {0: 'continuous'}
        params = {
            'input_size': 3,
            'hidden_layers': [8],
            'output_size': 3,
            'epochs': 2,
            'batch_size': 32
        }
        
        # Create model with mocked torch
        model = NNModelScore(variable_types, params)
        
        # Setup training data
        x = np.random.randn(100, 3)
        y = None  # y is unused for score matching
        
        # Mock the loss and optimizer
        mock_loss = Mock()
        mock_loss_class.return_value = mock_loss
        mock_loss.return_value = Mock(backward=Mock(), detach=Mock(return_value=0.5))
        
        # Call fit
        with patch.object(model.model, 'train') as mock_train:
            with patch.object(model.model, 'to') as mock_to:
                with patch('torch.optim.Adam') as mock_adam:
                    mock_opt = Mock()
                    mock_adam.return_value = mock_opt
                    
                    results = model.fit(x, y)
        
        # Check that results are NNModelScoreResults
        assert isinstance(results, NNModelScoreResults)
        assert np.array_equal(results.x, x)
        assert results.y is y


class TestNNModelResults:
    """Tests for NNModelResults class."""
    
    def test_inheritance_from_npm_results(self):
        """Test that NNModelResults inherits from NPModelResults."""
        assert issubclass(NNModelResults, NPModelResults)
    
    def test_initialization(self):
        """Test NNModelResults initialization."""
        mock_nn_model = Mock(spec=NNModel)
        mock_nn_model.model = Mock()
        mock_nn_model.variable_types = {0: 'continuous'}
        
        x = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        
        results = NNModelResults(mock_nn_model, x, y)
        
        assert results.model == mock_nn_model.model
        assert results.parent_model == mock_nn_model
        assert np.array_equal(results.x, x)
        assert np.array_equal(results.y, y)
    
    def test_predict_semi_elasticity_inherited(self):
        """Test that predict_semi_elasticity is inherited and works."""
        mock_nn_model = Mock(spec=NNModel)
        mock_nn_model.model = Mock()
        mock_nn_model.variable_types = {0: 'continuous', 1: 'binary'}
        
        x = np.array([[1, 0], [2, 1]])
        y = np.array([1, 2])
        
        results = NNModelResults(mock_nn_model, x, y)
        results.predict = Mock(return_value=np.array([10.0, 20.0]))
        results.derivative = Mock(return_value=np.array([5.0, 10.0]))
        
        # Should work for continuous variable
        semi_elast = results.predict_semi_elasticity(x, var_index=0)
        np.testing.assert_array_almost_equal(semi_elast, [0.5, 0.5])
        
        # Should raise error for binary variable
        with pytest.raises(ValueError, match="binary"):
            results.predict_semi_elasticity(x, var_index=1)


class TestIntegration:
    """Integration tests for the nonparametric module."""
    
    def test_npm_to_results_flow(self):
        """Test the flow from NPModel to NPModelResults."""
        variable_types = {0: 'continuous', 1: 'binary'}
        params = {'some_param': 'value'}
        
        model = NPModel(variable_types, params)
        assert model.variable_types == variable_types
        assert model.params == params
        
        # Mock the fit method to return results
        x = np.array([[1, 0], [2, 1]])
        y = np.array([3, 4])
        
        with patch.object(model, 'fit') as mock_fit:
            mock_results = NPModelResults(model, x, y)
            mock_fit.return_value = mock_results
            
            results = model.fit(x, y)
            
            assert isinstance(results, NPModelResults)
            assert results.parent_model == model
    
    def test_variable_type_propagation(self):
        """Test that variable types propagate from model to results."""
        variable_types = {0: 'continuous', 1: 'ordinal', 2: 'binary'}
        params = {}
        
        model = NPModel(variable_types, params)
        x = np.random.randn(10, 3)
        y = np.random.randn(10)
        
        results = NPModelResults(model, x, y)
        
        # Mock methods
        results.predict = Mock(return_value=np.ones(10))
        results.derivative = Mock(return_value=np.ones(10))
        
        # Should work for continuous
        _ = results.predict_semi_elasticity(x, var_index=0)
        
        # Should fail for ordinal
        with pytest.raises(ValueError, match="ordinal"):
            results.predict_semi_elasticity(x, var_index=1)
        
        # Should fail for binary
        with pytest.raises(ValueError, match="binary"):
            results.predict_semi_elasticity(x, var_index=2)
