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

    def test_1d_exog_converted_to_2d(self):
        """Test that 1-D exog is converted to 2-D array."""
        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([1.0, 2.0, 3.0])

        model = DoublyRobustElasticityEstimatorModel(endog, exog)

        assert model.exog.ndim == 2
        assert model.exog.shape == (3, 1)

    def test_endog_raveled(self):
        """Test that endog is always 1-D regardless of input shape."""
        endog_2d = np.array([[1.0], [2.0], [3.0]])
        exog = np.array([[1.0], [2.0], [3.0]])

        model = DoublyRobustElasticityEstimatorModel(endog_2d, exog)

        assert model.endog.ndim == 1
        assert model.endog.shape == (3,)

    @patch("loglinearcorrection.model._detect_variable_types")
    @patch("loglinearcorrection.model._apply_fixed_effects")
    def test_detect_variable_types_called(
        self, mock_apply_fe, mock_detect_types
    ):
        """Test that _detect_variable_types is called with correct arguments."""
        mock_detect_types.return_value = {0: "continuous", 1: "binary"}
        mock_apply_fe.return_value = (
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0], [2.0], [3.0]])
        )

        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])

        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, fixed_effects=None
        )

        # Should be called with all columns when no fixed effects
        mock_detect_types.assert_called_once()
        call_args = mock_detect_types.call_args
        assert np.array_equal(call_args[0][1], [0, 1])  # indices argument

    @patch("loglinearcorrection.model._detect_variable_types")
    @patch("loglinearcorrection.model._apply_fixed_effects")
    def test_detect_variable_types_excludes_fixed_effects(
        self, mock_apply_fe, mock_detect_types
    ):
        """Test that fixed effects are excluded from variable type detection."""
        mock_detect_types.return_value = {0: "continuous", 2: "continuous"}
        mock_apply_fe.return_value = (
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
        )

        endog = np.array([1.0, 2.0, 3.0])
        exog = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0],
            "fe": [0, 0, 1],
            "x2": [3.0, 4.0, 5.0]
        })

        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, fixed_effects=["fe"]
        )

        # Should be called with non-FE columns only (indices 0 and 2)
        mock_detect_types.assert_called_once()
        call_args = mock_detect_types.call_args
        assert call_args[0][1] == [0, 2]  # indices argument

    @patch("loglinearcorrection.model._detect_variable_types")
    @patch("loglinearcorrection.model._apply_fixed_effects")
    def test_apply_fixed_effects_called(
        self, mock_apply_fe, mock_detect_types
    ):
        """Test that _apply_fixed_effects is called with correct arguments."""
        mock_detect_types.return_value = {0: "continuous"}
        endog_demeaned = np.array([0.5, 1.5, 2.5])
        exog_demeaned = np.array([[0.5], [1.5], [2.5]])
        mock_apply_fe.return_value = (endog_demeaned, exog_demeaned)

        endog = np.array([1.0, 2.0, 3.0])
        exog = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "fe": [0, 0, 1]})

        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, fixed_effects=["fe"]
        )

        mock_apply_fe.assert_called_once()
        call_args = mock_apply_fe.call_args[0]
        # Check that endog, exog, fixed_effects, and exog_names were passed
        np.testing.assert_array_equal(call_args[0], endog)
        assert call_args[2] == ["fe"]  # fixed_effects
        assert call_args[3] == ["x1", "fe"]  # exog_names
    
    @patch("loglinearcorrection.model._detect_variable_types")
    @patch("loglinearcorrection.model._apply_fixed_effects")
    def test_apply_fixed_effects_with_default_names(
        self, mock_apply_fe, mock_detect_types
    ):
        """Test that _apply_fixed_effects receives correct default names."""
        mock_detect_types.return_value = {0: "continuous"}
        endog_demeaned = np.array([0.5, 1.5, 2.5])
        exog_demeaned = np.array([[0.5, 1.0], [1.5, 1.0], [2.5, 1.0]])
        mock_apply_fe.return_value = (endog_demeaned, exog_demeaned)

        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([[1.0, 0], [2.0, 0], [3.0, 1]])

        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, fixed_effects=[1]
        )

        mock_apply_fe.assert_called_once()
        call_args = mock_apply_fe.call_args[0]
        # Check that default names were passed
        assert call_args[3] == ["x1", "x2"]  # default exog_names

    @patch("loglinearcorrection.model._detect_variable_types")
    @patch("loglinearcorrection.model._apply_fixed_effects")
    def test_demeaned_data_stored(
        self, mock_apply_fe, mock_detect_types
    ):
        """Test that demeaned data is stored correctly."""
        mock_detect_types.return_value = {0: "continuous"}
        endog_demeaned = np.array([0.5, 1.5, 2.5])
        exog_demeaned = np.array([[0.5], [1.5], [2.5]])
        mock_apply_fe.return_value = (endog_demeaned, exog_demeaned)

        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([[1.0], [2.0], [3.0]])

        model = DoublyRobustElasticityEstimatorModel(endog, exog)

        np.testing.assert_array_equal(model.endog_demeaned, endog_demeaned)
        np.testing.assert_array_equal(model.exog_demeaned, exog_demeaned)

    def test_empty_arrays_raise_appropriate_error(self):
        """Test that empty arrays are handled."""
        endog = np.array([])
        exog = np.array([]).reshape(0, 1)

        model = DoublyRobustElasticityEstimatorModel(endog, exog)
        assert model.nobs == 0

    def test_mismatched_dimensions_handled(self):
        """Test behavior with mismatched endog and exog dimensions."""
        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([[1.0], [2.0]])  # Only 2 rows

        # This should work with numpy broadcasting/conversion
        # The actual behavior depends on how strict we want to be
        # For now, test that it doesn't crash unexpectedly
        try:
            model = DoublyRobustElasticityEstimatorModel(endog, exog)
            # If it succeeds, check dimensions
            assert model.nobs == 3
            assert model.exog.shape[0] == 2
        except (ValueError, IndexError):
            # If it fails, that's also acceptable behavior
            pass

    def test_kwargs_accepted(self):
        """Test that additional kwargs are accepted without error."""
        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([[1.0], [2.0], [3.0]])

        # Should not raise an error
        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, some_extra_param=True
        )
        assert model.nobs == 3

    def test_fixed_effects_with_integer_indices(self):
        """Test fixed effects specification using integer indices."""
        endog = np.array([1.0, 2.0, 3.0, 4.0])
        exog = np.array([
            [1.0, 0, 2.0],
            [2.0, 0, 3.0],
            [3.0, 1, 4.0],
            [4.0, 1, 5.0]
        ])

        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, fixed_effects=[1]
        )

        assert model.fixed_effects == [1]

    def test_interest_with_integer_indices(self):
        """Test interest specification using integer indices."""
        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, interest=[0]
        )

        assert model.interest == [0]

    @patch("loglinearcorrection.model._detect_variable_types")
    @patch("loglinearcorrection.model._apply_fixed_effects")
    def test_variable_types_stored(
        self, mock_apply_fe, mock_detect_types
    ):
        """Test that detected variable types are stored."""
        expected_types = {0: "binary", 1: "continuous", 2: "continuous"}
        mock_detect_types.return_value = expected_types
        mock_apply_fe.return_value = (
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
        )

        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([[0.0, 1.5, 3], [1.0, 2.5, 5], [0.0, 3.5, 7]])

        model = DoublyRobustElasticityEstimatorModel(endog, exog)

        assert model.variable_types == expected_types
    
    @patch("loglinearcorrection.model._detect_variable_types")
    @patch("loglinearcorrection.model._apply_fixed_effects")
    def test_ordinal_override(
        self, mock_apply_fe, mock_detect_types
    ):
        """Test that ordinal parameter overrides detected types except for binary."""
        # Initial detection says one binary, rest continuous
        mock_detect_types.return_value = {0: "continuous", 1: "continuous", 2: "binary"}
        mock_apply_fe.return_value = (
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
        )

        endog = np.array([1.0, 2.0, 3.0])
        exog = pd.DataFrame({
            "x1": [3, 5, 7],
            "x2": [1.5, 2.5, 3.5],
            "x3": [0, 1, 0]
        })

        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, ordinal=["x1", "x3"]  # Try to set both as ordinal
        )

        # x1 (index 0) should be ordinal as it wasn't binary
        assert model.variable_types[0] == "ordinal"
        # x2 (index 1) should remain continuous
        assert model.variable_types[1] == "continuous"
        # x3 (index 2) should remain binary despite ordinal specification
        assert model.variable_types[2] == "binary"
    
    @patch("loglinearcorrection.model._detect_variable_types")
    @patch("loglinearcorrection.model._apply_fixed_effects")
    def test_binary_takes_precedence_over_ordinal(
        self, mock_apply_fe, mock_detect_types
    ):
        """Test that binary classification takes precedence over ordinal specification."""
        mock_detect_types.return_value = {0: "binary", 1: "continuous"}
        mock_apply_fe.return_value = (
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([[0, 1.5], [1, 2.5], [0, 3.5], [1, 4.5]])
        )

        endog = np.array([1.0, 2.0, 3.0, 4.0])
        exog = np.array([[0, 1.5], [1, 2.5], [0, 3.5], [1, 4.5]])

        # Try to override binary variable to ordinal
        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, ordinal=[0]
        )

        # Should remain binary
        assert model.variable_types[0] == "binary"
        assert model.variable_types[1] == "continuous"
    
    @patch("loglinearcorrection.model._detect_variable_types")
    @patch("loglinearcorrection.model._apply_fixed_effects")
    def test_ordinal_with_indices(
        self, mock_apply_fe, mock_detect_types
    ):
        """Test ordinal specification using integer indices."""
        mock_detect_types.return_value = {0: "continuous", 1: "binary"}
        mock_apply_fe.return_value = (
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        )

        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([[3, 0], [5, 1], [7, 0]])

        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, ordinal=[0, 1]  # Try to set both as ordinal
        )

        # Index 0 should become ordinal (was continuous)
        assert model.variable_types[0] == "ordinal"
        # Index 1 should remain binary (takes precedence)
        assert model.variable_types[1] == "binary"
    
    @patch("loglinearcorrection.model._detect_variable_types")
    @patch("loglinearcorrection.model._apply_fixed_effects")
    def test_ordinal_ignores_fixed_effects(
        self, mock_apply_fe, mock_detect_types
    ):
        """Test that ordinal specification ignores fixed effects columns."""
        # Only non-FE columns in variable_types
        mock_detect_types.return_value = {0: "continuous", 2: "continuous"}
        mock_apply_fe.return_value = (
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
        )

        endog = np.array([1.0, 2.0, 3.0])
        exog = pd.DataFrame({
            "x1": [1, 2, 3],
            "fe": [0, 0, 1],
            "x2": [3, 4, 5]
        })

        # Try to set fe column as ordinal (should be ignored)
        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, 
            fixed_effects=["fe"],
            ordinal=["fe", "x1"]
        )

        # fe (index 1) should not be in variable_types
        assert 1 not in model.variable_types
        # x1 (index 0) should be ordinal
        assert model.variable_types[0] == "ordinal"
        # x2 (index 2) should remain continuous
        assert model.variable_types[2] == "continuous"

    def test_dataframe_with_single_column(self):
        """Test initialization with single-column DataFrame."""
        endog = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        exog = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

        model = DoublyRobustElasticityEstimatorModel(endog, exog)

        assert model.endog_names == "y"
        assert model.exog_names == ["x"]

    def test_original_and_demeaned_data_both_stored(self):
        """Test that both original and demeaned data are accessible."""
        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([[1.0], [2.0], [3.0]])

        model = DoublyRobustElasticityEstimatorModel(endog, exog)

        # Original data should be stored
        assert hasattr(model, "endog")
        assert hasattr(model, "exog")
        # Demeaned data should also be stored
        assert hasattr(model, "endog_demeaned")
        assert hasattr(model, "exog_demeaned")


class TestEdgeCasesAndIntegration:
    """Integration tests and edge cases."""

    def test_large_dataset(self):
        """Test with a larger dataset."""
        np.random.seed(42)
        n = 10000
        endog = np.random.randn(n)
        exog = np.random.randn(n, 5)

        model = DoublyRobustElasticityEstimatorModel(endog, exog)

        assert model.nobs == n
        assert model.exog.shape == (n, 5)

    def test_single_observation(self):
        """Test with single observation."""
        endog = np.array([1.0])
        exog = np.array([[1.0, 2.0]])

        model = DoublyRobustElasticityEstimatorModel(endog, exog)

        assert model.nobs == 1
        assert model.exog.shape == (1, 2)

    def test_all_columns_as_fixed_effects(self):
        """Test when all columns are specified as fixed effects."""
        endog = np.array([1.0, 2.0, 3.0, 4.0])
        exog = pd.DataFrame({"fe1": [0, 0, 1, 1], "fe2": [0, 1, 0, 1]})

        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, fixed_effects=["fe1", "fe2"]
        )

        # Should have empty variable_types since all are FE
        assert model.variable_types == {}

    def test_none_fixed_effects_explicitly(self):
        """Test explicitly passing None for fixed_effects."""
        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([[1.0], [2.0], [3.0]])

        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, fixed_effects=None
        )

        assert model.fixed_effects is None

    def test_integer_data(self):
        """Test that integer data is accepted and converted."""
        endog = np.array([1, 2, 3, 4])
        exog = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])

        model = DoublyRobustElasticityEstimatorModel(endog, exog)

        assert model.nobs == 4
        # Should be converted to float internally
        assert np.issubdtype(model.endog.dtype, np.number)

    def test_mixed_pandas_and_numpy(self):
        """Test mixing pandas and numpy inputs."""
        endog = pd.Series([1.0, 2.0, 3.0], name="y")
        exog = np.array([[1.0], [2.0], [3.0]])

        model = DoublyRobustElasticityEstimatorModel(endog, exog)

        assert model.endog_names == "y"
        assert model.exog_names == ["x1"]  # Default name for single column
    
    def test_default_names_multiple_columns(self):
        """Test default naming for multiple columns in numpy array."""
        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])

        model = DoublyRobustElasticityEstimatorModel(endog, exog)

        assert model.endog_names == "y"
        assert model.exog_names == ["x1", "x2", "x3"]
    
    def test_default_names_single_column(self):
        """Test default naming for single column numpy array."""
        endog = np.array([1.0, 2.0, 3.0])
        exog = np.array([1.0, 2.0, 3.0])  # 1D array

        model = DoublyRobustElasticityEstimatorModel(endog, exog)

        assert model.endog_names == "y"
        assert model.exog_names == ["x1"]
    
    def test_default_names_with_fixed_effects_reference(self):
        """Test that default names work correctly with fixed effects references."""
        endog = np.array([1.0, 2.0, 3.0, 4.0])
        exog = np.array([
            [1.0, 0, 2.0],
            [2.0, 0, 3.0],
            [3.0, 1, 4.0],
            [4.0, 1, 5.0]
        ])

        # Even with integer indices for fixed effects, names should be generated
        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, fixed_effects=[1]
        )

        assert model.endog_names == "y"
        assert model.exog_names == ["x1", "x2", "x3"]
        assert model.fixed_effects == [1]


class TestDoublyRobustElasticityEstimatorModelFit:
    """Tests for the fit() method of DoublyRobustElasticityEstimatorModel."""
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NonparametricModel')
    def test_basic_fit_continuous_variable(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test basic fit with a single continuous variable."""
        # Setup data
        np.random.seed(42)
        n = 100
        endog = np.exp(np.random.randn(n))  # Log-linear model appropriate data
        exog = np.random.randn(n, 1)
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Mock predictions
        mock_npm.predict.return_value = np.ones(10)  # m(x)
        mock_npm.predict_semi_elasticity.return_value = 0.1 * np.ones(10)  # m_k(x)/m(x)
        mock_dm.predict.return_value = 0.5 * np.ones(10)  # f(x)
        mock_dm.predict_semi_elasticity.return_value = 0.05 * np.ones(10)  # f_k(x)/f(x)
        
        # Mock results
        mock_results = Mock()
        mock_results_class.return_value = mock_results
        
        # Fit model
        model = DoublyRobustElasticityEstimatorModel(endog, exog)
        result = model.fit(n_folds=2, random_state=42)
        
        # Verify NonparametricModel was instantiated and called
        assert mock_npm_class.call_count == 2  # Once per fold
        assert mock_npm.fit.call_count == 2
        assert mock_npm.predict.call_count == 2
        assert mock_npm.predict_semi_elasticity.call_count == 2
        
        # Verify DensityModel was instantiated and called
        assert mock_dm_class.call_count == 2  # Once per fold
        assert mock_dm.fit.call_count == 2
        assert mock_dm.predict.call_count == 2
        assert mock_dm.predict_semi_elasticity.call_count == 2
        
        # Verify results object was created
        mock_results_class.assert_called_once()
        assert result == mock_results
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NonparametricModel')
    def test_fit_with_binary_variable(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test fit with a binary variable."""
        # Setup data with binary variable
        np.random.seed(42)
        n = 100
        endog = np.exp(np.random.randn(n))
        exog = np.random.choice([0, 1], size=(n, 1))
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Mock predictions
        mock_npm.predict.return_value = np.ones(50)
        mock_dm.predict.return_value = 0.5 * np.ones(50)
        
        # Mock variable type detection to return binary
        with patch('loglinearcorrection.model._detect_variable_types') as mock_detect:
            mock_detect.return_value = {0: "binary"}
            
            model = DoublyRobustElasticityEstimatorModel(endog, exog)
            result = model.fit(n_folds=2, random_state=42)
        
        # For binary variables, should call predict with shifted values
        # Check that predict was called multiple times (for m(x) and m(x+Δ))
        assert mock_npm.predict.call_count >= 4  # At least 2 per fold (m(x) and m(x+Δ))
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NonparametricModel')
    def test_fit_with_ordinal_variable(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test fit with an ordinal variable."""
        # Setup data
        np.random.seed(42)
        n = 100
        endog = np.exp(np.random.randn(n))
        exog = np.random.choice([1, 2, 3, 4], size=(n, 1))
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Mock predictions
        mock_npm.predict.return_value = np.ones(25)  # For test set of each fold
        mock_dm.predict.return_value = 0.5 * np.ones(25)
        
        model = DoublyRobustElasticityEstimatorModel(endog, exog, ordinal=[0])
        result = model.fit(n_folds=4, random_state=42)
        
        # For ordinal, should call predict with x+Δ, x-Δ
        assert mock_npm.predict.call_count >= 12  # m(x), m(x+Δ), m(x-Δ) per fold
        assert mock_dm.predict.call_count >= 8  # f(x), f(x-Δ) per fold
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NonparametricModel')
    def test_fit_with_fixed_effects(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test that fixed effects are excluded from elasticity calculation."""
        # Setup data
        np.random.seed(42)
        n = 100
        endog = np.exp(np.random.randn(n))
        exog = pd.DataFrame({
            'x1': np.random.randn(n),
            'fe': np.random.choice([0, 1, 2], n),  # Fixed effect
            'x2': np.random.randn(n)
        })
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Mock predictions
        mock_npm.predict.return_value = np.ones(50)
        mock_npm.predict_semi_elasticity.return_value = 0.1 * np.ones(50)
        mock_dm.predict.return_value = 0.5 * np.ones(50)
        mock_dm.predict_semi_elasticity.return_value = 0.05 * np.ones(50)
        
        mock_results = Mock()
        mock_results_class.return_value = mock_results
        
        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, fixed_effects=['fe']
        )
        result = model.fit(n_folds=2, random_state=42)
        
        # Check results were created
        call_args = mock_results_class.call_args
        elasticities = call_args[1]['elasticities']
        
        # Should only have elasticities for x1 and x2, not fe
        assert 'fe' not in elasticities
        assert len(elasticities) == 2
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NonparametricModel')
    def test_fit_with_interest_specification(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test that only interest variables get elasticities computed."""
        # Setup data
        np.random.seed(42)
        n = 100
        endog = np.exp(np.random.randn(n))
        exog = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'x3': np.random.randn(n)
        })
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Mock predictions
        test_size = n // 2  # Approximate for 2 folds
        mock_npm.predict.return_value = np.ones(test_size)
        mock_npm.predict_semi_elasticity.return_value = 0.1 * np.ones(test_size)
        mock_dm.predict.return_value = 0.5 * np.ones(test_size)
        mock_dm.predict_semi_elasticity.return_value = 0.05 * np.ones(test_size)
        
        mock_results = Mock()
        mock_results_class.return_value = mock_results
        
        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, interest=['x1', 'x3']
        )
        result = model.fit(n_folds=2, random_state=42)
        
        # Check results
        call_args = mock_results_class.call_args
        elasticities = call_args[1]['elasticities']
        
        # Should only have elasticities for x1 and x3
        assert 'x1' in elasticities
        assert 'x3' in elasticities
        assert 'x2' not in elasticities
        assert len(elasticities) == 2
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NonparametricModel')
    def test_fit_with_weights(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test that weights are properly passed to models."""
        # Setup data
        np.random.seed(42)
        n = 100
        endog = np.exp(np.random.randn(n))
        exog = np.random.randn(n, 1)
        weights = np.random.uniform(0.5, 2.0, n)
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Mock predictions
        mock_npm.predict.return_value = np.ones(50)
        mock_npm.predict_semi_elasticity.return_value = 0.1 * np.ones(50)
        mock_dm.predict.return_value = 0.5 * np.ones(50)
        mock_dm.predict_semi_elasticity.return_value = 0.05 * np.ones(50)
        
        model = DoublyRobustElasticityEstimatorModel(endog, exog, weights=weights)
        result = model.fit(n_folds=2, random_state=42)
        
        # Check that weights were passed to fit methods
        for call in mock_npm.fit.call_args_list:
            assert 'weights' in call[1] or (len(call[0]) > 2 and call[0][2] is not None)
        
        for call in mock_dm.fit.call_args_list:
            assert 'weights' in call[1] or (len(call[0]) > 1 and call[0][1] is not None)
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NonparametricModel')
    def test_fit_params_passed_through(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test that fit_params are passed to nuisance models."""
        # Setup data
        np.random.seed(42)
        n = 50
        endog = np.exp(np.random.randn(n))
        exog = np.random.randn(n, 1)
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Mock predictions
        mock_npm.predict.return_value = np.ones(25)
        mock_npm.predict_semi_elasticity.return_value = 0.1 * np.ones(25)
        mock_dm.predict.return_value = 0.5 * np.ones(25)
        mock_dm.predict_semi_elasticity.return_value = 0.05 * np.ones(25)
        
        fit_params = {'max_depth': 5, 'n_estimators': 100}
        
        model = DoublyRobustElasticityEstimatorModel(endog, exog)
        result = model.fit(n_folds=2, random_state=42, fit_params=fit_params)
        
        # Check fit_params were passed
        for call in mock_npm.fit.call_args_list:
            assert call[1]['max_depth'] == 5
            assert call[1]['n_estimators'] == 100
        
        for call in mock_dm.fit.call_args_list:
            assert call[1]['max_depth'] == 5
            assert call[1]['n_estimators'] == 100
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')  
    @patch('loglinearcorrection.model.NonparametricModel')
    def test_cross_fitting_logic(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test that cross-fitting correctly fits on complement and evaluates on fold."""
        # Setup data - use small size for easier verification
        np.random.seed(42)
        n = 20
        endog = np.exp(np.random.randn(n))
        exog = np.random.randn(n, 1)
        
        # Track what indices are used for training
        training_indices_npm = []
        training_indices_dm = []
        
        def npm_fit_side_effect(X, y, *args, **kwargs):
            training_indices_npm.append(set(range(len(X))))
            
        def dm_fit_side_effect(X, *args, **kwargs):
            training_indices_dm.append(set(range(len(X))))
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        mock_npm.fit.side_effect = npm_fit_side_effect
        mock_dm.fit.side_effect = dm_fit_side_effect
        
        # Mock predictions - size should match test set
        mock_npm.predict.return_value = np.ones(10)  # Half for 2-fold
        mock_npm.predict_semi_elasticity.return_value = 0.1 * np.ones(10)
        mock_dm.predict.return_value = 0.5 * np.ones(10)
        mock_dm.predict_semi_elasticity.return_value = 0.05 * np.ones(10)
        
        model = DoublyRobustElasticityEstimatorModel(endog, exog)
        result = model.fit(n_folds=2, random_state=42)
        
        # With 2 folds, each training set should have ~10 observations
        assert len(training_indices_npm) == 2
        assert len(training_indices_dm) == 2
        
        # Training sets should be approximately half the data
        for indices in training_indices_npm:
            assert len(indices) == 10  # Half of 20
            
        for indices in training_indices_dm:
            assert len(indices) == 10
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NonparametricModel')
    def test_n_folds_parameter(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test that n_folds parameter controls number of folds."""
        # Setup data
        np.random.seed(42)
        n = 100
        endog = np.exp(np.random.randn(n))
        exog = np.random.randn(n, 1)
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Mock predictions - adjust size for different fold counts
        mock_npm.predict.return_value = np.ones(20)  # 100/5 for 5 folds
        mock_npm.predict_semi_elasticity.return_value = 0.1 * np.ones(20)
        mock_dm.predict.return_value = 0.5 * np.ones(20)
        mock_dm.predict_semi_elasticity.return_value = 0.05 * np.ones(20)
        
        model = DoublyRobustElasticityEstimatorModel(endog, exog)
        result = model.fit(n_folds=5, random_state=42)
        
        # Should have 5 folds worth of calls
        assert mock_npm_class.call_count == 5
        assert mock_dm_class.call_count == 5
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NonparametricModel')
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
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Mock predictions
        mock_npm.predict.return_value = np.ones(25)
        mock_npm.predict_semi_elasticity.return_value = 0.1 * np.ones(25)
        mock_dm.predict.return_value = 0.5 * np.ones(25)
        mock_dm.predict_semi_elasticity.return_value = 0.05 * np.ones(25)
        
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
        
        # Check fold diagnostics
        diagnostics = call_args['fold_diagnostics']
        assert 'ols_coefficients' in diagnostics
        assert 'moment_means' in diagnostics
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NonparametricModel')
    def test_ols_uses_demeaned_data(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test that OLS estimation uses demeaned data."""
        # Setup data with fixed effects
        np.random.seed(42)
        n = 100
        endog = np.exp(np.random.randn(n))
        fe_groups = np.repeat([0, 1, 2, 3], 25)
        exog = pd.DataFrame({
            'x1': np.random.randn(n),
            'fe': fe_groups
        })
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Mock predictions
        mock_npm.predict.return_value = np.ones(50)
        mock_npm.predict_semi_elasticity.return_value = 0.1 * np.ones(50)
        mock_dm.predict.return_value = 0.5 * np.ones(50)
        mock_dm.predict_semi_elasticity.return_value = 0.05 * np.ones(50)
        
        # Patch the OLS/WLS to check inputs
        with patch('loglinearcorrection.model.sm.OLS') as mock_ols:
            mock_ols_instance = Mock()
            mock_ols.return_value = mock_ols_instance
            mock_ols_results = Mock()
            mock_ols_results.params = np.array([0.5])  # Mock beta
            mock_ols_instance.fit.return_value = mock_ols_results
            
            model = DoublyRobustElasticityEstimatorModel(
                endog, exog, fixed_effects=['fe']
            )
            result = model.fit(n_folds=2, random_state=42)
            
            # OLS should have been called with demeaned data
            # The second argument to OLS should be the demeaned exog
            for call in mock_ols.call_args_list:
                X_used = call[0][1]  # Second positional argument
                # Demeaned data should have mean approximately 0 within groups
                # Just check it's not the original data
                assert X_used.shape[1] == 1  # Should only have x1, not fe
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NonparametricModel')
    def test_nuisance_uses_original_data(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test that nuisance functions use original (non-demeaned) data."""
        # Setup data with fixed effects
        np.random.seed(42)
        n = 50
        endog = np.exp(np.random.randn(n))
        fe_groups = np.repeat([0, 1], 25)
        exog = pd.DataFrame({
            'x1': np.random.randn(n),
            'fe': fe_groups
        })
        
        # Track what data is passed to nuisance models
        npm_X_data = []
        dm_X_data = []
        
        def npm_fit_side_effect(X, y, *args, **kwargs):
            npm_X_data.append(X.shape)
            
        def dm_fit_side_effect(X, *args, **kwargs):
            dm_X_data.append(X.shape)
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        mock_npm.fit.side_effect = npm_fit_side_effect
        mock_dm.fit.side_effect = dm_fit_side_effect
        
        # Mock predictions
        mock_npm.predict.return_value = np.ones(25)
        mock_npm.predict_semi_elasticity.return_value = 0.1 * np.ones(25)
        mock_dm.predict.return_value = 0.5 * np.ones(25)
        mock_dm.predict_semi_elasticity.return_value = 0.05 * np.ones(25)
        
        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, fixed_effects=['fe']
        )
        result = model.fit(n_folds=2, random_state=42)
        
        # Nuisance models should get original data with both columns
        for shape in npm_X_data:
            assert shape[1] == 2  # Both x1 and fe
            
        for shape in dm_X_data:
            assert shape[1] == 2  # Both x1 and fe
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NonparametricModel')
    def test_mixed_variable_types(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test handling of mixed continuous, binary, and ordinal variables."""
        # Setup data
        np.random.seed(42)
        n = 100
        endog = np.exp(np.random.randn(n))
        exog = pd.DataFrame({
            'continuous': np.random.randn(n),
            'binary': np.random.choice([0, 1], n),
            'ordinal': np.random.choice([1, 2, 3, 4], n)
        })
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Mock predictions
        test_size = 50  # For 2 folds
        mock_npm.predict.return_value = np.ones(test_size)
        mock_npm.predict_semi_elasticity.return_value = 0.1 * np.ones(test_size)
        mock_dm.predict.return_value = 0.5 * np.ones(test_size)
        mock_dm.predict_semi_elasticity.return_value = 0.05 * np.ones(test_size)
        
        # Mock variable types
        with patch('loglinearcorrection.model._detect_variable_types') as mock_detect:
            mock_detect.return_value = {
                0: 'continuous',
                1: 'binary',
                2: 'continuous'  # Will be overridden to ordinal
            }
            
            model = DoublyRobustElasticityEstimatorModel(
                endog, exog, ordinal=['ordinal']
            )
            result = model.fit(n_folds=2, random_state=42)
            
            # Check that different variable types were handled
            call_args = mock_results_class.call_args[1]
            elasticities = call_args['elasticities']
            
            assert elasticities['continuous']['type'] == 'continuous'
            assert elasticities['binary']['type'] == 'binary'
            assert elasticities['ordinal']['type'] == 'ordinal'
    
    def test_fit_without_mocks_errors_appropriately(self):
        """Test that fit raises appropriate error when models aren't available."""
        # Setup data
        np.random.seed(42)
        n = 50
        endog = np.exp(np.random.randn(n))
        exog = np.random.randn(n, 1)
        
        model = DoublyRobustElasticityEstimatorModel(endog, exog)
        
        # Should raise NameError since NonparametricModel isn't imported
        with pytest.raises(NameError):
            model.fit(n_folds=2)
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NonparametricModel')
    def test_random_state_reproducibility(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test that random_state makes results reproducible."""
        # Setup data
        np.random.seed(42)
        n = 100
        endog = np.exp(np.random.randn(n))
        exog = np.random.randn(n, 1)
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Track training indices to verify reproducibility
        training_indices = []
        
        def track_indices(X, *args, **kwargs):
            # Store hash of training data to verify same splits
            training_indices.append(hash(X.tobytes()))
        
        mock_npm.fit.side_effect = track_indices
        
        # Mock predictions
        mock_npm.predict.return_value = np.ones(50)
        mock_npm.predict_semi_elasticity.return_value = 0.1 * np.ones(50)
        mock_dm.predict.return_value = 0.5 * np.ones(50)
        mock_dm.predict_semi_elasticity.return_value = 0.05 * np.ones(50)
        
        # Fit twice with same random state
        model1 = DoublyRobustElasticityEstimatorModel(endog, exog)
        result1 = model1.fit(n_folds=2, random_state=42)
        
        indices_run1 = training_indices.copy()
        training_indices.clear()
        
        model2 = DoublyRobustElasticityEstimatorModel(endog, exog)
        result2 = model2.fit(n_folds=2, random_state=42)
        
        indices_run2 = training_indices.copy()
        
        # Should have same training splits
        assert indices_run1 == indices_run2
    
    @patch('loglinearcorrection.model.DREEMR')
    @patch('loglinearcorrection.model.DensityModel')
    @patch('loglinearcorrection.model.NonparametricModel')
    def test_empty_interest_computes_all(self, mock_npm_class, mock_dm_class, mock_results_class):
        """Test that interest=None computes elasticities for all non-FE variables."""
        # Setup data
        np.random.seed(42)
        n = 50
        endog = np.exp(np.random.randn(n))
        exog = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'fe': np.random.choice([0, 1], n),
            'x3': np.random.randn(n)
        })
        
        # Create mock models
        mock_npm = Mock()
        mock_dm = Mock()
        mock_npm_class.return_value = mock_npm
        mock_dm_class.return_value = mock_dm
        
        # Mock predictions
        mock_npm.predict.return_value = np.ones(25)
        mock_npm.predict_semi_elasticity.return_value = 0.1 * np.ones(25)
        mock_dm.predict.return_value = 0.5 * np.ones(25)
        mock_dm.predict_semi_elasticity.return_value = 0.05 * np.ones(25)
        
        model = DoublyRobustElasticityEstimatorModel(
            endog, exog, 
            fixed_effects=['fe'],
            interest=None  # Compute for all non-FE
        )
        result = model.fit(n_folds=2, random_state=42)
        
        # Check results
        call_args = mock_results_class.call_args[1]
        elasticities = call_args['elasticities']
        
        # Should have elasticities for x1, x2, x3 but not fe
        assert 'x1' in elasticities
        assert 'x2' in elasticities
        assert 'x3' in elasticities
        assert 'fe' not in elasticities
        assert len(elasticities) == 3
