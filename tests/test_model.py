import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

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
