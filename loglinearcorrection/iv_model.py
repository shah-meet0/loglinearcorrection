"""
IV-DRNO: Doubly Robust Nonparametric Orthogonal Elasticity Estimator
with Instrumental Variables via Control Function Approach.

Implements Neyman-orthogonal estimation of the Average Structural Function
semi-elasticity when the treatment X is endogenous and instrumented by Z.

Model:
    X = g(Z) + V,  Z ⊥ V
    log Y = β X + ρ V + ε,  E[ε|X,V] = 0

Target:
    θ = β + E[μ'(X)/μ(X)]
    where μ(x) = E_V[m(x,V)] and m(x,v) = E[Y e^{-βX} | X=x, V=v]

Score:
    ψ = β + μ'(X)/μ(X) - (ω S_X / μ)(Y e^{-βX} - m) - λ̃(Z)(X - g(Z)) - θ
"""

from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import KFold
import statsmodels.api as sm

from .utils import (
    _apply_fixed_effects, _detect_variable_types, _delete_redundant,
    _initialize_fixed_effects, _adjust_names_after_deletion,
    _adjust_indices_after_deletion, _redundant_columns
)
from .nonparametric import NNModelNuisance, NNModelDensity
from .iv_nonparametric import NNModelDensityRatio, NNModelRieszLambda
from .iv_results import IVDREEMR


class IVDoublyRobustElasticityEstimatorModel:
    """
    IV-Doubly Robust Elasticity Estimator using Control Function approach.
    
    Estimates the semi-elasticity of the Average Structural Function
    E[Y(x)] with respect to x, accounting for endogeneity via instruments.
    
    Parameters
    ----------
    endog : array_like
        Dependent variable Y. Must be positive (log is taken internally).
    exog : array_like
        Endogenous treatment variable(s) X.
    instruments : array_like
        Instrumental variable(s) Z. Must satisfy Z ⊥ V and exclusion.
    exog_control : array_like, optional
        Exogenous control variables W that appear in both stages.
    weights : array_like, optional
        Observation weights.
    fixed_effects : list, optional
        Fixed effects specification (applied after first stage).
    interest : list, optional
        Indices of X variables for which to estimate elasticities.
    ordinal : list, optional
        Indices of ordinal variables.
        
    Attributes
    ----------
    endog : ndarray
        Outcome variable Y.
    exog : ndarray
        Endogenous regressors X.
    instruments : ndarray
        Instruments Z.
    V_hat : ndarray
        Estimated control function residuals (after fitting).
    beta : ndarray
        Control function OLS coefficients on X.
    rho : ndarray
        Control function OLS coefficients on V.
    """
    
    def __init__(
        self,
        endog: npt.ArrayLike,
        exog: npt.ArrayLike,
        instruments: npt.ArrayLike,
        exog_control: Optional[npt.ArrayLike] = None,
        weights: Optional[npt.ArrayLike] = None,
        fixed_effects: Optional[List] = None,
        interest: Optional[List] = None,
        ordinal: Optional[List] = None,
        **kwargs
    ) -> None:
        """Initialize the IV-DRNO estimator."""
        
        # Extract names
        self.endog_names = self._extract_name(endog, default="y")
        self._original_is_pandas = isinstance(exog, pd.DataFrame)
        
        # Convert to arrays
        endog_arr = np.asarray(endog).ravel().astype(np.float64)
        exog_arr = np.asarray(exog).astype(np.float64)
        instruments_arr = np.asarray(instruments).astype(np.float64)
        
        if exog_arr.ndim == 1:
            exog_arr = exog_arr.reshape(-1, 1)
        if instruments_arr.ndim == 1:
            instruments_arr = instruments_arr.reshape(-1, 1)
            
        # Extract exog names
        n_exog = exog_arr.shape[1]
        self.exog_names = (
            exog.columns.tolist() if isinstance(exog, pd.DataFrame)
            else [exog.name] if isinstance(exog, pd.Series)
            else [f"x{i+1}" for i in range(n_exog)]
        )
        
        # Extract instrument names
        n_inst = instruments_arr.shape[1]
        self.instrument_names = (
            instruments.columns.tolist() if isinstance(instruments, pd.DataFrame)
            else [instruments.name] if isinstance(instruments, pd.Series)
            else [f"z{i+1}" for i in range(n_inst)]
        )
        
        # Store data
        self.endog = endog_arr
        self.exog = exog_arr
        self.instruments = instruments_arr
        self.weights = np.asarray(weights) if weights is not None else None
        self.nobs = len(self.endog)
        
        # Exogenous controls
        if exog_control is not None:
            self.exog_control = np.asarray(exog_control).astype(np.float64)
            if self.exog_control.ndim == 1:
                self.exog_control = self.exog_control.reshape(-1, 1)
            n_control = self.exog_control.shape[1]
            self.control_names = (
                exog_control.columns.tolist() if isinstance(exog_control, pd.DataFrame)
                else [f"w{i+1}" for i in range(n_control)]
            )
        else:
            self.exog_control = None
            self.control_names = []
            
        # Validate
        if not np.issubdtype(self.endog.dtype, np.number):
            raise ValueError("endog must be numeric")
        if not np.all(self.endog > 0):
            raise ValueError("endog must be positive (log transformation applied)")
        if not np.issubdtype(self.exog.dtype, np.number):
            raise ValueError("exog must be numeric")
        if not np.issubdtype(self.instruments.dtype, np.number):
            raise ValueError("instruments must be numeric")
            
        # Fixed effects handling
        self.ordinal = ordinal
        if fixed_effects is not None:
            # For IV, FE are handled differently - store for later
            self._fe_spec = fixed_effects
            self.fixed_effects = None  # Will be initialized in fit
            self.fe_indices = []
        else:
            self._fe_spec = None
            self.fixed_effects = None
            self.fe_indices = []
            
        # Parse interest indices
        self.interest = self._parse_interest(interest)
        
        # Detect variable types for interest variables
        self.variable_types = _detect_variable_types(self.exog, self.interest)
        
        # Override with ordinal specification
        if ordinal is not None:
            ordinal_indices = self._parse_indices(ordinal, self.exog_names)
            for idx in ordinal_indices:
                if idx in self.variable_types and self.variable_types[idx] != "binary":
                    self.variable_types[idx] = "ordinal"
                    
        # Placeholders for fitted values
        self.V_hat = None
        self.beta = None
        self.rho = None
        self.ols_res = None
        
    def _extract_name(self, arr, default: str) -> str:
        """Extract name from pandas object or return default."""
        if isinstance(arr, pd.Series) and arr.name is not None:
            return arr.name
        if isinstance(arr, pd.DataFrame) and arr.columns[0] != 0:
            return arr.columns[0]
        return default
    
    def _parse_indices(self, spec: List, names: List[str]) -> List[int]:
        """Convert string names to indices if needed."""
        if spec is None:
            return []
        if isinstance(spec[0], str):
            return [names.index(name) for name in spec if name in names]
        return list(spec)
    
    def _parse_interest(self, interest) -> List[int]:
        """Parse interest specification."""
        if interest is None:
            return list(range(self.exog.shape[1]))
        if isinstance(interest, (int, str)):
            interest = [interest]
        if isinstance(interest[0], str):
            return [self.exog_names.index(name) for name in interest 
                    if name in self.exog_names]
        return list(interest)
    
    def _estimate_first_stage(
        self,
        train_idx: np.ndarray,
        first_stage_params: Dict
    ) -> 'NNModelNuisance':
        """
        Estimate first stage: g(Z, W) = E[X|Z, W].
        
        Handles:
        - Scalar or vector X (multiple endogenous variables)
        - Just-identified (k_z = k_x) or over-identified (k_z > k_x) cases
        - Optional exogenous controls W
        
        For over-identified case, the NN naturally handles dimension reduction.
        For just-identified, it learns the conditional expectation directly.
        
        Returns fitted model for prediction.
        """
        Z_train = self.instruments[train_idx]
        X_train = self.exog[train_idx]
        
        # If exogenous controls, include them in first stage
        if self.exog_control is not None:
            first_stage_input = np.column_stack([Z_train, self.exog_control[train_idx]])
        else:
            first_stage_input = Z_train
        
        k_z = self.instruments.shape[1]
        k_x = self.exog.shape[1]
        
        # Log identification status
        if k_z < k_x:
            raise ValueError(
                f"Under-identified: {k_z} instruments for {k_x} endogenous variables. "
                f"Need at least {k_x} instruments."
            )
        elif k_z == k_x:
            id_status = "just-identified"
        else:
            id_status = f"over-identified ({k_z} instruments for {k_x} endogenous)"
        
        print(f"    First stage: {id_status}")
        
        # Architecture params
        arch_params = first_stage_params.get('arch_params', {}).copy()
        arch_params['input_size'] = first_stage_input.shape[1]
        arch_params['output_size'] = k_x  # Output dimension matches X
        
        if 'hidden_layers' not in arch_params:
            # Scale network size with problem complexity
            base_width = max(64, min(256, 4 * (k_z + k_x)))
            arch_params['hidden_layers'] = [base_width, base_width, base_width]
        if 'output_activation' not in arch_params:
            arch_params['output_activation'] = 'identity'
            
        fit_params = first_stage_params.get('fit_params', {})
        
        # Use empty variable_types since first stage doesn't need type info
        g_model = NNModelNuisance(variable_types={}, **arch_params)
        g_results = g_model.fit(first_stage_input, X_train, **fit_params)
        
        return g_results
    
    def _predict_first_stage(
        self,
        g_results,
        Z: np.ndarray,
        W: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict g(Z, W) using fitted first stage model.
        
        Parameters
        ----------
        g_results : NNModelNuisanceResults
            Fitted first stage model.
        Z : ndarray, shape (n, k_z)
            Instrument values.
        W : ndarray, shape (n, k_w), optional
            Exogenous control values.
            
        Returns
        -------
        g_pred : ndarray, shape (n, k_x)
            Predicted E[X|Z, W].
        """
        if W is not None:
            first_stage_input = np.column_stack([Z, W])
        else:
            first_stage_input = Z
            
        g_pred = g_results.predict(first_stage_input)
        
        # Ensure 2D output
        if g_pred.ndim == 1:
            g_pred = g_pred.reshape(-1, 1)
            
        return g_pred
    
    def _estimate_control_function_ols(
        self,
        V_hat: np.ndarray,
        weight_fold: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
        """
        Estimate control function OLS: log Y = β'X + γ'W + ρ'V + ε.
        
        Parameters
        ----------
        V_hat : ndarray, shape (n, k_x)
            Estimated control function residuals.
        weight_fold : ndarray, optional
            Observation weights.
            
        Returns
        -------
        beta : ndarray, shape (k_x,)
            Coefficients on endogenous variables X.
        gamma : ndarray, shape (k_w,) or None
            Coefficients on exogenous controls W.
        rho : ndarray, shape (k_x,)
            Coefficients on control function V.
        ols_results : statsmodels results
            Full OLS results object.
        """
        k_x = self.exog.shape[1]
        k_v = V_hat.shape[1] if V_hat.ndim > 1 else 1
        
        # Ensure V_hat is 2D
        if V_hat.ndim == 1:
            V_hat = V_hat.reshape(-1, 1)
        
        # Build design matrix: [X, W, V]
        design_parts = [self.exog, V_hat]
        
        if self.exog_control is not None:
            k_w = self.exog_control.shape[1]
            design_parts = [self.exog, self.exog_control, V_hat]
        else:
            k_w = 0
            
        design = np.column_stack(design_parts)
        
        # Apply fixed effects if specified
        log_y = np.log(self.endog)
        
        if self._fe_spec is not None:
            # Initialize and apply FE demeaning
            if self.fixed_effects is None and hasattr(self, '_fe_data'):
                self.fixed_effects = _initialize_fixed_effects(self._fe_data)
            if self.fixed_effects is not None:
                from .utils import _apply_fixed_effects
                log_y_dm, design_dm = _apply_fixed_effects(
                    log_y, design, self.fixed_effects, 
                    weights=weight_fold.reshape(-1, 1) if weight_fold is not None else None
                )
                log_y = log_y_dm
                design = design_dm
        
        # Fit OLS
        if weight_fold is not None:
            ols = sm.WLS(log_y, design, weights=weight_fold)
        else:
            ols = sm.OLS(log_y, design)
            
        ols_res = ols.fit()
        
        # Extract coefficients
        # Order: [X (k_x), W (k_w), V (k_v)]
        beta = ols_res.params[:k_x]
        
        if k_w > 0:
            gamma = ols_res.params[k_x:k_x + k_w]
            rho = ols_res.params[k_x + k_w:k_x + k_w + k_v]
        else:
            gamma = None
            rho = ols_res.params[k_x:k_x + k_v]
        
        return beta, gamma, rho, ols_res
    
    def _compute_mu_and_derivative(
        self,
        m_results,
        X_test: np.ndarray,
        V_all: np.ndarray,
        interest_indices: List[int],
        n_mc_samples: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute μ(x) = E_V[m(x,V)] and μ'(x) via Monte Carlo integration.
        
        For each test point x_i, we average m(x_i, V_j) over sampled V_j.
        
        Parameters
        ----------
        m_results : NNModelNuisanceResults
            Fitted m(x,v) model. Input is [X, V] concatenated.
        X_test : ndarray, shape (n_test, k_x)
            Test X values.
        V_all : ndarray, shape (n_all, k_v) or (n_all,)
            All V values for MC integration. Can be vector-valued.
        interest_indices : list
            Indices of X variables (in original X) for derivative computation.
        n_mc_samples : int
            Number of V samples for MC integration.
            
        Returns
        -------
        mu : ndarray, shape (n_test,)
            μ(x) values.
        mu_prime : ndarray, shape (n_test, len(interest_indices))
            ∂μ(x)/∂x_k for each interest variable.
            
        Notes
        -----
        The m model takes input [X, V] where:
        - X has shape (*, k_x)
        - V has shape (*, k_v)
        
        So for derivative w.r.t. X_k, we need to pass index k directly
        since the input ordering is [x_0, x_1, ..., x_{k_x-1}, v_0, ..., v_{k_v-1}].
        """
        n_test = X_test.shape[0]
        k_x = X_test.shape[1]
        
        # Ensure V_all is 2D
        if V_all.ndim == 1:
            V_all = V_all.reshape(-1, 1)
        n_all = V_all.shape[0]
        k_v = V_all.shape[1]
        
        # Sample V indices for MC (sample entire rows)
        if n_mc_samples >= n_all:
            V_samples = V_all
        else:
            sample_idx = np.random.choice(n_all, size=n_mc_samples, replace=False)
            V_samples = V_all[sample_idx]
            
        n_samples = V_samples.shape[0]
        
        # For each test point, compute m(x_i, v_j) for all sampled v_j
        # This is expensive but necessary for marginal integration
        
        mu = np.zeros(n_test)
        mu_prime = np.zeros((n_test, len(interest_indices)))
        
        # For memory efficiency, process in batches
        batch_size = min(100, n_test)
        
        for batch_start in range(0, n_test, batch_size):
            batch_end = min(batch_start + batch_size, n_test)
            batch_size_actual = batch_end - batch_start
            
            X_batch = X_test[batch_start:batch_end]
            
            # Expand for MC integration
            # X_expanded[i*n_samples + j, :] = x_i for all j in 0..n_samples-1
            # V_expanded[i*n_samples + j, :] = v_j
            X_expanded = np.repeat(X_batch, n_samples, axis=0)  # (batch*n_samples, k_x)
            V_expanded = np.tile(V_samples, (batch_size_actual, 1))  # (batch*n_samples, k_v)
            
            # Input to m is [X, V], so shape is (batch*n_samples, k_x + k_v)
            m_input = np.column_stack([X_expanded, V_expanded])
            
            # For derivatives, we want ∂m/∂x_k for k in interest_indices
            # Since input is [X, V], the indices for X variables are unchanged
            # (they're still 0, 1, ..., k_x-1 in the concatenated input)
            
            # Get predictions and derivatives w.r.t. original X indices
            m_pred, m_deriv = m_results.derivative(
                m_input, 
                var_index=interest_indices  # These are X indices, valid in [X, V] input
            )
            
            # Reshape and average over V samples
            m_pred_reshaped = m_pred.reshape(batch_size_actual, n_samples)
            mu[batch_start:batch_end] = m_pred_reshaped.mean(axis=1)
            
            # Derivatives: m_deriv is (batch*n_samples, len(interest_indices))
            m_deriv_reshaped = m_deriv.reshape(batch_size_actual, n_samples, -1)
            mu_prime[batch_start:batch_end] = m_deriv_reshaped.mean(axis=1)
            
        return mu, mu_prime
    
    def _compute_pathwise_derivative_autodiff(
        self,
        X: np.ndarray,
        V: np.ndarray,
        Z: np.ndarray,
        Y_transformed: np.ndarray,
        m_model,
        omega_model,
        density_model,
        interest_indices: List[int],
        beta: np.ndarray
    ) -> np.ndarray:
        """
        Compute pathwise derivative D_g ψ using PyTorch automatic differentiation.
        
        Computes separately for each (interest variable, V dimension) pair.
        
        The pathwise derivative measures how the score changes when we perturb
        g(Z) → g(Z) + δ_j in direction j, equivalently V_j → V_j - δ.
        
        Returns
        -------
        D_g : ndarray, shape (n, n_interest, k_v)
            Pathwise derivative for each observation, interest variable, and V dimension.
            This captures Λ^dir + Λ^ω (the direct effects).
        """
        import torch
        
        device = m_model.device
        n = X.shape[0]
        k_v = V.shape[1] if V.ndim > 1 else 1
        n_interest = len(interest_indices)
        
        # Convert to tensors with gradient tracking on V
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        V_t = torch.tensor(V.reshape(-1, k_v), dtype=torch.float32, device=device, requires_grad=True)
        Y_trans_t = torch.tensor(Y_transformed, dtype=torch.float32, device=device)
        
        # Forward pass through m
        m_input = torch.cat([X_t, V_t], dim=1)
        m_model.model.eval()
        m_pred = m_model.model(m_input).squeeze(-1)
        
        # R = Y_trans - m
        R = Y_trans_t - m_pred
        
        # ω(X, V)
        omega_input = torch.cat([X_t, V_t], dim=1)
        omega_model.model.eval()
        omega_logits = omega_model.model(omega_input).squeeze(-1)
        p = torch.sigmoid(omega_logits)
        p = torch.clamp(p, min=1e-6, max=1 - 1e-6)
        r = p / (1 - p)
        omega = 1.0 / r
        omega = torch.clamp(omega, min=0.01, max=100.0)
        
        # S_X(X) - doesn't depend on V
        with torch.no_grad():
            S_X = density_model.alpha_weight(X, interest_indices)
            S_X_t = torch.tensor(S_X, dtype=torch.float32, device=device)
        
        # μ proxy (use m since we don't have μ with gradient tracking)
        mu_proxy = m_pred.detach()
        mu_proxy = torch.clamp(mu_proxy, min=1e-6)
        
        # Compute D_g for each interest variable separately
        D_g = np.zeros((n, n_interest, k_v))
        
        for k_idx, var_idx in enumerate(interest_indices):
            # Score component for interest variable k:
            # ψ_k,correction = -ω(X,V) S_{X,k}(X) R(X,V) / μ(X)
            alpha_k = -omega * S_X_t[:, k_idx] / mu_proxy
            score_k = alpha_k * R
            
            # Compute gradient w.r.t. V
            # Need to recompute graph for each k since we need separate gradients
            if k_idx > 0:
                # Recompute with fresh graph
                V_t = torch.tensor(V.reshape(-1, k_v), dtype=torch.float32, device=device, requires_grad=True)
                m_input = torch.cat([X_t, V_t], dim=1)
                m_pred = m_model.model(m_input).squeeze(-1)
                R = Y_trans_t - m_pred
                
                omega_input = torch.cat([X_t, V_t], dim=1)
                omega_logits = omega_model.model(omega_input).squeeze(-1)
                p = torch.sigmoid(omega_logits)
                p = torch.clamp(p, min=1e-6, max=1 - 1e-6)
                r = p / (1 - p)
                omega = 1.0 / r
                omega = torch.clamp(omega, min=0.01, max=100.0)
                
                mu_proxy = m_pred.detach()
                mu_proxy = torch.clamp(mu_proxy, min=1e-6)
                
                alpha_k = -omega * S_X_t[:, k_idx] / mu_proxy
                score_k = alpha_k * R
            
            grad_outputs = torch.ones_like(score_k)
            
            grads = torch.autograd.grad(
                outputs=score_k,
                inputs=V_t,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=(k_idx < n_interest - 1),
                allow_unused=False
            )[0]
            
            # D_g = -∂ψ/∂V (negative because g perturbation = -V perturbation)
            D_g[:, k_idx, :] = -grads.detach().cpu().numpy()
        
        return D_g
    
    def _compute_lambda_indirect(
        self,
        X: np.ndarray,
        V_samples: np.ndarray,
        m_model,
        mu: np.ndarray,
        mu_prime: np.ndarray,
        interest_indices: List[int],
        n_mc_samples: int = 200
    ) -> np.ndarray:
        """
        Compute Λ^ind_{k,j}(X) for each (interest variable k, V dimension j).
        
        Formula:
            Λ^ind_{k,j} = μ'_k(X) m̄_{v_j}(X) / μ(X)² - m̄_{x_k v_j}(X) / μ(X)
        
        where:
            m̄_{v_j}(x) = E_V[∂m/∂v_j(x, V)]
            m̄_{x_k v_j}(x) = E_V[∂²m/∂x_k∂v_j(x, V)]
        
        This captures the indirect effect of g-perturbation through the
        distribution shift in μ(x) = E_V[m(x,V)].
        
        Returns
        -------
        Lambda_ind : ndarray, shape (n, n_interest, k_v)
            Indirect pathwise derivative for each (observation, interest var, V dim).
        """
        import torch
        
        device = m_model.device
        n = X.shape[0]
        k_x = X.shape[1]
        n_interest = len(interest_indices)
        
        # Ensure V_samples is 2D
        if V_samples.ndim == 1:
            V_samples = V_samples.reshape(-1, 1)
        k_v = V_samples.shape[1]
        
        # Sample V for MC
        n_v = V_samples.shape[0]
        if n_mc_samples < n_v:
            sample_idx = np.random.choice(n_v, size=n_mc_samples, replace=False)
            V_mc = V_samples[sample_idx]
        else:
            V_mc = V_samples
        n_samples = V_mc.shape[0]
        
        # We need to compute for each x_i:
        # m̄_{v_j}(x) = E_V[∂m/∂v_j] for each V dimension j
        # m̄_{x_k v_j}(x) = E_V[∂²m/∂x_k∂v_j] for each (interest var k, V dim j)
        
        m_bar_v = np.zeros((n, k_v))  # E[∂m/∂v_j] for each j
        m_bar_xv = np.zeros((n, n_interest, k_v))  # E[∂²m/∂x_k∂v_j]
        
        # Process in batches for memory efficiency
        batch_size = min(50, n)
        
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            X_batch = X[batch_start:batch_end]
            batch_n = batch_end - batch_start
            
            # Expand for MC: (batch_n * n_samples, k_x + k_v)
            X_expanded = np.repeat(X_batch, n_samples, axis=0)
            V_expanded = np.tile(V_mc, (batch_n, 1))
            
            # Convert to tensors with gradients
            X_t = torch.tensor(X_expanded, dtype=torch.float32, device=device, requires_grad=True)
            V_t = torch.tensor(V_expanded, dtype=torch.float32, device=device, requires_grad=True)
            
            m_input = torch.cat([X_t, V_t], dim=1)
            
            # Forward pass
            m_model.model.eval()
            m_pred = m_model.model(m_input).squeeze(-1)
            
            # Compute ∂m/∂v (gradient w.r.t. V, shape: batch*n_samples x k_v)
            grad_v = torch.autograd.grad(
                outputs=m_pred.sum(),
                inputs=V_t,
                create_graph=True,  # Need for second derivatives
                retain_graph=True
            )[0]  # shape: (batch_n * n_samples, k_v)
            
            # Compute ∂²m/∂x_k∂v_j for each (interest variable k, V dimension j)
            # This is the mixed Hessian
            for k_idx, var_idx in enumerate(interest_indices):
                for v_dim in range(k_v):
                    # ∂/∂x_k of (∂m/∂v_j)
                    grad_v_j = grad_v[:, v_dim]
                    
                    hess = torch.autograd.grad(
                        outputs=grad_v_j.sum(),
                        inputs=X_t,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    
                    if hess is not None:
                        hess_kj = hess[:, var_idx].detach().cpu().numpy()
                    else:
                        hess_kj = np.zeros(X_t.shape[0])
                    
                    # Reshape and average over MC samples
                    hess_reshaped = hess_kj.reshape(batch_n, n_samples)
                    m_bar_xv[batch_start:batch_end, k_idx, v_dim] = hess_reshaped.mean(axis=1)
            
            # Average ∂m/∂v over MC samples
            grad_v_np = grad_v.detach().cpu().numpy()
            grad_v_reshaped = grad_v_np.reshape(batch_n, n_samples, k_v)
            m_bar_v[batch_start:batch_end] = grad_v_reshaped.mean(axis=1)
        
        # Compute Λ^ind_{k,j} for each (interest variable k, V dimension j)
        # Λ^ind_{k,j} = μ'_k(X) * m̄_{v_j}(X) / μ(X)² - m̄_{x_k v_j}(X) / μ(X)
        Lambda_ind = np.zeros((n, n_interest, k_v))
        
        mu_safe = np.maximum(mu, 1e-10)
        
        for k_idx in range(n_interest):
            for v_dim in range(k_v):
                term1 = mu_prime[:, k_idx] * m_bar_v[:, v_dim] / (mu_safe ** 2)
                term2 = m_bar_xv[:, k_idx, v_dim] / mu_safe
                Lambda_ind[:, k_idx, v_dim] = term1 - term2
        
        return Lambda_ind
    
    def _process_fold(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        weight_fold: np.ndarray,
        interest_indices: List[int],
        first_stage_params: Dict,
        m_params: Dict,
        density_params: Dict,
        omega_params: Dict,
        lambda_params: Dict
    ) -> Dict:
        """
        Process a single cross-fitting fold.
        
        Steps:
        1. Estimate first stage g(Z, W) on train
        2. Compute V_hat = X - g(Z, W) on full sample (for OLS)
        3. Estimate beta, gamma, rho via control function OLS
        4. Estimate m(X, V) on train
        5. Compute mu(x), mu'(x) on test via MC integration
        6. Estimate omega(X, V) via classification on train
        7. Estimate S_X(X) via score matching on train
        8. Estimate lambda(Z) via automatic DML with autodiff
        9. Construct moments
        """
        
        k_x = self.exog.shape[1]
        k_v = k_x  # V has same dimension as X
        
        # Step 1: First stage
        print("  Estimating first stage g(Z, W)...")
        g_results = self._estimate_first_stage(train_idx, first_stage_params)
        
        # Predict g(Z, W) on full sample for OLS
        W_full = self.exog_control if self.exog_control is not None else None
        g_pred_full = self._predict_first_stage(g_results, self.instruments, W_full)
        
        V_hat_full = self.exog - g_pred_full  # Shape: (n, k_x)
        
        # Step 2-3: Control function OLS on full sample
        print("  Estimating control function OLS...")
        beta, gamma, rho, ols_res = self._estimate_control_function_ols(V_hat_full, weight_fold)
        
        # Store for later use
        self.V_hat = V_hat_full
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.ols_res = ols_res
        
        # Compute residuals for m estimation
        # m(x,v) = E[Y e^{-β'X} | X=x, V=v]
        Y_transformed = self.endog * np.exp(-self.exog @ beta)
        
        # Step 4: Estimate m(X, V) on train
        print("  Estimating nuisance m(X, V)...")
        X_train = self.exog[train_idx]
        V_train = V_hat_full[train_idx]
        Y_trans_train = Y_transformed[train_idx]
        
        # Input to m is [X, V], shape: (n, k_x + k_v)
        m_input_train = np.column_stack([X_train, V_train])
        
        # Update m_params for joint (X, V) input
        m_arch = m_params.get('arch_params', {}).copy()
        m_arch['input_size'] = m_input_train.shape[1]
        m_arch['output_size'] = 1
        if 'hidden_layers' not in m_arch:
            base_width = max(256, min(512, 8 * (k_x + k_v)))
            m_arch['hidden_layers'] = [base_width, base_width, base_width]
        if 'output_activation' not in m_arch:
            m_arch['output_activation'] = 'softplus'  # Ensure positivity
            
        m_model = NNModelNuisance(variable_types={}, **m_arch)
        m_results = m_model.fit(m_input_train, Y_trans_train, **m_params.get('fit_params', {}))
        
        # Step 5: Compute mu(x) and mu'(x) on test
        print("  Computing marginal integration μ(x)...")
        X_test = self.exog[test_idx]
        V_test = V_hat_full[test_idx]
        
        # For MC integration, use V from training set (independent of test X)
        mu_test, mu_prime_test = self._compute_mu_and_derivative(
            m_results, X_test, V_train, interest_indices
        )
        
        # Ensure positivity
        mu_test = np.maximum(mu_test, 1e-10)
        
        # Get m(X_test, V_test) for residual
        m_input_test = np.column_stack([X_test, V_test])
        m_test = m_results.predict(m_input_test)
        m_test = np.maximum(m_test, 1e-10)
        
        # R = Y e^{-β'X} - m(X, V)
        Y_trans_test = Y_transformed[test_idx]
        R_test = Y_trans_test - m_test
        
        # Step 6: Estimate omega(X, V) via classification
        print("  Estimating density ratio ω(X, V)...")
        omega_model = NNModelDensityRatio(**omega_params.get('arch_params', {}))
        omega_results = omega_model.fit(
            X_train, V_train,
            **omega_params.get('fit_params', {})
        )
        omega_test = omega_results.predict(X_test, V_test)
        
        # Step 7: Estimate S_X(X) via score matching
        print("  Estimating score S_X(X)...")
        density_model = NNModelDensity(variable_types=self.variable_types, **density_params.get('arch_params', {}))
        density_results = density_model.fit(X_train, interest=interest_indices, **density_params.get('fit_params', {}))
        
        # Get score for continuous interest variables
        S_X_test = density_results.alpha_weight(X_test, interest_indices)
        
        # Step 8: Estimate lambda(Z) via automatic DML with autodiff
        print("  Estimating λ(Z) correction via autodiff...")
        Z_train = self.instruments[train_idx]
        Z_test = self.instruments[test_idx]
        n_interest = len(interest_indices)
        
        # Compute pathwise derivative on training set using autodiff
        # This captures Λ^dir + Λ^ω, shape: (n_train, n_interest, k_v)
        D_g_direct = self._compute_pathwise_derivative_autodiff(
            X_train, V_train, Z_train, Y_trans_train,
            m_model, omega_model, density_results,
            interest_indices, beta
        )
        
        # Compute Λ^ind separately (distribution shift effect)
        # Need mu and mu_prime on training set for this
        # Shape: (n_train, n_interest, k_v)
        mu_train, mu_prime_train = self._compute_mu_and_derivative(
            m_results, X_train, V_train, interest_indices
        )
        mu_train = np.maximum(mu_train, 1e-10)
        
        Lambda_ind = self._compute_lambda_indirect(
            X_train, V_train, m_model,
            mu_train, mu_prime_train, interest_indices
        )
        
        # Full pathwise derivative: D_g = (Λ^dir + Λ^ω) + Λ^ind
        # Both have shape (n_train, n_interest, k_v)
        D_g_full = D_g_direct + Lambda_ind
        
        # Regress D_g on Z for each (interest variable, V dimension) pair
        # lambda_test will have shape (n_test, n_interest, k_v)
        lambda_test = np.zeros((len(test_idx), n_interest, k_v))
        lambda_results_dict = {}
        
        for k_idx in range(n_interest):
            lambda_results_dict[k_idx] = []
            for v_idx in range(k_v):
                D_g_kv = D_g_full[:, k_idx, v_idx]
                
                lambda_model_kv = NNModelRieszLambda(**lambda_params.get('arch_params', {}))
                lambda_res = lambda_model_kv.fit(Z_train, D_g_kv, **lambda_params.get('fit_params', {}))
                lambda_results_dict[k_idx].append(lambda_res)
                
                lambda_test[:, k_idx, v_idx] = lambda_res.predict(Z_test)
        
        # Step 9: Construct moments
        print("  Constructing moments...")
        n_test = len(test_idx)
        
        fold_moments = np.zeros((n_test, n_interest))
        
        for moment_idx, var_idx in enumerate(interest_indices):
            var_type = self.variable_types.get(var_idx, 'continuous')
            
            if var_type == 'continuous':
                # θ(x) = β_k + μ'_k(x)/μ(x)
                theta_x = beta[var_idx] + mu_prime_test[:, moment_idx] / mu_test
                
                # α(x,v) = -ω(x,v) S_X,k(x) / μ(x)
                alpha_x = -omega_test * S_X_test[:, moment_idx] / mu_test
                
                # Correction term: λ̃_k(Z) · V = Σ_j λ̃_{k,j}(Z) V_j
                # lambda_test has shape (n_test, n_interest, k_v)
                # V_test has shape (n_test, k_v)
                lambda_k = lambda_test[:, moment_idx, :]  # shape (n_test, k_v)
                lambda_correction = np.sum(lambda_k * V_test, axis=1)
                
                fold_moments[:, moment_idx] = theta_x + alpha_x * R_test - lambda_correction
                
            elif var_type == 'binary':
                # Binary treatment: compute discrete change
                X_test_flip = X_test.copy()
                X_test_flip[:, var_idx] = 1 - X_test_flip[:, var_idx]
                
                m_input_flip = np.column_stack([X_test_flip, V_test])
                m_flip = m_results.predict(m_input_flip)
                m_flip = np.maximum(m_flip, 1e-10)
                
                # Compute μ(x_flip) via MC
                mu_flip, _ = self._compute_mu_and_derivative(
                    m_results, X_test_flip, V_train, interest_indices
                )
                mu_flip = np.maximum(mu_flip, 1e-10)
                
                # Probability weights from density model for binary
                p_var = density_results.alpha_weight(X_test, [var_idx])[:, 0]
                
                # Elasticity for binary: exp(β_k) * μ(x^{+k})/μ(x) - 1
                theta_x = np.exp(beta[var_idx]) * (mu_flip / mu_test) - 1
                
                # Influence function for binary (simplified)
                alpha_0 = (1 - X_test[:, var_idx]) * m_flip / (m_test**2 * (np.abs(p_var) + 1e-10))
                alpha_1 = X_test[:, var_idx] / ((np.abs(p_var) + 1e-10) * m_flip)
                alpha_x = np.exp(beta[var_idx]) * (alpha_1 - alpha_0)
                
                # Lambda correction for this interest variable
                lambda_k = lambda_test[:, moment_idx, :]  # shape (n_test, k_v)
                lambda_correction = np.sum(lambda_k * V_test, axis=1)
                
                fold_moments[:, moment_idx] = theta_x + alpha_x * R_test - lambda_correction
                
            else:  # ordinal
                fold_moments[:, moment_idx] = 0  # Placeholder
                
        return {
            'beta': beta,
            'gamma': gamma,
            'rho': rho,
            'g_results': g_results,
            'm_results': m_results,
            'm_model': m_model,
            'omega_results': omega_results,
            'omega_model': omega_model,
            'density_results': density_results,
            'density_model': density_model,
            'lambda_results': lambda_results_dict,
            'D_g_full': D_g_full,  # Full pathwise derivative (Λ^dir + Λ^ω + Λ^ind)
            'moments': fold_moments,
            'mu_test': mu_test,
            'mu_prime_test': mu_prime_test,
            'test_idx': test_idx
        }
    
    def fit(
        self,
        n_folds: int = 5,
        random_state: Optional[int] = None,
        first_stage_params: Optional[Dict] = None,
        m_params: Optional[Dict] = None,
        density_params: Optional[Dict] = None,
        omega_params: Optional[Dict] = None,
        lambda_params: Optional[Dict] = None,
        n_mc_samples: int = 500
    ) -> 'IVDREEMR':
        """
        Fit the IV-DRNO estimator via cross-fitting.
        
        Parameters
        ----------
        n_folds : int, default=5
            Number of cross-fitting folds.
        random_state : int, optional
            Random seed for reproducibility.
        first_stage_params : dict, optional
            Parameters for first stage g(Z) estimation.
        m_params : dict, optional
            Parameters for m(X,V) nuisance estimation.
        density_params : dict, optional
            Parameters for S_X score estimation.
        omega_params : dict, optional
            Parameters for density ratio ω estimation.
        lambda_params : dict, optional
            Parameters for λ(Z) estimation.
        n_mc_samples : int, default=500
            Number of samples for Monte Carlo integration of μ(x).
            
        Returns
        -------
        IVDREEMR
            Results object with estimates and inference.
        """
        
        # Parse parameters with defaults
        first_stage_params = self._parse_first_stage_params(first_stage_params)
        m_params = self._parse_m_params(m_params)
        density_params = self._parse_density_params(density_params)
        omega_params = self._parse_omega_params(omega_params)
        lambda_params = self._parse_lambda_params(lambda_params)
        
        # Set up cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Process each fold
        fold_results = []
        interest_indices = self.interest
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.exog)):
            print(f"Processing fold {fold_idx + 1}/{n_folds}")
            
            weight_fold = np.ones(self.nobs)
            if self.weights is not None:
                weight_fold = weight_fold * self.weights
                
            fold_result = self._process_fold(
                train_idx, test_idx, weight_fold,
                interest_indices, first_stage_params, m_params,
                density_params, omega_params, lambda_params
            )
            fold_results.append(fold_result)
            
        # Aggregate moments
        n_interest = len(interest_indices)
        moments = np.zeros((self.nobs, n_interest))
        fold_weights = np.zeros(self.nobs)
        
        for fold in fold_results:
            moments[fold['test_idx']] = fold['moments']
            fold_weights[fold['test_idx']] = (
                self.weights[fold['test_idx']] if self.weights is not None else 1.0
            )
            
        # Compute estimates
        moment_means = np.average(moments, axis=0, weights=fold_weights)
        elasticity_estimates = moment_means
        
        # Build results DataFrame
        if self._original_is_pandas:
            elasticities_df = pd.DataFrame({
                'variable': [self.exog_names[i] for i in interest_indices],
                'estimate': elasticity_estimates,
                'type': [self.variable_types.get(i, 'continuous') for i in interest_indices],
                'index': interest_indices
            }).set_index('variable')
        else:
            elasticities_df = elasticity_estimates
            
        # Create results object
        results = IVDREEMR(
            elasticities=elasticities_df,
            beta=self.beta[interest_indices] if self.beta is not None else None,
            gamma=self.gamma,
            rho=self.rho,
            ols_results=self.ols_res,
            exog_names=self.exog_names,
            endog_name=self.endog_names,
            instrument_names=self.instrument_names,
            control_names=self.control_names,
            n_folds=n_folds,
            nobs=self.nobs,
            variable_types=self.variable_types,
            interest_indices=interest_indices,
            fold_results=fold_results,
            moments=moments,
            fold_weights=fold_weights,
            V_hat=self.V_hat
        )
        
        # Compute variances
        results.compute_variances()
        
        return results
    
    def _parse_first_stage_params(self, params: Optional[Dict]) -> Dict:
        """Parse first stage parameters with defaults."""
        if params is None:
            params = {}
        
        defaults = {
            'arch_params': {
                'hidden_layers': [256, 256, 256],
                'output_activation': 'identity'
            },
            'fit_params': {
                'epochs': 100,
                'patience': 15
            }
        }
        
        result = {
            'arch_params': {**defaults['arch_params'], **params.get('arch_params', {})},
            'fit_params': {**defaults['fit_params'], **params.get('fit_params', {})}
        }
        return result
    
    def _parse_m_params(self, params: Optional[Dict]) -> Dict:
        """Parse m(X,V) model parameters with defaults."""
        if params is None:
            params = {}
            
        defaults = {
            'arch_params': {
                'hidden_layers': [512, 512, 512],
                'output_activation': 'softplus'  # Ensure m > 0
            },
            'fit_params': {
                'epochs': 150,
                'patience': 20
            }
        }
        
        result = {
            'arch_params': {**defaults['arch_params'], **params.get('arch_params', {})},
            'fit_params': {**defaults['fit_params'], **params.get('fit_params', {})}
        }
        return result
    
    def _parse_density_params(self, params: Optional[Dict]) -> Dict:
        """Parse density/score model parameters."""
        if params is None:
            params = {}
            
        defaults = {
            'arch_params': {
                'shared': {'hidden_layers': [512, 512, 512]},
                'score': {},
                'cond': {}
            },
            'fit_params': {}
        }
        
        result = {
            'arch_params': {**defaults['arch_params'], **params.get('arch_params', {})},
            'fit_params': {**defaults['fit_params'], **params.get('fit_params', {})}
        }
        return result
    
    def _parse_omega_params(self, params: Optional[Dict]) -> Dict:
        """Parse density ratio omega parameters."""
        if params is None:
            params = {}
            
        defaults = {
            'arch_params': {
                'hidden_layers': [256, 256],
            },
            'fit_params': {
                'epochs': 100,
                'n_permutations': 5
            }
        }
        
        result = {
            'arch_params': {**defaults['arch_params'], **params.get('arch_params', {})},
            'fit_params': {**defaults['fit_params'], **params.get('fit_params', {})}
        }
        return result
    
    def _parse_lambda_params(self, params: Optional[Dict]) -> Dict:
        """Parse lambda correction parameters."""
        if params is None:
            params = {}
            
        defaults = {
            'arch_params': {
                'hidden_layers': [128, 128],
            },
            'fit_params': {
                'epochs': 50
            }
        }
        
        result = {
            'arch_params': {**defaults['arch_params'], **params.get('arch_params', {})},
            'fit_params': {**defaults['fit_params'], **params.get('fit_params', {})}
        }
        return result


# Convenience alias
IVDREEM = IVDoublyRobustElasticityEstimatorModel
