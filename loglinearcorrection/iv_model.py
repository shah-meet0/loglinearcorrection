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
import torch


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
    
    def _estimate_control_function_ols_fold(
        self,
        train_idx: np.ndarray,
        V_hat_full: np.ndarray,
        weight_fold: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
        """
        Estimate control function OLS on training data only.

        log Y = β'X + δ'W + ρ'V + ε

        Parameters
        ----------
        train_idx : ndarray
            Indices of training observations.
        V_hat_full : ndarray, shape (n, k_x)
            Estimated control function residuals for all observations.
        weight_fold : ndarray, optional
            Observation weights for all observations.

        Returns
        -------
        beta : ndarray, shape (k_x,)
            Coefficients on endogenous variables X.
        delta : ndarray, shape (k_w,) or None
            Coefficients on exogenous controls W.
        rho : ndarray, shape (k_x,)
            Coefficients on control function V.
        ols_results : statsmodels results
            Full OLS results object.
        """
        k_x = self.exog.shape[1]
        k_v = V_hat_full.shape[1] if V_hat_full.ndim > 1 else 1

        # Ensure V_hat is 2D
        if V_hat_full.ndim == 1:
            V_hat_full = V_hat_full.reshape(-1, 1)

        # Extract training data
        X_train = self.exog[train_idx]
        V_train = V_hat_full[train_idx]
        log_y_train = np.log(self.endog[train_idx])

        # Build design matrix: [X, W, V] on training data
        if self.exog_control is not None:
            k_w = self.exog_control.shape[1]
            W_train = self.exog_control[train_idx]
            design_train = np.column_stack([X_train, W_train, V_train])
        else:
            k_w = 0
            design_train = np.column_stack([X_train, V_train])

        # Get training weights
        weights_train = weight_fold[train_idx] if weight_fold is not None else None

        # Fit OLS on training data
        if weights_train is not None:
            ols = sm.WLS(log_y_train, design_train, weights=weights_train)
        else:
            ols = sm.OLS(log_y_train, design_train)

        ols_res = ols.fit()

        # Extract coefficients
        # Order: [X (k_x), W (k_w), V (k_v)]
        beta = ols_res.params[:k_x]

        if k_w > 0:
            delta = ols_res.params[k_x:k_x + k_w]
            rho = ols_res.params[k_x + k_w:k_x + k_w + k_v]
        else:
            delta = None
            rho = ols_res.params[k_x:k_x + k_v]

        return beta, delta, rho, ols_res
    
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
    
    def _compute_uncorrected_score_with_grad(
        self,
        X: np.ndarray,
        V: np.ndarray,
        Y_transformed: np.ndarray,
        m_results,  # NNModelNuisanceResults
        omega_results,  # NNModelDensityRatioResults  
        density_results,  # NNModelDensityResults
        mu: np.ndarray,
        mu_prime: np.ndarray,
        beta: np.ndarray,
        interest_indices: list
    ) -> tuple:
        """
        Compute uncorrected score and its gradient w.r.t. V via autodiff.
        
        Fixed version that properly handles results objects.
        """
        import torch
        
        # Get device from m_results
        device = m_results.device
        n = X.shape[0]
        k_v = V.shape[1] if V.ndim > 1 else 1
        n_interest = len(interest_indices)
        
        if V.ndim == 1:
            V = V.reshape(-1, 1)
        
        # Convert to tensors
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        V_t = torch.tensor(V, dtype=torch.float32, device=device, requires_grad=True)
        Y_trans_t = torch.tensor(Y_transformed, dtype=torch.float32, device=device)
        mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
        mu_prime_t = torch.tensor(mu_prime, dtype=torch.float32, device=device)
        
        mu_t = torch.clamp(mu_t, min=1e-8)
        
        # Forward pass through m(X, V) - use the underlying model from results
        m_input = torch.cat([X_t, V_t], dim=1)
        m_results.model.eval()
        m_pred = m_results.model(m_input).squeeze(-1)
        m_pred = torch.clamp(m_pred, min=1e-8)
        
        R = Y_trans_t - m_pred
        
        # Forward pass through ω(X, V) - use underlying model from results
        omega_input = torch.cat([X_t, V_t], dim=1)
        omega_results.model.eval()
        omega_logits = omega_results.model(omega_input).squeeze(-1)
        p = torch.sigmoid(omega_logits)
        p = torch.clamp(p, min=1e-6, max=1 - 1e-6)
        omega = (1 - p) / p
        omega = torch.clamp(omega, min=0.01, max=100.0)
        
        # Get S_X(X) - no grad needed
        with torch.no_grad():
            S_X = density_results.alpha_weight(X, interest_indices)
            S_X_t = torch.tensor(S_X, dtype=torch.float32, device=device)
        
        psi_uncorr = np.zeros((n, n_interest))
        D_g_psi = np.zeros((n, n_interest, k_v))
        
        for k_idx, var_idx in enumerate(interest_indices):
            theta_k = beta[var_idx] + mu_prime_t[:, k_idx] / mu_t
            alpha_k = -omega * S_X_t[:, k_idx] / mu_t
            psi_k = theta_k + alpha_k * R
            
            psi_uncorr[:, k_idx] = psi_k.detach().cpu().numpy()
            
            grad_outputs = torch.ones_like(psi_k)
            grads = torch.autograd.grad(
                outputs=psi_k,
                inputs=V_t,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=(k_idx < n_interest - 1),
                allow_unused=False
            )[0]
            
            D_g_psi[:, k_idx, :] = grads.detach().cpu().numpy()
        
        return psi_uncorr, D_g_psi


    def _estimate_lambda_autodml(
        self,
        Z_train: np.ndarray,
        D_g_train: np.ndarray,
        Z_test: np.ndarray,
        lambda_params: Dict
    ) -> np.ndarray:
        """
        Estimate λ(Z) = E[D_g ψ | Z] via regression.
        
        This is the automatic DML approach: regress the pathwise derivative
        on the instruments to get the Riesz representer.
        
        Parameters
        ----------
        Z_train : ndarray, shape (n_train, k_z)
            Training instruments.
        D_g_train : ndarray, shape (n_train, n_interest, k_v)
            Pathwise derivatives on training set.
        Z_test : ndarray, shape (n_test, k_z)
            Test instruments.
        lambda_params : dict
            Parameters for lambda model architecture and fitting.
            
        Returns
        -------
        lambda_test : ndarray, shape (n_test, n_interest, k_v)
            Predicted λ(Z) on test set.
        """
        from .iv_nonparametric import NNModelRieszLambda
        
        n_test = Z_test.shape[0]
        n_interest = D_g_train.shape[1]
        k_v = D_g_train.shape[2]
        
        lambda_test = np.zeros((n_test, n_interest, k_v))
        
        # Fit separate λ model for each (interest variable, V dimension) pair
        # Alternatively, could fit a single model with multi-output
        for k_idx in range(n_interest):
            for v_idx in range(k_v):
                # Target: D_g ψ for this (k, v) pair
                target = D_g_train[:, k_idx, v_idx]
                
                # Fit λ(Z) = E[D_g ψ | Z]
                lambda_model = NNModelRieszLambda(**lambda_params.get('arch_params', {}))
                lambda_results = lambda_model.fit(
                    Z_train, target,
                    **lambda_params.get('fit_params', {})
                )
                
                # Predict on test set
                lambda_test[:, k_idx, v_idx] = lambda_results.predict(Z_test)
        
        return lambda_test


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
        Fold processing using automatic DML for λ estimation.

        Steps:
        1. Estimate first stage g(Z) on train
        2. Compute V_hat = X - g(Z) on full data (using train-fitted model)
        3. Estimate β, ρ via control function OLS on TRAIN only
        4. Estimate m(X, V) on train
        5. Compute μ(x), μ'(x) on test via MC integration
        6. Estimate ω(X, V) on train
        7. Estimate S_X(X) on train
        8. Compute uncorrected score AND its gradient ∂ψ/∂V on train
        9. Estimate λ(Z) = E[∂ψ/∂V | Z] via regression
        10. Construct final corrected moments and derivatives on test
        """
        k_x = self.exog.shape[1]
        k_v = k_x  # V has same dimension as X
        n_interest = len(interest_indices)

        # ===== Steps 1-2: First stage =====
        print("  Estimating first stage g(Z)...")
        g_results = self._estimate_first_stage(train_idx, first_stage_params)

        W_full = self.exog_control if self.exog_control is not None else None
        g_pred_full = self._predict_first_stage(g_results, self.instruments, W_full)
        V_hat_full = self.exog - g_pred_full

        # ===== Step 3: Control function OLS on TRAINING data only =====
        print("  Estimating control function OLS...")
        beta, delta, rho, ols_res = self._estimate_control_function_ols_fold(
            train_idx, V_hat_full, weight_fold
        )

        Y_transformed = self.endog * np.exp(-self.exog @ beta)
        
        # ===== Step 4: Estimate m(X, V) =====
        print("  Estimating nuisance m(X, V)...")
        X_train = self.exog[train_idx]
        V_train = V_hat_full[train_idx]
        Y_trans_train = Y_transformed[train_idx]
        
        m_input_train = np.column_stack([X_train, V_train])
        
        m_arch = m_params.get('arch_params', {}).copy()
        m_arch['input_size'] = m_input_train.shape[1]
        m_arch['output_size'] = 1
        if 'hidden_layers' not in m_arch:
            m_arch['hidden_layers'] = [256, 256, 256]
        if 'output_activation' not in m_arch:
            m_arch['output_activation'] = 'softplus'
        
        from .nonparametric import NNModelNuisance
        m_model = NNModelNuisance(variable_types={}, **m_arch)
        m_results = m_model.fit(m_input_train, Y_trans_train, **m_params.get('fit_params', {}))
        
        # ===== Step 5: Compute μ(x), μ'(x) on test =====
        print("  Computing marginal integration μ(x)...")
        X_test = self.exog[test_idx]
        V_test = V_hat_full[test_idx]
        
        mu_test, mu_prime_test = self._compute_mu_and_derivative(
            m_results, X_test, V_train, interest_indices
        )
        mu_test = np.maximum(mu_test, 1e-10)
        
        # ===== Step 6: Estimate ω(X, V) =====
        print("  Estimating density ratio ω(X, V)...")
        from .iv_nonparametric import NNModelDensityRatio
        omega_model = NNModelDensityRatio(**omega_params.get('arch_params', {}))
        omega_results = omega_model.fit(X_train, V_train, **omega_params.get('fit_params', {}))
        
        # ===== Step 7: Estimate S_X(X) =====
        print("  Estimating score S_X(X)...")
        from .nonparametric import NNModelDensity
        density_model = NNModelDensity(variable_types=self.variable_types, **density_params.get('arch_params', {}))
        density_results = density_model.fit(X_train, interest=interest_indices, **density_params.get('fit_params', {}))
        
        # ===== Step 8: Compute uncorrected score and gradient on TRAIN =====
        print("  Computing pathwise derivatives via autodiff...")
        
        # Need μ and μ' on training set for computing D_g
        mu_train, mu_prime_train = self._compute_mu_and_derivative(
            m_results, X_train, V_train, interest_indices
        )
        mu_train = np.maximum(mu_train, 1e-10)
        
        # Compute uncorrected score and its gradient w.r.t. V
        _, D_g_train = self._compute_uncorrected_score_with_grad(
            X_train, V_train, Y_trans_train,
            m_results, omega_results, density_results,
            mu_train, mu_prime_train,
            beta, interest_indices
        )
        
        # ===== Step 9: Estimate λ(Z) = E[D_g ψ | Z] =====
        print("  Estimating λ(Z) via automatic DML...")
        Z_train = self.instruments[train_idx]
        Z_test = self.instruments[test_idx]
        
        lambda_test = self._estimate_lambda_autodml(
            Z_train, D_g_train, Z_test, lambda_params
        )
        
        # ===== Step 10: Construct final moments on test =====
        print("  Constructing moments...")
        
        # Get test-set quantities
        Y_trans_test = Y_transformed[test_idx]
        omega_test = omega_results.predict(X_test, V_test)
        S_X_test = density_results.alpha_weight(X_test, interest_indices)
        
        m_input_test = np.column_stack([X_test, V_test])
        m_test = m_results.predict(m_input_test)
        m_test = np.maximum(m_test, 1e-10)
        R_test = Y_trans_test - m_test
        
        n_test = len(test_idx)
        fold_moments = np.zeros((n_test, n_interest))

        # Derivative matrix for sandwich variance: shape (n_test, n_interest, 2*n_interest)
        # First n_interest columns: ∂ψ/∂θ = -I
        # Second n_interest columns: ∂ψ/∂β (for joint inference)
        fold_derivative = np.zeros((n_test, n_interest, 2 * n_interest))
        identity = np.eye(n_interest)
        fold_derivative[:, :, :n_interest] = -identity  # ∂ψ/∂θ = -I

        fold_alpha_x = []
        fold_theta_x = []

        for k_idx, var_idx in enumerate(interest_indices):
            var_type = self.variable_types.get(var_idx, 'continuous')

            if var_type == 'continuous':
                # θ(x) = β_k + μ'_k(x)/μ(x)
                theta_x = beta[var_idx] + mu_prime_test[:, k_idx] / mu_test

                # α(x,v) = -ω(x,v) S_{X,k}(x) / μ(x)
                alpha_x = -omega_test * S_X_test[:, k_idx] / mu_test

                # λ correction: λ̃_k(Z) · V = Σ_j λ_{k,j}(Z) V_j
                lambda_k = lambda_test[:, k_idx, :]  # shape (n_test, k_v)
                lambda_correction = np.sum(lambda_k * V_test, axis=1)

                # Final moment: θ + α·R - λ̃·V
                fold_moments[:, k_idx] = theta_x + alpha_x * R_test - lambda_correction

                # Derivative w.r.t. β: ∂ψ/∂β_j = δ_{kj} - α·R·X_j (from Y_transformed dependence)
                # For the k-th moment, ∂ψ_k/∂β_j = I_{k=j} + α_k * Y_trans * X_j
                Y_trans_test = Y_transformed[test_idx]
                fold_derivative[:, k_idx, n_interest + k_idx] = 1.0  # ∂θ/∂β_k = 1
                for j_idx, j_var in enumerate(interest_indices):
                    fold_derivative[:, k_idx, n_interest + j_idx] -= alpha_x * Y_trans_test * X_test[:, j_var]

            elif var_type == 'binary':
                X_test_flip = X_test.copy()
                X_test_flip[:, var_idx] = 1 - X_test_flip[:, var_idx]

                m_input_flip = np.column_stack([X_test_flip, V_test])
                m_flip = m_results.predict(m_input_flip)
                m_flip = np.maximum(m_flip, 1e-10)

                mu_flip, _ = self._compute_mu_and_derivative(
                    m_results, X_test_flip, V_train, interest_indices
                )
                mu_flip = np.maximum(mu_flip, 1e-10)

                p_var = S_X_test[:, k_idx]

                theta_x = np.exp(beta[var_idx]) * (mu_flip / mu_test) - 1

                alpha_0 = (1 - X_test[:, var_idx]) * m_flip / (m_test**2 * (np.abs(p_var) + 1e-10))
                alpha_1 = X_test[:, var_idx] / ((np.abs(p_var) + 1e-10) * m_flip)
                alpha_x = np.exp(beta[var_idx]) * (alpha_1 - alpha_0)

                lambda_k = lambda_test[:, k_idx, :]
                lambda_correction = np.sum(lambda_k * V_test, axis=1)

                fold_moments[:, k_idx] = theta_x + alpha_x * R_test - lambda_correction

                # Binary derivative w.r.t. β
                Y_trans_test = Y_transformed[test_idx]
                fold_derivative[:, k_idx, n_interest + k_idx] = theta_x + 1 + alpha_x * (Y_trans_test - m_test)
                for j_idx, j_var in enumerate(interest_indices):
                    fold_derivative[:, k_idx, n_interest + j_idx] -= alpha_x * Y_trans_test * X_test[:, j_var]

            else:  # ordinal
                theta_x = np.zeros(n_test)
                alpha_x = np.zeros(n_test)
                fold_moments[:, k_idx] = 0

            fold_alpha_x.append(alpha_x)
            fold_theta_x.append(theta_x)

        return {
            'beta': beta,
            'delta': delta,
            'rho': rho,
            'g_results': g_results,
            'm_results': m_results,
            'm_model': m_model,
            'omega_results': omega_results,
            'omega_model': omega_model,
            'density_results': density_results,
            'density_model': density_model,
            'D_g_train': D_g_train,
            'moments': fold_moments,
            'derivative': fold_derivative,
            'alpha_x': np.column_stack(fold_alpha_x) if fold_alpha_x else np.array([]),
            'theta_x': np.column_stack(fold_theta_x) if fold_theta_x else np.array([]),
            'mu_test': mu_test,
            'mu_prime_test': mu_prime_test,
            'test_idx': test_idx,
            'V_hat_full': V_hat_full
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
            
        # Aggregate moments and derivatives
        n_interest = len(interest_indices)
        n_params = 2 * n_interest  # Elasticities + OLS beta for interest vars
        moments = np.zeros((self.nobs, n_params))
        derivative = np.zeros((self.nobs, n_params, n_params))
        fold_weights = np.zeros(self.nobs)

        # Aggregate V_hat from last fold (they should all be similar)
        V_hat_final = fold_results[-1]['V_hat_full']

        # Average beta across folds for final estimate
        beta_avg = np.mean([fold['beta'] for fold in fold_results], axis=0)
        rho_avg = np.mean([fold['rho'] for fold in fold_results], axis=0)
        delta_list = [fold['delta'] for fold in fold_results if fold['delta'] is not None]
        delta_avg = np.mean(delta_list, axis=0) if delta_list else None

        for fold in fold_results:
            test_idx = fold['test_idx']
            moments[test_idx, :n_interest] = fold['moments']
            derivative[test_idx, :n_interest, :n_params] = fold['derivative']
            fold_weights[test_idx] = (
                self.weights[test_idx] if self.weights is not None else 1.0
            )

        # Add OLS moment conditions for joint inference
        # OLS moments: ε_i * X_i where ε = log Y - β'X - ρ'V
        log_Y = np.log(self.endog)
        eps_ols = log_Y - self.exog @ beta_avg - V_hat_final @ rho_avg
        X_interest = self.exog[:, interest_indices]
        ols_moments = eps_ols[:, None] * X_interest
        moments[:, n_interest:] = ols_moments

        # OLS derivative: E[X_i X_i'] for the interest variables
        ols_derivative = X_interest[:, :, None] * X_interest[:, None, :]
        derivative[:, n_interest:, n_interest:] = ols_derivative

        # Compute estimates
        moment_means = np.average(moments[:, :n_interest], axis=0, weights=fold_weights)
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
            beta=beta_avg[interest_indices],
            delta=delta_avg,
            rho=rho_avg,
            ols_results=None,  # No single OLS result with cross-fitting
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
            derivative=derivative,
            fold_weights=fold_weights,
            V_hat=V_hat_final
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
