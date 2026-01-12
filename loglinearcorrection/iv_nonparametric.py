"""
IV-specific nonparametric models for IVDRNO estimator.

Contains:
- NNModelDensityRatio: Classification-based density ratio ω(x,v) = f_V(v)/f_{V|X}(v|x)
- NNModelRieszLambda: Automatic DML estimation of λ(Z) correction
- NNModelPropensity: Propensity score π(s) = P(X_1=1|S) for binary treatment
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, List, Tuple

from .nonparametric import NNModel, NPModelResults


class NNModelDensityRatio(NNModel):
    """
    Classification-based density ratio estimation.
    
    Estimates ω(x,v) = f_V(v) / f_{V|X}(v|x) using a binary classifier
    that distinguishes joint samples (X,V) from product-of-marginals samples.
    
    The key insight is that for classification between:
        - Class 1: (X_i, V_i) ~ f_{X,V}  (joint distribution)
        - Class 0: (X_i, V_j) ~ f_X × f_V  (product of marginals, j ≠ i)
    
    The Bayes-optimal classifier satisfies:
        P(class=1 | x,v) / P(class=0 | x,v) = f_{X,V}(x,v) / (f_X(x) f_V(v))
    
    And we have:
        ω(x,v) = f_V(v) / f_{V|X}(v|x) 
               = f_V(v) f_X(x) / f_{X,V}(x,v)
               = 1 / r(x,v)
    
    where r(x,v) = f_{X,V}(x,v) / (f_X(x) f_V(v)) is the density ratio
    estimated by the classifier.
    
    Parameters
    ----------
    hidden_layers : list of int, default=[256, 256]
        Hidden layer sizes for the classifier.
    activation : str, default='leaky_relu'
        Activation function.
    dropout : float, default=0.1
        Dropout rate.
    **params : dict
        Additional parameters passed to base NNModel.
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = None,
        activation: str = 'leaky_relu',
        dropout: float = 0.1,
        **params
    ):
        # Store architecture params before calling super
        self._arch_config = {
            'hidden_layers': hidden_layers or [256, 256],
            'activation': activation,
            'dropout': dropout,
            'output_activation': 'identity',  # Will apply sigmoid in loss
            'bias': True,
            'weight_init': 'default'
        }
        
        # Merge with any passed params
        for key, val in params.items():
            if key in self._arch_config:
                self._arch_config[key] = val
                
        # Don't build model yet - need input size from fit
        super().__init__(variable_types={}, build_now=False, **self._arch_config)
        
    def fit(
        self,
        X: npt.ArrayLike,
        V: npt.ArrayLike,
        *,
        n_permutations: int = 5,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        val_frac: float = 0.2,
        patience: int = 15,
        verbose: bool = True,
        **kwargs
    ) -> 'NNModelDensityRatioResults':
        """
        Fit the density ratio classifier.
        
        Creates a binary classification dataset:
        - Positive class (y=1): Original (X_i, V_i) pairs
        - Negative class (y=0): Permuted (X_i, V_{π(i)}) pairs
        
        Parameters
        ----------
        X : array_like, shape (n, k_x)
            Treatment/covariate values.
        V : array_like, shape (n,) or (n, 1)
            Control function residuals.
        n_permutations : int, default=5
            Number of random permutations for negative samples.
            More permutations = more negative samples = better calibration.
        epochs : int, default=100
            Training epochs.
        batch_size : int, default=256
            Batch size.
        learning_rate : float, default=1e-3
            Learning rate.
        weight_decay : float, default=1e-4
            L2 regularization.
        val_frac : float, default=0.2
            Validation fraction for early stopping.
        patience : int, default=15
            Early stopping patience.
        verbose : bool, default=True
            Print training progress.
            
        Returns
        -------
        NNModelDensityRatioResults
            Fitted model with prediction methods.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        X = np.asarray(X, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        if V.ndim == 1:
            V = V.reshape(-1, 1)
            
        n = X.shape[0]
        k_x = X.shape[1]
        k_v = V.shape[1]
        
        # Store for later
        self.k_x = k_x
        self.k_v = k_v
        
        # Create classification dataset
        # Positive: (X, V) joint pairs
        X_pos = X
        V_pos = V
        y_pos = np.ones(n, dtype=np.float32)
        
        # Negative: (X, V_permuted) - multiple permutations
        X_neg_list = []
        V_neg_list = []
        
        for _ in range(n_permutations):
            perm = np.random.permutation(n)
            X_neg_list.append(X)
            V_neg_list.append(V[perm])
            
        X_neg = np.vstack(X_neg_list)
        V_neg = np.vstack(V_neg_list)
        y_neg = np.zeros(n * n_permutations, dtype=np.float32)
        
        # Combine
        X_all = np.vstack([X_pos, X_neg])
        V_all = np.vstack([V_pos, V_neg])
        y_all = np.concatenate([y_pos, y_neg])
        
        # Input is [X, V]
        input_all = np.column_stack([X_all, V_all])
        input_size = input_all.shape[1]
        
        # Build model
        self._arch_config['input_size'] = input_size
        self._arch_config['output_size'] = 1
        self.model = self._create_model().to(self.device)
        
        # Split train/val
        n_total = len(y_all)
        n_val = int(n_total * val_frac)
        perm = np.random.permutation(n_total)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        
        # Create dataloaders
        X_tr = torch.as_tensor(input_all[train_idx], dtype=torch.float32)
        y_tr = torch.as_tensor(y_all[train_idx], dtype=torch.float32).unsqueeze(-1)
        X_va = torch.as_tensor(input_all[val_idx], dtype=torch.float32)
        y_va = torch.as_tensor(y_all[val_idx], dtype=torch.float32).unsqueeze(-1)
        
        ds_tr = TensorDataset(X_tr, y_tr)
        ds_va = TensorDataset(X_va, y_va)
        
        pin = (self.device.type == "cuda")
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, pin_memory=pin)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, pin_memory=pin)
        
        # Loss and optimizer
        # Use class-weighted BCE to handle imbalanced classes
        pos_weight = torch.tensor([n_permutations], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Training loop
        best_loss = float('inf')
        best_state = None
        bad = 0
        log_every = max(1, epochs // 10)
        
        for ep in range(1, epochs + 1):
            # Train
            self.model.train()
            tr_sum = tr_cnt = 0
            for xb, yb in dl_tr:
                xb = xb.to(self.device, non_blocking=pin)
                yb = yb.to(self.device, non_blocking=pin)
                
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                
                bs = xb.size(0)
                tr_sum += float(loss.detach()) * bs
                tr_cnt += bs
            tr_loss = tr_sum / max(tr_cnt, 1)
            
            # Validate
            self.model.eval()
            va_sum = va_cnt = 0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb = xb.to(self.device, non_blocking=pin)
                    yb = yb.to(self.device, non_blocking=pin)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    bs = xb.size(0)
                    va_sum += float(loss) * bs
                    va_cnt += bs
            va_loss = va_sum / max(va_cnt, 1)
            
            improved = va_loss < best_loss - 1e-6
            if improved:
                best_loss = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                
            if verbose and (ep % log_every == 0 or ep == 1 or ep == epochs):
                print(f"[DensityRatio ep {ep:>3}/{epochs}] train={tr_loss:.4f} val={va_loss:.4f}{' *' if improved else ''}")
                
            if bad >= patience:
                if verbose:
                    print(f"Early stopping at epoch {ep}")
                break
                
        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        return NNModelDensityRatioResults(self, val_loss=best_loss)
    
    def _parse_params(self) -> dict:
        """Return stored architecture config."""
        return self._arch_config


class NNModelDensityRatioResults(NPModelResults):
    """
    Results class for density ratio estimation.
    
    Provides methods to compute ω(x,v) = f_V(v) / f_{V|X}(v|x).
    """
    
    def __init__(self, model: NNModelDensityRatio, **metrics):
        super().__init__(model, **metrics)
        self.device = model.device
        self.k_x = model.k_x
        self.k_v = model.k_v
        
    def predict(
        self, 
        X: npt.ArrayLike, 
        V: npt.ArrayLike,
        clip_range: Tuple[float, float] = (0.01, 100.0)
    ) -> npt.NDArray[np.float64]:
        """
        Compute density ratio ω(x,v) = f_V(v) / f_{V|X}(v|x).
        
        Parameters
        ----------
        X : array_like, shape (n, k_x)
            Treatment/covariate values.
        V : array_like, shape (n,) or (n, 1)
            Control function residuals.
        clip_range : tuple, default=(0.01, 100.0)
            Range to clip omega for numerical stability.
            
        Returns
        -------
        omega : ndarray, shape (n,)
            Density ratio values.
        """
        import torch
        
        X = np.asarray(X, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        if V.ndim == 1:
            V = V.reshape(-1, 1)
            
        # Input is [X, V]
        input_arr = np.column_stack([X, V])
        input_t = torch.as_tensor(input_arr, dtype=torch.float32, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_t)
            # P(joint | x,v) = sigmoid(logits)
            # r(x,v) = P(joint) / P(marginal) = p / (1-p)
            p = torch.sigmoid(logits).squeeze(-1)
            
            # Avoid division by zero
            p = torch.clamp(p, min=1e-6, max=1 - 1e-6)
            
            # Density ratio f_{X,V} / (f_X f_V)
            r = p / (1 - p)
            
            # omega = 1/r
            omega = 1.0 / r
            
        omega_np = omega.cpu().numpy()
        
        # Clip for stability
        omega_np = np.clip(omega_np, clip_range[0], clip_range[1])
        
        return omega_np.astype(np.float64)
    
    def predict_log_ratio(
        self,
        X: npt.ArrayLike,
        V: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """
        Compute log density ratio log ω(x,v).
        
        More numerically stable for extreme values.
        """
        import torch
        
        X = np.asarray(X, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        if V.ndim == 1:
            V = V.reshape(-1, 1)
            
        input_arr = np.column_stack([X, V])
        input_t = torch.as_tensor(input_arr, dtype=torch.float32, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_t).squeeze(-1)
            # log r = log(p / (1-p)) = logit(p) = logits (by definition)
            # log omega = -log r = -logits
            log_omega = -logits
            
        return log_omega.cpu().numpy().astype(np.float64)
    
    def predict_marginal_prob(
        self,
        X: npt.ArrayLike,
        V: npt.ArrayLike,
        var_idx: int
    ) -> npt.NDArray[np.float64]:
        """
        Estimate marginal probability P(X_k = observed | other variables).
        
        This is useful for binary treatment variables.
        Currently returns omega as a proxy; proper implementation
        would need separate handling for discrete variables.
        """
        # For binary variables, this should return P(X_k | X_{-k}, V)
        # For now, return omega as a proxy
        return self.predict(X, V)


class NNModelRieszLambda(NNModel):
    """
    Automatic DML estimation of λ(Z) correction term.
    
    Estimates λ(Z) = E[D_g ψ_0 | Z] where D_g ψ_0 is the pathwise
    derivative of the score with respect to the first stage g.
    
    This is implemented as a simple regression: given computed pathwise
    derivatives on training data, regress them on instruments Z to
    estimate the conditional expectation.
    
    Parameters
    ----------
    hidden_layers : list of int, default=[128, 128]
        Hidden layer sizes.
    **params : dict
        Additional parameters passed to base NNModel.
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = None,
        activation: str = 'relu',
        dropout: float = 0.0,
        **params
    ):
        self._arch_config = {
            'hidden_layers': hidden_layers or [128, 128],
            'activation': activation,
            'dropout': dropout,
            'output_activation': 'identity',
            'bias': True,
            'weight_init': 'default'
        }
        
        for key, val in params.items():
            if key in self._arch_config:
                self._arch_config[key] = val
                
        super().__init__(variable_types={}, build_now=False, **self._arch_config)
        
    def fit(
        self,
        Z: npt.ArrayLike,
        pathwise_deriv: npt.ArrayLike,
        *,
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        val_frac: float = 0.2,
        patience: int = 10,
        verbose: bool = True,
        **kwargs
    ) -> 'NNModelRieszLambdaResults':
        """
        Fit λ(Z) = E[D_g ψ_0 | Z] via regression.
        
        Parameters
        ----------
        Z : array_like, shape (n, k_z)
            Instrument values.
        pathwise_deriv : array_like, shape (n,) or (n, 1)
            Computed pathwise derivatives D_g ψ_0.
        epochs : int, default=50
            Training epochs.
        batch_size : int, default=128
            Batch size.
        learning_rate : float, default=1e-3
            Learning rate.
        weight_decay : float, default=1e-4
            L2 regularization.
        val_frac : float, default=0.2
            Validation fraction.
        patience : int, default=10
            Early stopping patience.
        verbose : bool, default=True
            Print training progress.
            
        Returns
        -------
        NNModelRieszLambdaResults
            Fitted model with prediction methods.
        """
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        
        Z = np.asarray(Z, dtype=np.float32)
        y = np.asarray(pathwise_deriv, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
            
        n = Z.shape[0]
        input_size = Z.shape[1]
        
        # Build model
        self._arch_config['input_size'] = input_size
        self._arch_config['output_size'] = y.shape[1]
        self.model = self._create_model().to(self.device)
        
        # Split
        n_val = int(n * val_frac)
        perm = np.random.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        
        # Dataloaders
        Z_tr = torch.as_tensor(Z[train_idx], dtype=torch.float32)
        y_tr = torch.as_tensor(y[train_idx], dtype=torch.float32)
        Z_va = torch.as_tensor(Z[val_idx], dtype=torch.float32)
        y_va = torch.as_tensor(y[val_idx], dtype=torch.float32)
        
        ds_tr = TensorDataset(Z_tr, y_tr)
        ds_va = TensorDataset(Z_va, y_va) if n_val > 0 else None
        
        pin = (self.device.type == "cuda")
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, pin_memory=pin)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, pin_memory=pin) if ds_va else None
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training
        best_loss = float('inf')
        best_state = None
        bad = 0
        log_every = max(1, epochs // 10)
        
        for ep in range(1, epochs + 1):
            self.model.train()
            tr_sum = tr_cnt = 0
            for zb, yb in dl_tr:
                zb = zb.to(self.device, non_blocking=pin)
                yb = yb.to(self.device, non_blocking=pin)
                
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(zb)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                optimizer.step()
                
                bs = zb.size(0)
                tr_sum += float(loss.detach()) * bs
                tr_cnt += bs
            tr_loss = tr_sum / max(tr_cnt, 1)
            
            # Validate
            if dl_va is not None:
                self.model.eval()
                va_sum = va_cnt = 0
                with torch.no_grad():
                    for zb, yb in dl_va:
                        zb = zb.to(self.device, non_blocking=pin)
                        yb = yb.to(self.device, non_blocking=pin)
                        pred = self.model(zb)
                        loss = F.mse_loss(pred, yb)
                        bs = zb.size(0)
                        va_sum += float(loss) * bs
                        va_cnt += bs
                va_loss = va_sum / max(va_cnt, 1)
                
                improved = va_loss < best_loss - 1e-6
                if improved:
                    best_loss = va_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    
                if verbose and (ep % log_every == 0 or ep == 1 or ep == epochs):
                    print(f"[Lambda ep {ep:>3}/{epochs}] train={tr_loss:.6f} val={va_loss:.6f}{' *' if improved else ''}")
                    
                if bad >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {ep}")
                    break
            else:
                if verbose and (ep % log_every == 0 or ep == 1 or ep == epochs):
                    print(f"[Lambda ep {ep:>3}/{epochs}] train={tr_loss:.6f}")
                    
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        return NNModelRieszLambdaResults(self, val_loss=best_loss)
    
    def _parse_params(self) -> dict:
        return self._arch_config


class NNModelRieszLambdaResults(NPModelResults):
    """Results class for lambda estimation."""
    
    def __init__(self, model: NNModelRieszLambda, **metrics):
        super().__init__(model, **metrics)
        self.device = model.device
        
    def predict(self, Z: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Predict λ(Z).
        
        Parameters
        ----------
        Z : array_like, shape (n, k_z)
            Instrument values.
            
        Returns
        -------
        lambda_pred : ndarray, shape (n,)
            Predicted λ(Z) values.
        """
        import torch
        
        Z = np.asarray(Z, dtype=np.float32)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
            
        Z_t = torch.as_tensor(Z, dtype=torch.float32, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(Z_t)
            
        pred_np = pred.squeeze(-1).cpu().numpy()
        return pred_np.astype(np.float64)


class NNModelPropensity(NNModel):
    """
    Propensity score estimation for binary treatment in IV setting.

    Estimates π(s) = P(X_1 = 1 | S = s) where S = (X_{-1}, W, V) includes
    the control function residual V. This conditioning on V is required
    for Neyman orthogonality in the IV-DRNO binary treatment case.

    Unlike standard propensity scores that only condition on observed
    covariates, this conditions on the estimated first-stage residual,
    which captures the endogeneity structure.

    Parameters
    ----------
    hidden_layers : list of int, default=[128, 128]
        Hidden layer sizes for the classifier.
    activation : str, default='leaky_relu'
        Activation function.
    dropout : float, default=0.1
        Dropout rate.
    **params : dict
        Additional parameters passed to base NNModel.
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        activation: str = 'leaky_relu',
        dropout: float = 0.1,
        **params
    ):
        self._arch_config = {
            'hidden_layers': hidden_layers or [128, 128],
            'activation': activation,
            'dropout': dropout,
            'output_activation': 'identity',  # Sigmoid applied in loss
            'bias': True,
            'weight_init': 'default'
        }

        for key, val in params.items():
            if key in self._arch_config:
                self._arch_config[key] = val

        super().__init__(variable_types={}, build_now=False, **self._arch_config)

    def fit(
        self,
        S: npt.ArrayLike,
        X1: npt.ArrayLike,
        *,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        val_frac: float = 0.2,
        patience: int = 15,
        verbose: bool = True,
        **kwargs
    ) -> 'NNModelPropensityResults':
        """
        Fit the propensity score model.

        Parameters
        ----------
        S : array_like, shape (n, dim_S)
            Conditioning variables S = (X_{-1}, W, V).
        X1 : array_like, shape (n,)
            Binary treatment indicator (0 or 1).
        epochs : int, default=100
            Training epochs.
        batch_size : int, default=256
            Batch size.
        learning_rate : float, default=1e-3
            Learning rate.
        weight_decay : float, default=1e-4
            L2 regularization.
        val_frac : float, default=0.2
            Validation fraction for early stopping.
        patience : int, default=15
            Early stopping patience.
        verbose : bool, default=True
            Print training progress.

        Returns
        -------
        NNModelPropensityResults
            Fitted model with prediction methods.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        S = np.asarray(S, dtype=np.float32)
        X1 = np.asarray(X1, dtype=np.float32).ravel()

        if S.ndim == 1:
            S = S.reshape(-1, 1)

        n = S.shape[0]
        input_size = S.shape[1]

        # Validate binary treatment
        unique_vals = np.unique(X1)
        if not np.allclose(unique_vals, [0, 1]) and not np.allclose(unique_vals, [0]) and not np.allclose(unique_vals, [1]):
            raise ValueError(f"X1 must be binary (0/1), got unique values: {unique_vals}")

        # Build model
        self._arch_config['input_size'] = input_size
        self._arch_config['output_size'] = 1
        self.model = self._create_model().to(self.device)

        # Split train/val
        n_val = int(n * val_frac)
        perm = np.random.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        # Create dataloaders
        S_tr = torch.as_tensor(S[train_idx], dtype=torch.float32)
        y_tr = torch.as_tensor(X1[train_idx], dtype=torch.float32).unsqueeze(-1)
        S_va = torch.as_tensor(S[val_idx], dtype=torch.float32)
        y_va = torch.as_tensor(X1[val_idx], dtype=torch.float32).unsqueeze(-1)

        ds_tr = TensorDataset(S_tr, y_tr)
        ds_va = TensorDataset(S_va, y_va)

        pin = (self.device.type == "cuda")
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, pin_memory=pin)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, pin_memory=pin)

        # Loss and optimizer
        # Use class-weighted BCE if imbalanced
        n_pos = X1[train_idx].sum()
        n_neg = len(train_idx) - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=self.device)
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float32, device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Training loop
        best_loss = float('inf')
        best_state = None
        bad = 0
        log_every = max(1, epochs // 10)

        for ep in range(1, epochs + 1):
            # Train
            self.model.train()
            tr_sum = tr_cnt = 0
            for sb, yb in dl_tr:
                sb = sb.to(self.device, non_blocking=pin)
                yb = yb.to(self.device, non_blocking=pin)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(sb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                bs = sb.size(0)
                tr_sum += float(loss.detach()) * bs
                tr_cnt += bs
            tr_loss = tr_sum / max(tr_cnt, 1)

            # Validate
            self.model.eval()
            va_sum = va_cnt = 0
            with torch.no_grad():
                for sb, yb in dl_va:
                    sb = sb.to(self.device, non_blocking=pin)
                    yb = yb.to(self.device, non_blocking=pin)
                    logits = self.model(sb)
                    loss = criterion(logits, yb)
                    bs = sb.size(0)
                    va_sum += float(loss) * bs
                    va_cnt += bs
            va_loss = va_sum / max(va_cnt, 1)

            improved = va_loss < best_loss - 1e-6
            if improved:
                best_loss = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if verbose and (ep % log_every == 0 or ep == 1 or ep == epochs):
                print(f"[Propensity ep {ep:>3}/{epochs}] train={tr_loss:.4f} val={va_loss:.4f}{' *' if improved else ''}")

            if bad >= patience:
                if verbose:
                    print(f"Early stopping at epoch {ep}")
                break

        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return NNModelPropensityResults(self, val_loss=best_loss)

    def _parse_params(self) -> dict:
        return self._arch_config


class NNModelPropensityResults(NPModelResults):
    """
    Results class for propensity score estimation.

    Provides methods to compute π(s) = P(X_1 = 1 | S = s).
    """

    def __init__(self, model: NNModelPropensity, **metrics):
        super().__init__(model, **metrics)
        self.device = model.device

    def predict(
        self,
        S: npt.ArrayLike,
        clip: Tuple[float, float] = (0.01, 0.99)
    ) -> npt.NDArray[np.float64]:
        """
        Compute propensity score π(s) = P(X_1 = 1 | S = s).

        Parameters
        ----------
        S : array_like, shape (n, dim_S)
            Conditioning variables S = (X_{-1}, W, V).
        clip : tuple, default=(0.01, 0.99)
            Range to clip probabilities for numerical stability.

        Returns
        -------
        pi : ndarray, shape (n,)
            Propensity scores.
        """
        import torch

        S = np.asarray(S, dtype=np.float32)
        if S.ndim == 1:
            S = S.reshape(-1, 1)

        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(S_t)
            pi = torch.sigmoid(logits).squeeze(-1)

        pi_np = pi.cpu().numpy()

        # Clip for stability
        pi_np = np.clip(pi_np, clip[0], clip[1])

        return pi_np.astype(np.float64)

    def predict_logit(
        self,
        S: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """
        Compute log-odds logit(π(s)) = log(π/(1-π)).

        More numerically stable for extreme probabilities.
        """
        import torch

        S = np.asarray(S, dtype=np.float32)
        if S.ndim == 1:
            S = S.reshape(-1, 1)

        S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(S_t).squeeze(-1)

        return logits.cpu().numpy().astype(np.float64)
