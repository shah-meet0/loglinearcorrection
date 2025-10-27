import numpy as np
import numpy.typing as npt


class NPModel:
    def __init__(self, variable_types: dict, **params):
        self.variable_types = variable_types
        self.params = params
        self.model = None

    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike,  **fit_params) -> "NPModelResults":
        pass


class NPModelResults:

    def __init__(self, model: NPModel, x: npt.ArrayLike, y: npt.ArrayLike, **metrics):
        self.model = model.model
        self.parent_model = model  # Store parent NPModel to access variable_types
        self.x = x
        self.y = y
        self.metrics = metrics


class NPModelResultsNuisance(NPModelResults):

    def __init__(self, model: NPModel, x: npt.ArrayLike, y: npt.ArrayLike, **metrics):
        super().__init__(model, x, y, **metrics)
        pass

    def predict(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        pass

    def derivative(self, x: npt.NDArray[np.float64], var_index: int) -> npt.NDArray[np.float64]:
        pass

    def second_derivative(self, x: npt.NDArray[np.float64], var_index: int) -> npt.NDArray[np.float64]:
        pass
    
    def predict_semi_elasticity(self, x: npt.ArrayLike, var_index: int) -> npt.NDArray[np.float64]:
        """
        Compute semi-elasticity m_k(x)/m(x) where m_k is the derivative w.r.t. variable k.
        
        Only valid for continuous variables. For binary or ordinal variables, use 
        predict with shifted x values to compute percentage changes.
        
        Parameters
        ----------
        x : array_like
            Points at which to evaluate semi-elasticity, shape (n_samples, n_features)
        var_index : int
            Index of variable for which to compute the semi-elasticity
            
        Returns
        -------
        ndarray
            Semi-elasticity values m_k(x)/m(x), shape (n_samples,)
            
        Raises
        ------
        ValueError
            If var_index corresponds to a binary or ordinal variable
        ZeroDivisionError
            If m(x) equals zero for any observation
            
        Notes
        -----
        The semi-elasticity represents the proportional change in m(x) with respect 
        to a unit change in variable k. It is computed as the ratio of the partial 
        derivative to the function value. This is only meaningful for continuous 
        variables where derivatives exist.
        
        For discrete variables (binary/ordinal), percentage changes should be computed
        as [m(x + Δ) - m(x)] / m(x) using the predict method with shifted inputs.
        """
        # Check if this is being called on a non-continuous variable
        if hasattr(self, 'parent_model') and hasattr(self.parent_model, 'variable_types'):
            var_type = self.parent_model.variable_types.get(var_index, 'continuous')
            if var_type in ['binary', 'ordinal']:
                raise ValueError(
                    f"predict_semi_elasticity called on {var_type} variable at index {var_index}. "
                    f"For {var_type} variables, compute percentage changes using predict() with shifted x values."
                )
        
        m_x = self.predict(x)
        m_k_x = self.derivative(x, var_index)
        return m_k_x / m_x


class NPModelResultsDensity(NPModelResults):

    def __init__(self, model: NPModel, x: npt.ArrayLike, y: npt.ArrayLike, **metrics):
        super().__init__(model, x, y, **metrics)
        pass

    def predict(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        pass


class NNModel(NPModel):
    def __init__(self, variable_types: dict, build_now=True, **params):
        import torch
        super().__init__(variable_types, **params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        if build_now:
            self.model = self._create_model().to(self.device)

    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike, *,
            epochs: int = 100,
            batch_size: int = 128,
            learning_rate: float = 1e-3,
            val_frac: float = 0.2,
            patience: int = 10,
            min_delta: float = 0.0,
            weight_decay: float = 1e-4,
            grad_clip_norm: float | None = None,
            verbose: bool = True, **kwargs) -> "NPModelResults":

        self._parse_fit_args(epcohs = epochs, batch_size=batch_size, learning_rate =learning_rate,
                             val_frac = val_frac, patience=patience, min_delta= min_delta,
                             weight_decay=weight_decay, grad_clip_norm=grad_clip_norm,
                             verbose=verbose)

        metrics = {'final_epoch_loss': float('nan'), 'val_loss': float('nan')}  # Placeholder for actual metrics

        return NPModelResults(self, x, y, **metrics)

    def training_step(self, batch):
        pass

    def validation_step(self, batch):
        pass

    def _create_model(self):
        config = self._parse_params()
        return self._create_model_from_config(config)

    def _create_model_from_config(self, config):
        from loglinearcorrection.neural_network_models import FeedForwardNNModel
        return FeedForwardNNModel(config)


    def _parse_params(self):
        """
        Parse, default-fill, and validate model hyperparameters using self.params.

        Behavior
        --------
        Uses self.params (a dict) to configure the model. Missing values are filled with defaults.
        The method validates data types and value ranges, then returns a clean parameter dict.

        Required keys after parsing:
            - input_size : int > 0
            - hidden_layers : list[int] (each > 0)
            - output_size : int > 0

        Optional keys with defaults:
            - activation : str in {'relu','tanh','sigmoid','leaky_relu'}            [default: 'relu']
            - output_activation : str or None in {'identity','relu','tanh','sigmoid','softmax','log_softmax'}  [default: 'identity']
            - learning_rate : float > 0                                            [default: 1e-3]
            - dropout : float in [0,1)                                             [default: 0.0]
            - bias : bool                                                          [default: True]
            - weight_init : str in {'xavier_uniform','xavier_normal','kaiming_uniform','kaiming_normal','default'} [default: 'default']

        Returns
        -------
        dict
            Validated and default-filled hyperparameter dictionary.

        Raises
        ------
        ValueError
            If required keys are missing or invalid.
        """

        if not hasattr(self, "params") or not isinstance(self.params, dict):
            raise ValueError("self.params must be a dictionary of hyperparameters")

        defaults = {
            "activation": "relu",
            "output_activation": "identity",
            "learning_rate": 1e-3,
            "dropout": 0.0,
            "bias": True,
            "weight_init": "default",
        }

        required = ["input_size", "hidden_layers", "output_size"]

        parsed = {**defaults, **self.params} # This merges the two dictionaries, giving precedence to self.params

        for key in required:
            if key not in parsed:
                raise ValueError(f"Missing required parameter: '{key}'")

        if not isinstance(parsed["input_size"], int) or parsed["input_size"] <= 0:
            raise ValueError("input_size must be a positive integer")

        if not isinstance(parsed["output_size"], int) or parsed["output_size"] <= 0:
            raise ValueError("output_size must be a positive integer")

        if not isinstance(parsed["hidden_layers"], list) or not all(
                isinstance(n, int) and n > 0 for n in parsed["hidden_layers"]):
            raise ValueError("hidden_layers must be a list of positive integers")

        if parsed["activation"] not in {"relu", "tanh", "sigmoid", "leaky_relu"}:
            raise ValueError("Unsupported activation function")

        if parsed["output_activation"] not in {"identity", "relu", "tanh", "sigmoid", "softmax", "log_softmax"}:
            raise ValueError("Unsupported output activation function")

        if not (0.0 <= parsed["dropout"] < 1.0):
            raise ValueError("dropout must be between 0 and 1")

        if parsed["learning_rate"] <= 0:
            raise ValueError("learning_rate must be positive")

        if parsed["weight_init"] not in {"xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal",
                                         "default"}:
            raise ValueError("Unsupported weight initialization method")

        return parsed

    def _parse_fit_args(
        self,
        epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        val_frac: float = 0.2,
        patience: int = 10,
        min_delta: float = 0.0,
        weight_decay: float = 0.0,
        grad_clip_norm: float | None = None,
        verbose: bool = True,
        **kwargs
    ) -> dict[str, object]:
        if epochs <= 0: raise ValueError("epochs must be > 0")
        if batch_size <= 0: raise ValueError("batch_size must be > 0")
        if learning_rate <= 0: raise ValueError("learning_rate must be > 0")
        if not (0.0 <= val_frac < 1.0): raise ValueError("val_frac in [0,1)")
        if patience < 0: raise ValueError("patience must be >= 0")
        if min_delta < 0: raise ValueError("min_delta must be >= 0")
        if weight_decay < 0: raise ValueError("weight_decay must be >= 0")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0 if set")
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "val_frac": val_frac,
            "patience": patience,
            "min_delta": min_delta,
            "weight_decay": weight_decay,
            "grad_clip_norm": grad_clip_norm,
            "verbose": verbose,
        }



class NNModelNuisance(NNModel):

    def __init__(self, variable_types: dict, **params):
        super().__init__(variable_types, build_now=True, **params)

    def fit(self, X, y, *,
            epochs: int = 100,
            batch_size: int = 128,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            val_frac: float = 0.2,
            patience: int = 10,
            min_delta: float = 0.0,
            grad_clip_norm: float | None = None,
            num_workers: int = 0,
            shuffle: bool = True,
            verbose: bool = True,
            train_idx=None,
            val_idx=None,
    ) -> "NNModelResults":

        import torch
        from torch.utils.data import DataLoader, TensorDataset, Subset

        # use self.device from base class
        pin = (self.device.type == "cuda")
        self.model.to(self.device)

        # tensors
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        # split (fixed if indices provided)
        N = X.size(0)
        if train_idx is None or val_idx is None:
            n_val = int(N * val_frac)
            perm = torch.randperm(N)
            val_idx, train_idx = (
                (perm[:n_val], perm[n_val:]) if n_val > 0 else (torch.empty(0, dtype=torch.long), torch.arange(N))
            )

        ds = TensorDataset(X, y)
        dl_tr = DataLoader(
            Subset(ds, train_idx),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=False,
        )
        dl_va = (
            DataLoader(
                Subset(ds, val_idx),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin,
                drop_last=False,
            )
            if len(val_idx)
            else None
        )

        # optimizer
        opt = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_loss = float("inf")
        best_state = None
        bad = 0
        log_every = max(1, epochs // 10)

        for ep in range(1, epochs + 1):
            # ---- train epoch
            self.model.train()
            tr_sum = tr_cnt = 0
            for batch in dl_tr:
                opt.zero_grad(set_to_none=True)
                loss = self.training_step(batch)
                loss.backward()
                if grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                opt.step()
                bs = batch[0].size(0)
                tr_sum += float(loss.detach()) * bs
                tr_cnt += bs
            tr_loss = tr_sum / max(tr_cnt, 1)

            # ---- validate
            if dl_va is not None:
                self.model.eval()
                va_sum = va_cnt = 0
                with torch.no_grad():
                    for batch in dl_va:
                        l = self.validation_step(batch)
                        bs = batch[0].size(0)
                        va_sum += float(l) * bs
                        va_cnt += bs
                va_loss = va_sum / max(va_cnt, 1)

                improved = va_loss + min_delta < best_loss
                if improved:
                    best_loss = va_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1

                if verbose and (ep % log_every == 0 or ep == 1 or ep == epochs):
                    print(
                        f"[Epoch {ep:>3}/{epochs}] train={tr_loss:.6f} val={va_loss:.6f}{' *' if improved else ''}")

                if bad >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {ep} (best val={best_loss:.6f}).")
                    break
            else:
                if verbose and (ep % log_every == 0 or ep == 1 or ep == epochs):
                    print(f"[Epoch {ep:>3}/{epochs}] train={tr_loss:.6f}")

        # restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return NNModelNuisanceResults(
            self,
            X.cpu().numpy(),
            y.cpu().numpy(),
            **{"val_loss": best_loss if best_state is not None else None}
        )

    # ---------- PER-BATCH LOSS: TRAIN ----------
    def training_step(self, batch):
        """
        One training step for supervised nuisance model (MSE).
        Expects batch = (xb, yb). Returns scalar loss tensor.
        """
        import torch
        import torch.nn.functional as F
        xb, yb = batch
        xb = xb.to(self.device, non_blocking=(self.device.type == "cuda"))
        yb = yb.to(self.device, non_blocking=(self.device.type == "cuda"))
        pred = self.model(xb)
        if pred.shape != yb.shape:
            pred = pred.view_as(yb)
        return F.mse_loss(pred, yb)

    # ---------- PER-BATCH LOSS: VAL ----------
    def validation_step(self, batch):
        """
        Validation step. Identical to training_step here, but kept separate
        to allow different metrics or no-regularizer evaluation later.
        """
        import torch
        with torch.inference_mode():
            return self.training_step(batch)


class NNModelDensity(NNModel):
    def __init__(self, variable_types: dict, **params):
        super().__init__(variable_types, build_now=False, **params)


    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike, **fit_params) -> "NNModelDensityResults":
        return super().fit(x, y, **fit_params)



class NNModelScoreResults(NPModelResultsDensity):
    def __init__(self, model: NNModelScore, x: npt.ArrayLike, y: npt.ArrayLike):
        super().__init__(model, x, y)
        pass

class NNModelNuisanceResults(NPModelResultsNuisance):
    def __init__(self, model: NNModelNuisance, x: npt.ArrayLike, y: npt.ArrayLike, **metrics):
        super().__init__(model, x, y, **metrics)
        pass