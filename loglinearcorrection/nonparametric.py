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

    def __init__(self, model: NPModel, **metrics):
        self.model = model.model
        self.variable_types = model.variable_types
        self.metrics = metrics


class NPModelResultsNuisance(NPModelResults):

    def __init__(self, model: NPModel, **metrics):
        super().__init__(model, **metrics)
        pass

    def predict(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        pass

    def derivative(self, x: npt.NDArray[np.float64], var_index: list[int]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pass

    
    # def predict_semi_elasticity(self, x: npt.ArrayLike, var_index: int) -> npt.NDArray[np.float64]:
    #     """
    #     Compute semi-elasticity m_k(x)/m(x) where m_k is the derivative w.r.t. variable k.
    #
    #     Only valid for continuous variables. For binary or ordinal variables, use
    #     predict with shifted x values to compute percentage changes.
    #
    #     Parameters
    #     ----------
    #     x : array_like
    #         Points at which to evaluate semi-elasticity, shape (n_samples, n_features)
    #     var_index : int
    #         Index of variable for which to compute the semi-elasticity
    #
    #     Returns
    #     -------
    #     ndarray
    #         Semi-elasticity values m_k(x)/m(x), shape (n_samples,)
    #
    #     Raises
    #     ------
    #     ValueError
    #         If var_index corresponds to a binary or ordinal variable
    #     ZeroDivisionError
    #         If m(x) equals zero for any observation
    #
    #     Notes
    #     -----
    #     The semi-elasticity represents the proportional change in m(x) with respect
    #     to a unit change in variable k. It is computed as the ratio of the partial
    #     derivative to the function value. This is only meaningful for continuous
    #     variables where derivatives exist.
    #
    #     For discrete variables (binary/ordinal), percentage changes should be computed
    #     as [m(x + Δ) - m(x)] / m(x) using the predict method with shifted inputs.
    #     """
    #     # Check if this is being called on a non-continuous variable
    #     if hasattr(self, 'parent_model') and hasattr(self.parent_model, 'variable_types'):
    #         var_type = self.parent_model.variable_types.get(var_index, 'continuous')
    #         if var_type in ['binary', 'ordinal']:
    #             raise ValueError(
    #                 f"predict_semi_elasticity called on {var_type} variable at index {var_index}. "
    #                 f"For {var_type} variables, compute percentage changes using predict() with shifted x values."
    #             )
    #
    #     m_x = self.predict(x)
    #     m_k_x = self.derivative(x, var_index)
    #     return m_k_x / m_x


class NPModelResultsDensity(NPModelResults):

    def __init__(self, model: NPModel, **metrics):
        super().__init__(model, **metrics)
        pass

    def predict(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        # should return f'(x)/f(x) for all continuous variables of interest
        # not sure about discrete variables here yet
        pass


class NNModel(NPModel):
    """
    Base neural network model wrapper for nuisance and density models.
    Handles device selection, model construction, parameter parsing,
    and training configuration. Subclasses implement the actual
    training and loss functions.
    """
    def __init__(self, variable_types: dict, build_now=True, **params):
        """
                Initialize the NNModel.

                Parameters
                ----------
                variable_types : dict
                    Mapping from variable index → {'continuous','binary','ordinal'}.
                    Used by subclasses to interpret model outputs.
                build_now : bool, default=True
                    If True, the network architecture is constructed immediately.
                    If False, creation is deferred (used in two-head density model).
                **params : dict
                    Raw hyperparameter dictionary passed to `_parse_params()`.
                """
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

        return NPModelResults(self, **metrics)

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
    """
        Supervised nuisance regression model m(x) with PyTorch training loop.
        Supports early stopping, gradient clipping, and GPU execution.
        """
    def __init__(self, variable_types: dict, **params):
        """
                Parameters
                ----------
                variable_types : dict
                    Map {col_index: 'continuous'|'binary'|'ordinal'}.
                **params : dict
                    Architecture hyperparameters consumed by `_parse_params()`.
                """
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
        """
           Train m(x) via MSE on (X, y), with optional fixed split or random split.

           Parameters
           ----------
           X : array_like, shape (n, d)
               Features.
           y : array_like, shape (n,) or (n,1)
               Targets; reshaped to (n,1) internally.
           epochs, batch_size, learning_rate, weight_decay : see base.
           val_frac : float
               Fraction for validation if indices not provided.
           patience, min_delta : int, float
               Early stopping control on validation loss.
           grad_clip_norm : float or None
               Clip gradient global norm if set.
           num_workers : int
               DataLoader workers.
           shuffle : bool
               Shuffle training subset per epoch.
           verbose : bool
               Print training logs if True.
           train_idx, val_idx : array_like or None
               Predefined index splits. If both None, a random split is used.

           Returns
           -------
           NNModelNuisanceResults
               Fitted model with metrics and inference APIs.
        """

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
            **{"val_loss": best_loss if best_state is not None else None}
        )

    # ---------- PER-BATCH LOSS: TRAIN ----------
    def training_step(self, batch):
        """
            One supervised training step with MSE loss.

            Parameters
            ----------
            batch : tuple
                (xb, yb) tensors.

            Returns
            -------
            torch.Tensor
                Scalar MSE loss.
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
    """
    Density model with two networks:
      - score_model: estimates ∇_{cont} log f(x) on selected continuous coords
      - cond_model : estimates p(z_disc | x_other) over joint discrete states
    Builds, trains, and returns results with unified alpha() access.
    """
    def __init__(self, variable_types: dict, **params):
        """
        Parameters
        ----------
        variable_types : dict
           Map {col_index: 'continuous'|'binary'|'ordinal'}.
        **params : dict
           Nested configs for shared/score/cond heads, parsed by `_parse_params()`.

        Parse architecture hyperparameters for score and cond networks.

        Expected structure
        ------------------
        {
          "shared": {...},    # defaults applied to both heads
          "score":  {...},    # overrides for score_model
          "cond":   {...}     # overrides for cond_model
        }

        Returns
        -------
        dict
            {"score": cfg_score, "cond": cfg_cond} with validated fields.
       """
        super().__init__(variable_types, build_now=False, **params)

    def _parse_params(self) -> dict:
        """
        Parse architecture hyperparameters for both network heads:
          - shared  : defaults applied to all heads
          - score   : overrides for the score network
          - cond    : overrides for the conditional density network

        Expected self.params structure:
        {
          "shared": {...},    # defaults for both heads
          "score":  {...},    # optional overrides
          "cond":   {...}     # optional overrides
        }

        Does NOT require or inject input/output sizes.
        These are added later when data dimensions are known.

        Returns
        -------
        dict
            {"score": cfg_score, "cond": cfg_cond}
        """
        if not hasattr(self, "params") or not isinstance(self.params, dict):
            raise ValueError("self.params must be a dict")

        shared_defaults = {
            "hidden_layers": [1028, 1028, 1028, 1028, 1028],
            "activation": "leaky_relu",
            "output_activation": "identity",
            "dropout": 0.2,
            "bias": True,
            "weight_init": "default",
        }

        shared = self.params.get("shared", {})
        score = self.params.get("score", {})
        cond = self.params.get("cond", {})

        base = {**shared_defaults, **shared}
        cfg_score = {**base, **score}
        cfg_cond = {**base, **cond, 'output_activation': 'identity'}  # cond always identity

        def _validate(cfg, name):
            hl = cfg.get("hidden_layers")
            if not (isinstance(hl, list) and all(isinstance(n, int) and n > 0 for n in hl)):
                raise ValueError(f"{name}.hidden_layers must be list of positive ints")
            if cfg["activation"] not in {"relu", "tanh", "sigmoid", "leaky_relu"}:
                raise ValueError(f"{name}.activation invalid")
            if cfg["output_activation"] not in {"identity","relu","tanh","sigmoid","softmax","log_softmax"}:
                raise ValueError(f"{name}.output_activation invalid")
            d = cfg["dropout"]
            if not (isinstance(d, (int,float)) and 0.0 <= d < 1.0):
                raise ValueError(f"{name}.dropout in [0,1)")
            if cfg["weight_init"] not in {"default","xavier_uniform","xavier_normal","kaiming_uniform","kaiming_normal"}:
                raise ValueError(f"{name}.weight_init invalid")
            if not isinstance(cfg["bias"], bool):
                raise ValueError(f"{name}.bias must be bool")

        _validate(cfg_score, "score")
        _validate(cfg_cond, "cond")

        return {"score": cfg_score, "cond": cfg_cond}

    def fit(
            self, X, y=None, *,
            interest: list[int] | None = None,
            # shared defaults
            epochs: int = 100,
            batch_size: int = 256,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-3,
            val_frac: float = 0.2,
            patience: int = 20,
            min_delta: float = 0.0,
            grad_clip_norm: float | None = None,
            num_workers: int = 0,
            shuffle: bool = True,
            verbose: bool = True,
            **kwargs
    ) -> "NNModelDensityResults":
        """
           Fit score_model and cond_model using provided interest indices.

           Parameters
           ----------
           X : array_like, shape (n, d)
               Inputs for density modeling.
           y : ignored
               Present for API symmetry.
           interest : list[int] or None
               Variables of interest; partitioned into continuous vs discrete.
           epochs, batch_size, learning_rate, weight_decay, val_frac, patience,
           min_delta, grad_clip_norm, num_workers, shuffle, verbose : see base.
           **kwargs : dict
               Head-specific kwargs using prefixes 'score__' and 'cond__'.

           Returns
           -------
           NNModelDensityResults
               Trained density results with score/cond access.
           """

        shared = {
            "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
            "weight_decay": weight_decay, "val_frac": val_frac, "patience": patience,
            "min_delta": min_delta, "grad_clip_norm": grad_clip_norm,
            "num_workers": num_workers, "shuffle": shuffle, "verbose": verbose,
        }

        score_kw, cond_kw = self._split_head_kwargs(kwargs)
        score_fit_cfg = {**shared, **score_kw}
        cond_fit_cfg = {**shared, **cond_kw}

        arch_config = self._parse_params()
        score_arch_cfg = arch_config["score"]
        cond_arch_cfg = arch_config["cond"]

        dims_dict = self._infer_dims(X, interest)

        score_dims, score_meta = dims_dict["score"]
        cond_dims, cond_meta = dims_dict["cond"]

        score_model = self._fit_score_model(X=X, dims=score_dims, meta=score_meta, arch_cfg = score_arch_cfg,fit_cfg=score_fit_cfg)
        cond_model = self._fit_conditional_model(X=X, dims=cond_dims, meta=cond_meta, arch_cfg=cond_arch_cfg, fit_cfg=cond_fit_cfg)

        self.model = {
            "score": (score_model, score_meta),
            "cond": (cond_model, cond_meta)
        }

        return NNModelDensityResults(
            self,
            **{}
        )

    def _fit_score_model(self, X, dims,meta, arch_cfg, fit_cfg):
        """
        Train a score network on all inputs X to predict scores for interested
        continuous coordinates only.

        Parameters
        ----------
        X : array-like, shape (n, d)
        dims : dict with keys:
            - "input_size": int
            - "output_size": int
        meta : dict with keys:
            - "interest_cont_indices": list[int]  # order matches model outputs
        arch_cfg : dict  # hyperparameters for FeedForwardNNModel
        fit_cfg : dict   # training hyperparameters:
            epochs, batch_size, learning_rate, weight_decay, val_frac,
            patience, min_delta, grad_clip_norm, num_workers, shuffle, verbose

        Returns
        -------
        nn.Module
            Trained score model (or NullNN if no outputs requested).
        """

        import torch
        from torch.utils.data import DataLoader, TensorDataset, Subset
        from loglinearcorrection.neural_network_models import FeedForwardNNModel, NullNN, ScoreMatchingLossRestricted

        if dims["output_size"] == 0:
            return NullNN()

        interest_idx = meta.get("interest_cont_indices", [])
        if len(interest_idx) != dims["output_size"]:
            raise ValueError("meta['interest_cont_indices'] size must match dims['output_size'].")

        device = getattr(self, "device", torch.device("cpu"))
        X = torch.as_tensor(X, dtype=torch.float32)
        n = X.size(0)

        # model
        cfg = {**arch_cfg, **dims}
        model = FeedForwardNNModel(cfg).to(device)

        # loss
        loss_fn = ScoreMatchingLossRestricted(interest=interest_idx).to(device)

        # data split
        val_frac = float(fit_cfg.get("val_frac", 0.2))
        n_val = int(n * val_frac)
        perm = torch.randperm(n)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:] if n_val > 0 else torch.arange(n)

        ds = TensorDataset(X)
        pin = (device.type == "cuda")
        bs = int(fit_cfg.get("batch_size", 128))
        nw = int(fit_cfg.get("num_workers", 0))

        dl_tr = DataLoader(
            Subset(ds, tr_idx),
            batch_size=bs,
            shuffle=bool(fit_cfg.get("shuffle", True)),
            num_workers=nw,
            pin_memory=pin,
            drop_last=False,
        )
        dl_va = (
            DataLoader(
                Subset(ds, val_idx),
                batch_size=bs,
                shuffle=False,
                num_workers=nw,
                pin_memory=pin,
                drop_last=False,
            )
            if n_val > 0 else None
        )

        # optimizer
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(fit_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(fit_cfg.get("weight_decay", 0.0)),
        )
        grad_clip = fit_cfg.get("grad_clip_norm", None)
        patience = int(fit_cfg.get("patience", 10))
        min_delta = float(fit_cfg.get("min_delta", 0.0))
        epochs = int(fit_cfg.get("epochs", 100))
        verbose = bool(fit_cfg.get("verbose", True))

        best_loss = float("inf")
        best_state = None
        bad = 0
        log_every = max(1, epochs // 10)

        for ep in range(1, epochs + 1):
            # ---- train
            model.train()
            tr_sum = tr_cnt = 0
            for (xb,) in dl_tr:
                xb = xb.to(device, non_blocking=pin)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model, xb)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                bs_ = xb.size(0)
                tr_sum += float(loss.detach()) * bs_
                tr_cnt += bs_
            tr_loss = tr_sum / max(tr_cnt, 1)

            # ---- validate
            if dl_va is not None:
                model.eval()
                va_sum = va_cnt = 0
                with torch.enable_grad():  # need grads for score-matching val
                    for (xb,) in dl_va:
                        xb = xb.to(device, non_blocking=pin).detach().requires_grad_(True)
                        l = loss_fn(model, xb).detach()
                        bs_ = xb.size(0)
                        va_sum += float(l) * bs_
                        va_cnt += bs_
                va_loss = va_sum / max(va_cnt, 1)

                improved = va_loss + min_delta < best_loss
                if improved:
                    best_loss = va_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1

                if verbose and (ep % log_every == 0 or ep == 1 or ep == epochs):
                    print(
                        f"[Score ep {ep:>3}/{epochs}] train={tr_loss:.6f} val={va_loss:.6f}{' *' if improved else ''}")

                if bad >= patience:
                    if verbose:
                        print(f"Early stopping score head at epoch {ep} (best val={best_loss:.6f}).")
                    break
            else:
                if verbose and (ep % log_every == 0 or ep == 1 or ep == epochs):
                    print(f"[Score ep {ep:>3}/{epochs}] train={tr_loss:.6f}")

        if best_state is not None:
            model.load_state_dict(best_state)

        return model

    def _fit_conditional_model(self, X, dims, meta, arch_cfg, fit_cfg):
        """
        Train a joint conditional classifier over interested *binary* variables.

        Input  = X without interested discrete dims
        Target = class id for joint state of interested binary dims
        Output = logits over all joint states (len(meta['combos']))
        """

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, Subset
        from loglinearcorrection.neural_network_models import FeedForwardNNModel, NullNN

        # no discrete vars to model
        if dims["output_size"] == 0 or len(meta.get("interest_disc_indices", [])) == 0:
            return NullNN()

        interest = meta["interest_disc_indices"]
        combos = meta["combos"]  # (K, k_disc)
        enc = meta["combo_encoder"]  # tuple -> class id
        K = combos.shape[0]

        # build inputs X_other and labels y
        X = np.asarray(X, dtype=np.float32)
        d = X.shape[1]
        mask_other = np.ones(d, dtype=bool)
        mask_other[interest] = False

        X_in = X[:, mask_other]
        Z = X[:, interest].astype(int)
        # Combines for eg (1,1,1) -> class 7
        y = np.array([enc[tuple(z.tolist())] if tuple(z.tolist()) in enc else -1 for z in Z], dtype=np.int64)


        # filter out unknown labels (shouldn't happen if combos built from data)
        keep = (y >= 0)
        X_in = X_in[keep]
        y = y[keep]
        n = X_in.shape[0]

        if n == 0:
            return NullNN()

        device = getattr(self, "device", torch.device("cpu"))

        # model
        cfg = {**arch_cfg, **{"input_size": dims["input_size"], "output_size": dims["output_size"],
                              "output_activation": "identity"}}
        model = FeedForwardNNModel(cfg).to(device)

        # loss with optional class weights for imbalance
        class_weights = fit_cfg.get("class_weights", None)
        if class_weights is None:
            # inverse frequency
            counts = np.bincount(y, minlength=K).astype(np.float32)
            cw = counts.max() / np.maximum(counts, 1.0)
            # class_weights = torch.tensor(cw, dtype=torch.float32, device=device)
            class_weights = torch.ones(K, dtype=torch.float32, device=device)
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # split
        val_frac = float(fit_cfg.get("val_frac", 0.2))
        n_val = int(n * val_frac)
        perm = np.random.permutation(n)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:] if n_val > 0 else np.arange(n)

        # dataloaders
        pin = (device.type == "cuda")
        bs = int(fit_cfg.get("batch_size", 128))
        nw = int(fit_cfg.get("num_workers", 0))
        ds = TensorDataset(torch.as_tensor(X_in, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))

        dl_tr = DataLoader(Subset(ds, torch.as_tensor(tr_idx)), batch_size=bs,
                           shuffle=bool(fit_cfg.get("shuffle", True)),
                           num_workers=nw, pin_memory=pin, drop_last=False)
        dl_va = (DataLoader(Subset(ds, torch.as_tensor(val_idx)), batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=pin, drop_last=False) if n_val > 0 else None)

        # optimizer
        opt = torch.optim.AdamW(model.parameters(),
                                lr=float(fit_cfg.get("learning_rate", 1e-3)),
                                weight_decay=float(fit_cfg.get("weight_decay", 0.0)))
        grad_clip = fit_cfg.get("grad_clip_norm", None)
        patience = int(fit_cfg.get("patience", 10))
        min_delta = float(fit_cfg.get("min_delta", 0.0))
        epochs = int(fit_cfg.get("epochs", 100))
        verbose = bool(fit_cfg.get("verbose", True))

        best_loss = float("inf")
        best_state = None
        bad = 0
        log_every = max(1, epochs // 10)

        for ep in range(1, epochs + 1):
            # train
            model.train()
            tr_sum = tr_cnt = 0
            for xb, yb in dl_tr:
                xb = xb.to(device, non_blocking=pin)
                yb = yb.to(device, non_blocking=pin)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)  # (B, K)
                loss = criterion(logits, yb)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                bs_ = xb.size(0)
                tr_sum += float(loss.detach()) * bs_
                tr_cnt += bs_
            tr_loss = tr_sum / max(tr_cnt, 1)

            # validate
            if dl_va is not None:
                model.eval()
                va_sum = va_cnt = 0
                with torch.no_grad():
                    for xb, yb in dl_va:
                        xb = xb.to(device, non_blocking=pin)
                        yb = yb.to(device, non_blocking=pin)
                        logits = model(xb)
                        l = criterion(logits, yb)
                        bs_ = xb.size(0)
                        va_sum += float(l) * bs_
                        va_cnt += bs_
                va_loss = va_sum / max(va_cnt, 1)

                improved = va_loss + min_delta < best_loss
                if improved:
                    best_loss = va_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1

                if verbose and (ep % log_every == 0 or ep == 1 or ep == epochs):
                    print(f"[Cond ep {ep:>3}/{epochs}] train={tr_loss:.6f} val={va_loss:.6f}{' *' if improved else ''}")

                if bad >= patience:
                    if verbose:
                        print(f"Early stopping cond head at epoch {ep} (best val={best_loss:.6f}).")
                    break
            else:
                if verbose and (ep % log_every == 0 or ep == 1 or ep == epochs):
                    print(f"[Cond ep {ep:>3}/{epochs}] train={tr_loss:.6f}")

        if best_state is not None:
            model.load_state_dict(best_state)

        return model


    def _infer_dims(self, X: np.ndarray, interest: list[int]):
        """
        Determine input/output sizes for:
          - score network: input = all X, output = interested continuous coords
          - conditional network: input = X without interested coords,
            output = number of unique joint states over interested discrete coords

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        interest : list[int]

        Returns
        -------
        dict
            {
              "score": {
                "input_size": int,
                "output_size": int,
                "interest_cont_indices": list[int]
              },
              "cond": {
                "input_size": int,
                "output_size": int,
                "interest_disc_indices": list[int],
                "combos": np.ndarray,            # shape (n_unique, k_disc)
                "combo_encoder": dict[tuple,int] # mapping state -> class id
              }
            }
        """
        X = np.asarray(X)
        n, d = X.shape
        interest = list(interest)

        # classify variables
        def is_cont(j):
            t = self.variable_types.get(j, "continuous")
            return t == "continuous"

        def is_disc(j):
            t = self.variable_types.get(j, "continuous")
            return t in ("binary", "ordinal")

        # score head
        cont_interest = [j for j in interest if is_cont(j)]
        score_cfg = {
            "input_size": d,
            "output_size": len(cont_interest)
        }

        score_meta = {"interest_cont_indices": cont_interest}

        # conditional head
        disc_interest = [j for j in interest if is_disc(j)]
        if len(disc_interest) > 0:
            Z = X[:, disc_interest]
            # ensure integer encoding for uniqueness; safe cast if already ints
            if not np.issubdtype(Z.dtype, np.integer):
                Z = Z.astype(int)
            # unique joint states
            combos, inv = np.unique(Z, axis=0, return_inverse=True)
            # build encoder
            encoder = {tuple(state.tolist()): k for k, state in enumerate(combos)}
            cond_out = combos.shape[0]
        else:
            combos = np.empty((0, 0), dtype=int)
            encoder = {}
            cond_out = 0

        cond_cfg = {
            "input_size": d - len(disc_interest),
            "output_size": cond_out
        }

        cond_meta = {
            "interest_disc_indices": disc_interest,
            "combos": combos,
            "combo_encoder": encoder
        }

        return {"score": (score_cfg, score_meta), "cond": (cond_cfg, cond_meta)}



    def _split_head_kwargs(self, kw: dict) -> tuple[dict, dict]:
        """
            Split kwargs by head prefix.

            Parameters
            ----------
            kw : dict
                Possibly prefixed kwargs.

            Returns
            -------
            (dict, dict)
                (score_kwargs, cond_kwargs) with prefixes stripped.

            Raises
            ------
            ValueError
                On unknown, unprefixed keys.
        """
        score_kw, cond_kw = {}, {}
        if len(kw) == 0:
            return score_kw, cond_kw
        for k, v in kw.items():
            if k.startswith("score__"):
                score_kw[k[len("score__"):]] = v
            elif k.startswith("cond__"):
                cond_kw[k[len("cond__"):]] = v
            else:
                raise ValueError(f"Unknown kwarg '{k}'. Use 'score__' or 'cond__' prefix.")
        return score_kw, cond_kw




class NNModelDensityResults(NPModelResultsDensity):
    def __init__(self, model: NNModelDensity, **metrics):
        super().__init__(model, **metrics)
        self.device = model.device

    def alpha_weight(self, X: npt.ArrayLike, var_index: list[int]) -> npt.NDArray[np.float64]:
        """
        For each requested index in 'indices':
          - if continuous-interest: return score f'(x)/f(x) for that coord
          - if discrete-interest:   return 0  (placeholder)
        Order is preserved. Mixed lists are allowed.
        """
        import torch

        if not var_index:
            raise ValueError("indices must be non-empty.")

        # unpack models + metadata
        score_model, score_meta = self.model["score"]
        cond_model, cond_meta = self.model["cond"]

        cont_int = list(score_meta.get("interest_cont_indices", []))
        disc_int = list(cond_meta.get("interest_disc_indices", []))

        idxs = [int(i) for i in var_index]

        # compute scores once if any cont index requested
        need_scores = any(i in cont_int for i in idxs)
        need_cond = any(i in disc_int for i in idxs)

        scores = None
        if need_scores:
            Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32, device=self.device)
            score_model.eval()
            with torch.no_grad():
                scores = score_model(Xt).detach().cpu().numpy()  # shape (n, |cont_int|)

        conds = None
        discrete_x = None
        original_encoding = None
        if need_cond:
            Z = np.asarray(X, dtype=np.float32)
            Z = np.delete(Z, disc_int, axis=1)  # remove interested discrete cols
            discrete_x = np.asarray(X, dtype=np.int64)[:, disc_int]
            classes = [tuple(row.tolist()) for row in discrete_x]
            original_encoding  = [cond_meta['combo_encoder'].get(k, -99) for k in classes]
            if -99 in original_encoding:
                raise ValueError("Some discrete variable combinations not seen during training.")
            Zt = torch.as_tensor(Z, dtype=torch.float32, device=self.device)
            cond_model.eval()
            with torch.no_grad():
                probs = torch.softmax(cond_model(Zt), dim=1).detach().cpu().numpy()  # shape (n, num_classes)

        n = np.asarray(X).shape[0]
        cols = []
        for k in idxs:
            if k in cont_int:
                cols.append(scores[:, cont_int.index(k)])
            elif k in disc_int:
                cols.append(self.get_prob_discrete(probs, discrete_x, original_encoding, k))
            else:
                raise ValueError(f"index {k} not in continuous-interest or discrete-interest sets")

        return np.column_stack(cols).astype(np.float64)

    def get_prob_discrete(self, probs: np.ndarray, discrete_x, original_encoding, var_index: int) -> np.ndarray:
        cond_meta = self.model["cond"][1]
        n = probs.shape[0]
        disc_int = cond_meta.get("interest_disc_indices", [])

        if not np.issubdtype(discrete_x.dtype, np.integer):
            discrete_x = discrete_x.astype(int)

        discrete_x_bit_flip = discrete_x.copy()
        discrete_x_bit_flip[:, disc_int.index(var_index)] = 1 - discrete_x_bit_flip[:, disc_int.index(var_index)]

        new_encoding = [cond_meta['combo_encoder'].get(tuple(row.tolist()), -99) for row in discrete_x_bit_flip]

        rows = np.arange(n)
        p = probs[rows, original_encoding]
        p_flip = probs[rows, new_encoding]
        probs = p / (p + p_flip)

        return probs.astype(np.float64)




class NNModelNuisanceResults(NPModelResultsNuisance):
    def __init__(self, model: NNModelNuisance, **metrics):
        super().__init__(model, **metrics)
        self.device = model.device

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        import torch
        self.model.eval()
        device = self.device
        x_tensor = torch.as_tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model(x_tensor)
        preds = preds.squeeze(-1) if preds.ndim == 2 and preds.shape[1] == 1 else preds
        return preds.detach().cpu().numpy()

    def derivative(self, X: npt.ArrayLike, var_index) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # this should eventually return m(1,x) and m(0,x) for binary variables (grad=m(1,x), pred=m(0,x)), so correction= grad/pred similar to continuous case
        """
        Returns (pred, grad) where:
          pred[i] = m(x_i)
          grad[i, k] = ∂m(x_i)/∂x_k  for k in interest_continuous
        """
        import torch
        self.model.eval()

        xt = torch.tensor(X, dtype=torch.float32, device=self.device, requires_grad=True)
        yhat = self.model(xt)

        if yhat.ndim == 1 or (yhat.ndim == 2 and yhat.shape[1] == 1):
            ysc = yhat.squeeze(-1)
            grads_all = torch.autograd.grad(
                ysc, xt,
                grad_outputs=torch.ones_like(ysc),
                create_graph=False, retain_graph=False
            )[0]
            G = grads_all[:, var_index]  # (n, K)
            preds = ysc.detach().cpu().numpy()  # (n,)
            grads = G.detach().cpu().numpy()  # (n, K)
            return preds, grads
        else:
            outs = yhat.shape[1]
            cols = []
            for o in range(outs):
                g_all = torch.autograd.grad(
                    yhat[:, o].sum(), xt, create_graph=False, retain_graph=True
                )[0]
                cols.append(g_all[:, var_index].unsqueeze(1))  # (n,1,K)
            G = torch.cat(cols, dim=1)  # (n, q, K)
            preds = yhat.detach().cpu().numpy()  # (n, q)
            grads = G.detach().cpu().numpy()  # (n, q, K)
            return preds, grads