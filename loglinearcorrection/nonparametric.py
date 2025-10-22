import numpy as np
import numpy.typing as npt


class NPModel:
    def __init__(self, variable_types: dict, params: dict):
        self.variable_types = variable_types
        self.params = params
        self.model = None

    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> "NPModelResults":
        pass


class NPModelResults:

    def __init__(self, model: NPModel, x: npt.ArrayLike, y: npt.ArrayLike):
        self.model = model.model
        self.x = x
        self.y = y

    def predict(self, x: npt.ArrayLike) -> npt.NDArray[np.float_]:
        pass

    def derivative(self, x: npt.NDArray[np.float_], var_index: int) -> npt.NDArray[np.float_]:
        pass

    def second_derivative(self, x: npt.NDArray[np.float_], var_index: int) -> npt.NDArray[np.float_]:
        pass



class NNModel(NPModel):
    def __init__(self, variable_types: dict, params: dict):
        import torch
        super().__init__(variable_types, params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model().to(self.device)


    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> "NNModelResults":
        pass

    def _create_model(self):
        nn_params = self._parse_params()
        return self._create_model_torch(nn_params)

    def _create_model_torch(self, nn_params):
        from loglinearcorrection.neural_network_models import FeedForwardNNModel
        model = FeedForwardNNModel(nn_params)
        return model

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


class NNModelScore(NNModel):

    def __init__(self, variable_types: dict, params: dict):
        if params['input_size'] != params['output_size']:
            raise ValueError("For NNModelScore, input_size must equal output_size")
        super().__init__(variable_types, params)

    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> "NNModelResults":
        # y unused for score matching
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from loglinearcorrection.neural_network_models import SlicedScoreMatchingLoss

        X = torch.as_tensor(x, dtype=torch.float32)
        ds = TensorDataset(X)
        dl = DataLoader(
            ds,
            batch_size=self.params.get("batch_size", 128),
            shuffle=True,
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
        )

        loss_fn = SlicedScoreMatchingLoss(M=self.params.get("projections", 1))
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.get("learning_rate", 1e-3),
            weight_decay=self.params.get("weight_decay", 0.0),
        )
        epochs = self.params.get("epochs", 100)
        clip = self.params.get("grad_clip_norm", None)

        self.model.train().to(self.device)
        final_epoch_loss = float("nan")

        for _ in range(epochs):
            running, count = 0.0, 0
            for (xb,) in dl:
                xb = xb.to(self.device, non_blocking=(self.device.type == "cuda"))
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.model, xb)  # SSM loss expects (model, x)
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                opt.step()

                bs = xb.size(0)
                running += float(loss.detach()) * bs
                count += bs
            final_epoch_loss = running / max(count, 1)

        return NNModelScoreResults(self, x, y)



class NNModelNuisance(NNModel):
    pass


class NNModelResults(NPModelResults):
    def __init__(self, model: NNModel, x: npt.ArrayLike, y: npt.ArrayLike):
        super().__init__(model, x, y)
        pass

    def predict(self, x: npt.ArrayLike) -> npt.NDArray[np.float_]:
        pass

    def derivative(self, x: npt.NDArray[np.float_], var_index: int) -> npt.NDArray[np.float_]:
        pass

    def second_derivative(self, x: npt.NDArray[np.float_], var_index: int) -> npt.NDArray[np.float_]:
        pass

class NNModelScoreResults(NNModelResults):
    def __init__(self, model: NNModelScore, x: npt.ArrayLike, y: npt.ArrayLike):
        super().__init__(model, x, y)
        pass