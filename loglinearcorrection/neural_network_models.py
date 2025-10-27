# This file is separate to prevent importing heavy dependencies like PyTorch when not needed.

import torch
import torch.nn as nn


class FeedForwardNNModel(nn.Module):

    def __init__(self, config:dict):

        """
         Initialize the NNModel.

         Parameters
         ----------
         config : dict
             Parsed hyperparameter dictionary used to construct the network.
         """

        super().__init__()
        self.model = self._build_model(config)

    def _build_model(self, config:dict) -> nn.Module:
        """
        Build a PyTorch MLP from a parsed hyperparameter dict.

        Expected config keys:
          - input_size : int > 0
          - hidden_layers : list[int]
          - output_size : int > 0
          - activation : {'relu','tanh','sigmoid','leaky_relu'}
          - output_activation : {'identity','relu','tanh','sigmoid','softmax','log_softmax'}
          - dropout : float in [0,1)
          - bias : bool
          - weight_init : {'default','xavier_uniform','xavier_normal','kaiming_uniform','kaiming_normal'}

        Returns
        -------
        torch.nn.Module
            Constructed model.
        """
        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "identity": nn.Identity,
            "softmax": lambda: nn.Softmax(dim=1),
            "log_softmax": lambda: nn.LogSoftmax(dim=1),
        }

        layers = []
        in_dim = config["input_size"]
        bias = config.get("bias", True)
        p = float(config.get("dropout", 0.0))

        hidden_act = act_map[config["activation"]]

        for h in config["hidden_layers"]:
            layers.append(nn.BatchNorm1d(in_dim))
            layers.append(nn.Linear(in_dim, h, bias=bias))
            layers.append(hidden_act())
            if p > 0:
                layers.append(nn.Dropout(p))
            in_dim = h

        layers.append(nn.Linear(in_dim, config["output_size"], bias=bias))

        out_act = config.get("output_activation", "identity")
        if out_act != "identity":
            layers.append(act_map[out_act]())

        model = nn.Sequential(*layers)

        # weight init
        scheme = config.get("weight_init", "default")

        def _init(m):
            if isinstance(m, nn.Linear):
                if scheme == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif scheme == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif scheme == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif scheme == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        model.apply(_init)
        return model

    def forward(self, x):
        return self.model(x)


class SlicedScoreMatchingLoss(nn.Module):
    def __init__(self, M:int=1):
        """
        Initialize the Sliced Score Matching loss.

        Parameters
        ----------
        n_projections : int
            Number of random projections to use.
        """
        super().__init__()
        self.M = M

    @torch.enable_grad()
    def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().requires_grad_(True) # First make sure computation graph starts at x
        loss = torch.zeros((), device=x.device)
        feat_dims = tuple(range(1, x.ndim))

        for _ in range(self.M):
            v = torch.randn_like(x)  # pv with E[v v^T] = I
            h = model(x)  # h_theta(x), same shape as x # This is an estimate of the score
            hv = (h * v).sum(dim=feat_dims)  # v^T h(x), per-sample
            # ∇_x (v^T h(x))
            grad_hv = torch.autograd.grad(hv.sum(), x, create_graph=True)[0]
            # v^T [∇_x h(x)] v  =  (∇_x (v^T h(x))) · v
            jvp = (grad_hv * v).sum(dim=feat_dims)  # per-sample
            loss = loss + 0.5 * (hv ** 2) + jvp
        return loss.mean() / self.M
