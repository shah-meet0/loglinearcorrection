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
          - activation : {'relu','tanh','sigmoid','leaky_relu','silu'}
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
            "silu": nn.SiLU,
            "softplus": nn.Softplus,
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


class EmbeddingFeedForwardNNModel(nn.Module):
    """MLP with learned embeddings for trailing categorical input columns.

    The first ``non_fe_inputs`` columns remain differentiable floating
    point inputs.  Each remaining column is a zero-based category code.  This
    representation is particularly useful for high-dimensional fixed effects:
    it avoids allocating an ``nobs`` by ``n_levels`` dense dummy matrix while
    retaining category-specific trainable parameters.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.non_fe_inputs = int(config["non_fe_inputs"])
        cardinalities = [int(value) for value in config["categorical_cardinalities"]]
        if any(value < 1 for value in cardinalities):
            raise ValueError("categorical cardinalities must be positive")
        cap = int(config.get("embedding_dimension_cap", 32))
        if cap < 1:
            raise ValueError("embedding_dimension_cap must be positive")
        dimensions = [min(cap, max(2, int(round(value ** 0.25 * 4)))) for value in cardinalities]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, dimension) for cardinality, dimension in zip(cardinalities, dimensions)]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.05)

        mlp_config = dict(config)
        mlp_config["input_size"] = self.non_fe_inputs + sum(dimensions)
        self.mlp = FeedForwardNNModel(mlp_config)

    def forward(self, x):
        continuous = x[:, : self.non_fe_inputs]
        categorical = x[:, self.non_fe_inputs :]
        if categorical.shape[1] != len(self.embeddings):
            raise ValueError(
                f"Expected {len(self.embeddings)} categorical columns; "
                f"found {categorical.shape[1]}"
            )
        features = [continuous]
        for column, embedding in enumerate(self.embeddings):
            codes = categorical[:, column].round().long()
            codes = codes.clamp(0, embedding.num_embeddings - 1)
            features.append(embedding(codes))
        return self.mlp(torch.cat(features, dim=1))


def build_feed_forward_model(config: dict) -> nn.Module:
    """Build a plain or categorical-embedding feed-forward model."""
    cardinalities = config.get("categorical_cardinalities")
    if cardinalities:
        model = EmbeddingFeedForwardNNModel(config)
    else:
        model = FeedForwardNNModel(config)

    if config.get("positive_log_link", False):
        output_layer = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                output_layer = module
        if output_layer is None:
            raise ValueError("Could not find the nuisance network output layer")
        nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)
        model = BoundedLogLinkModel(
            model, float(config.get("log_link_max_abs", 3.0))
        )

    positive_floor = config.get("positive_floor")
    if positive_floor is not None:
        model = PositiveFloorModel(model, float(positive_floor))
    return model


class PositiveFloorModel(nn.Module):
    """Keep a positive-output network away from floating-point underflow."""

    def __init__(self, model: nn.Module, floor: float):
        super().__init__()
        if floor <= 0:
            raise ValueError("positive_floor must be positive")
        self.model = model
        self.floor = float(floor)

    def forward(self, x):
        return self.model(x) + self.floor


class BoundedLogLinkModel(nn.Module):
    """Positive mean-one link with bounded log predictions.

    A zero-initialized raw output maps exactly to one.  Bounding the log mean
    prevents tiny fitted means from turning ``m_prime / m`` into a numerical
    singularity in generalized DREAm.
    """

    def __init__(self, raw_model: nn.Module, max_abs: float):
        super().__init__()
        if max_abs <= 0:
            raise ValueError("log_link_max_abs must be positive")
        self.raw_model = raw_model
        self.max_abs = float(max_abs)

    def forward(self, x):
        log_mean = self.max_abs * torch.tanh(self.raw_model(x) / self.max_abs)
        return torch.exp(log_mean)


class NullNN(nn.Module):
    def __init__(self, input_size=0, output_size=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.present = False

    def forward(self, x):
        if self.output_size == 0:
            # return zero scalar or zero tensor matching expected shape
            return torch.zeros((x.size(0), 1), device=x.device)
        return torch.zeros((x.size(0), self.output_size), device=x.device)


class ScoreMatchingLossRestricted(nn.Module):
    """
    Exact Hyvärinen score-matching loss restricted to selected coordinates.

    For interest index set I:
        J(h) = E_x[ 0.5 * sum_{i in I} h_i(x)^2 + sum_{i in I} ∂h_i/∂x_i ].

    The model must output h(x) ∈ R^{|I|} with columns ordered as `interest`.
    """
    def __init__(self, interest: list[int]):
        """
           Parameters
           ----------
           interest : list[int]
               Indices of continuous variables whose score coordinates are modeled.
        """
        super().__init__()
        self.register_buffer("interest_idx", torch.tensor(interest, dtype=torch.long))

    @torch.enable_grad()
    def forward(self, model: nn.Module, x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute restricted Hyvärinen loss on a batch.

        Parameters
        ----------
        model : nn.Module
            Score model, output shape (n, |I|) in `interest` order.
        x : torch.Tensor, shape (n, d)
            Inputs with requires_grad True inside.
        weight : torch.Tensor, shape (n,), optional
            Per-sample weights; the loss is then the weighted mean.

        Returns
        -------
        torch.Tensor
            Scalar loss (mean, or weighted mean, over batch).
        """
        if self.interest_idx.numel() == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        x = x.detach().requires_grad_(True)
        h = model(x)                              # (n, |I|)
        if h.dim() != 2 or h.size(1) != self.interest_idx.numel():
            raise ValueError("model(x) must have shape (n, |interest|) in the same order as interest.")

        # 0.5 * ||h||^2 per-sample
        sq = 0.5 * (h * h).sum(dim=1)            # (n,)

        # divergence restricted to interest: sum_i ∂h_i/∂x_i
        # compute per-output gradient wrt corresponding input coord
        div_terms = []

        # Compute the parital derivatives for each interest variable
        for out_j, var_k in enumerate(self.interest_idx.tolist()):
            g = torch.autograd.grad(
                h[:, out_j].sum(),               # scalar
                x,
                create_graph=True,
                retain_graph=True,
                allow_unused=False
            )[0][:, var_k]                        # (n,)
            div_terms.append(g)
        div = torch.stack(div_terms, dim=1).sum(dim=1)  # (n,)

        per_sample = sq + div
        if weight is None:
            return per_sample.mean()
        weight = weight.to(per_sample.dtype).reshape(-1)
        return (weight * per_sample).sum() / weight.sum()

# class SlicedScoreMatchingLoss(nn.Module):
#     def __init__(self, M:int=1):
#         """
#         Initialize the Sliced Score Matching loss.
#
#         Parameters
#         ----------
#         n_projections : int
#             Number of random projections to use.
#         """
#         super().__init__()
#         self.M = M
#
#     @torch.enable_grad()
#     def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
#         x = x.detach().requires_grad_(True) # First make sure computation graph starts at x
#         loss = torch.zeros((), device=x.device)
#         feat_dims = tuple(range(1, x.ndim))
#
#         for _ in range(self.M):
#             v = torch.randn_like(x)  # pv with E[v v^T] = I
#             h = model(x)  # h_theta(x), same shape as x # This is an estimate of the score
#             hv = (h * v).sum(dim=feat_dims)  # v^T h(x), per-sample
#             # ∇_x (v^T h(x))
#             grad_hv = torch.autograd.grad(hv.sum(), x, create_graph=True)[0]
#             # v^T [∇_x h(x)] v  =  (∇_x (v^T h(x))) · v
#             jvp = (grad_hv * v).sum(dim=feat_dims)  # per-sample
#             loss = loss + 0.5 * (hv ** 2) + jvp
#         return loss.mean() / self.M
