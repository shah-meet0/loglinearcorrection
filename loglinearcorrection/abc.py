import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

n = 1000
X = np.random.normal(loc = 0, scale = 2, size = (n,1))
# have heteroskedastic error with variance = ax^2
a = 0.3
b = 0.1
errors = np.random.normal(loc=0, scale=np.sqrt(a * X**2 + b).flatten()).reshape(-1,1)

beta = 0.5
theta = beta + 0.5 * a

Y = np.exp(2.0 + beta * X + errors)

X_reg = sm.add_constant(X)
model = sm.OLS(np.log(Y), X_reg).fit()
model2 = sm.GLM(Y, X_reg, family=sm.families.Poisson()).fit()
print(model.summary())
print(model2.summary())

res = model.resid
exponentiated_resid = np.exp(res)

from loglinearcorrection.nonparametric import NNModelNuisance
import torch

nuisance_model = NNModelNuisance(variable_types={}, input_size=2, hidden_layers= [64,64, 64, 64], output_size=1, activation='relu', dropout=0.3)
nuisance_model.fit(X_reg, exponentiated_resid, val_frac=0.25, patience=30, weight_decay=0.0)
plt.plot(X, nuisance_model.model(torch.from_numpy(X_reg.astype('float32')).to('cuda')).cpu().detach().numpy(), 'o', color='red')
plt.plot(X, np.exp((a * X**2 +b)/2), 'o')
# plt.plot(X, exponentiated_resid, 'o', alpha=0.6)
plt.ylim(0, 100)
plt.show()


import torch
import numpy as np
from loglinearcorrection.neural_network_models import FeedForwardNNModel, NullNN, ScoreMatchingLossRestricted

def test_fit_score_model():
    # ----- synthetic data -----
    n, d = 1000, 5
    X = np.random.randn(n, d).astype(np.float32)

    # variable 0,1 continuous, 2,3 discrete but irrelevant here
    variable_types = {0: "continuous", 1: "continuous", 2: "binary", 3: "ordinal"}
    interest = [0]  # learn score only for first two dims

    # model init (dummy self substitute)
    class Dummy:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = Dummy()

    # dims/meta
    dims = {"input_size": d, "output_size": len(interest)}
    meta = {"interest_cont_indices": interest}

    arch_cfg = {
        "hidden_layers": [512,512,512, 512],
        "activation": "leaky_relu",
        "output_activation": "identity",
        "dropout": 0.3,
        "bias": True,
        "weight_init": "default",
    }
    fit_cfg = {
        "epochs": 100,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "val_frac": 0.2,
        "verbose": True,
        "patience": 100,
        'weight_decay': 1e-3,
    }

    # attach method
    from loglinearcorrection.nonparametric import NNModelDensity  # or wherever _fit_score_model lives
    model = NNModelDensity(variable_types=variable_types)
    model.device = dummy.device

    score_net = model._fit_score_model(X, dims, meta, arch_cfg, fit_cfg)

    # ----- assertions -----
    score_net.eval()
    with torch.no_grad():
        X_t = torch.tensor(X[:10], dtype=torch.float32, device=model.device)
        out = score_net(X_t)
        print("Output shape:", out.shape)
        assert out.shape == (10, len(interest))
        assert torch.isfinite(out).all()
    print("✓ _fit_score_model basic functionality OK")
    return score_net, X, correlation_score_vs_input(score_net, X, interest, device=model.device)


def correlation_score_vs_input(model, X, interest, device=None):
    """
    Compute Pearson correlation between predicted score h(x)
    and negative input -x for interested dimensions.
    """
    device = device or next(model.parameters()).device
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()
    X_np = np.asarray(X)[:, interest]

    corrs = []
    for j in range(len(interest)):
        xj = X_np[:, j]
        hj = pred[:, j]
        if np.std(xj) > 0 and np.std(hj) > 0:
            c = np.corrcoef(hj, -xj)[0, 1]
        else:
            c = np.nan
        corrs.append(c)

    mean_corr = np.nanmean(corrs)
    print(f"Mean correlation(h(x), -x) = {mean_corr:.3f}")
    return corrs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    score_net, X, corr = test_fit_score_model()
    plt.plot(X[:,0], -1 * score_net.model(torch.from_numpy(X.astype('float32')).to('cuda')).cpu().detach().numpy()[:,0], 'o')
    plt.plot(X[:,0],  X[:,0], 'o', color='red')
    plt.show()

import numpy as np
import torch
import matplotlib.pyplot as plt

# fake binary-conditional dataset
def make_toy_conditional(n=2000):
    x_cont = np.random.randn(n, 1)
    # binary variable depends on x_cont
    p1 = 1 / (1 + np.exp(-2 * x_cont))      # sigmoid of x_cont
    z = (np.random.rand(n, 1) < p1).astype(int)
    X = np.hstack([x_cont, z])              # shape (n, 2)
    return X, z

# suppose index 1 is binary interest
X, y_disc = make_toy_conditional()
variable_types = {0: "continuous", 1: "binary"}

# --- build config ---
from loglinearcorrection.neural_network_models import FeedForwardNNModel  # your base MLP
from loglinearcorrection.nonparametric import NNModelDensity

arch_cfg = {
    "hidden_layers": [16, 16],
    "activation": "relu",
    "output_activation": "identity",
    "dropout": 0.0,
    "bias": True,
    "weight_init": "default",
}

fit_cfg = {
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "val_frac": 0.2,
    "verbose": True,
    "patience": 20,
    'weight_decay': 1e-3,
}

# emulate meta info from _infer_dims
combos = np.array([[0], [1]], dtype=int)
encoder = {(0,): 0, (1,): 1}
meta = {"interest_disc_indices": [1], "combos": combos, "combo_encoder": encoder}

dims = {"input_size": 1, "output_size": 2}  # input=cont var, output=two binary states

# --- train test ---
model_density = NNModelDensity(variable_types)
cond_net = model_density._fit_conditional_model(X, dims, meta, arch_cfg, fit_cfg)

# --- evaluation ---
x_test = np.linspace(-3, 3, 200).reshape(-1, 1)
with torch.no_grad():
    logits = cond_net(torch.as_tensor(x_test, dtype=torch.float32, device=model_density.device))
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

plt.plot(x_test, probs[:, 1], label="Model P(z=1|x)")
plt.plot(x_test, 1 / (1 + np.exp(-2 * x_test)), "--", label="True sigmoid")
plt.legend(); plt.show()