import numpy as np
from loglinearcorrection.model import DoublyRobustElasticityEstimatorModel as DREEM
import statsmodels.api as sm
import matplotlib.pyplot as plt


# Data
n = 15000
mu_x = 0.3
X1 = np.random.normal(mu_x, 2, size=n)
X2 = np.random.normal(0, 1, size=n)
a = 0.3
b = 0.1
beta = 0.5
gamma = 0.7
n_folds = 3
errors = np.random.normal(0, np.sqrt(a * X1**2 + b))
Y = np.exp(2 + beta * X1 + gamma * X2 + errors)

X = sm.add_constant(np.column_stack([X1, X2]), prepend=False)

# Fit DR estimator
model = DREEM(
    endog=Y,
    exog=X,
    interest=[0,1]          # Only X1 elasticity
)

arch_params_m = {
    'hidden_layers': [1028, 1028, 1028, 1028],
}
fit_params_m = {
    "epochs": 200,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-3,
    "patience": 20,
    "val_frac": 0.2,
}

results = model.fit(n_folds=n_folds, m_params={'fit_params': fit_params_m, 'arch_params': arch_params_m})

model = sm.OLS(np.log(Y), X).fit()
model2 = sm.GLM(Y, X, family=sm.families.Poisson()).fit()

print("Estimated elasticity for X1:", results.elasticities['x1']['estimate'])
print('Estimated standard error for X1:', results.elasticities['x1']['std_est'])
print('OLS estimate:', model.params[0])
print('PPML estimate:', model2.params[0])


print("True elasticity:", beta + a * mu_x)

# Plotting the nuisance model fit

fig, ax = plt.subplots(2, n_folds, figsize=(10,4))


for i in range(n_folds):
    # panel 1: m(x)
    ax[0][i].plot(X[:,0], results.fold_diagnostics['m_results'][i].predict(X), 'o', alpha=0.4, label='m_hat(x)')
    ax[0][i].plot(X[:,0], np.exp((a * X[:,0]**2 + b)/2), 'o', alpha=0.4, label='m_true(x)')
    ax[0][i].set_ylim(-10,100)
    ax[0][i].set_title(f"m(x), fold {i+1}")
    ax[0][i].legend()

    # panel 2: score
    score_true = -(X[:,0] - mu_x)/4
    score_hat = results.fold_diagnostics['f_results'][i].alpha_weight(X, [0])

    ax[1][i].plot(score_true, score_hat, 'o', alpha=0.4, label='score_hat vs score_true')
    ax[1][i].plot(score_true, score_true, 'o', alpha=0.4, label='45°')
    ax[1][i].set_title("score")
    ax[1][i].legend()

    plt.tight_layout()
    plt.show()
