import numpy as np
from loglinearcorrection.model import DoublyRobustElasticityEstimatorModel as DREEM
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Data
np.random.seed(42)  # For reproducibility
n = 15000
mu_x = 0.3
X1 = np.random.normal(mu_x, 2, size=n)
X2 = np.random.normal(0, 1, size=n)

# Binary X3 with probability depending on X2
# Use logistic function to map X2 to probability
p_x3 = 1 / (1 + np.exp(-X2))  # Probability increases with X2
X3 = np.random.binomial(1, p_x3, size=n)

# Parameters
a = 0.3  # variance component for X1
b = 0.1  # base variance
c = 0.5  # variance component for X3
beta = 0.5
gamma = 0.7
delta = 0.3  # coefficient for X3
n_folds = 3

# Heteroskedastic errors with variance depending on X1 and X3
errors = np.random.normal(0, np.sqrt(a * X1**2 + b + c * X3))
Y = np.exp(2 + beta * X1 + gamma * X2 + delta * X3 + errors)

X = sm.add_constant(np.column_stack([X1, X2, X3]), prepend=False)

# Fit DR estimator
model = DREEM(
    endog=Y,
    exog=X,
    interest=[0, 1, 2]  # X1, X2, and X3 elasticities
)

arch_params_m = {
    'hidden_layers': [1028, 1028, 1028, 1028],
    'output_activation': 'identity'
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

# Fit comparison models
ols_model = sm.OLS(np.log(Y), X).fit()
ppml_model = sm.GLM(Y, X, family=sm.families.Poisson()).fit()

# Get elasticity estimates
print("="*60)
print("ELASTICITY ESTIMATES")
print("="*60)

# X1 (continuous)
estimate_x1, std_err_x1 = results.get_elasticity('x1')
print("\nX1 (Continuous):")
print(f"  DR estimate:    {estimate_x1:.4f} (SE: {std_err_x1:.4f})")
print(f"  OLS estimate:   {ols_model.params[0]:.4f}")
print(f"  PPML estimate:  {ppml_model.params[0]:.4f}")
print(f"  True elasticity: {beta + a * mu_x:.4f}")

# X2 (continuous)
estimate_x2, std_err_x2 = results.get_elasticity('x2')
print("\nX2 (Continuous):")
print(f"  DR estimate:    {estimate_x2:.4f} (SE: {std_err_x2:.4f})")
print(f"  OLS estimate:   {ols_model.params[1]:.4f}")
print(f"  PPML estimate:  {ppml_model.params[1]:.4f}")
print(f"  True elasticity: {gamma:.4f}")

# X3 (binary)
estimate_x3, std_err_x3 = results.get_elasticity('x3')
print("\nX3 (Binary):")
print(f"  DR estimate:    {estimate_x3:.4f} (SE: {std_err_x3:.4f})")
print(f"  OLS estimate:   {ols_model.params[2]:.4f}")
print(f"  PPML estimate:  {ppml_model.params[2]:.4f}")
# True elasticity for binary variable
# m(0) = E[exp(u)|X3=0] = exp((a*E[X1^2] + b)/2)
# m(1) = E[exp(u)|X3=1] = exp((a*E[X1^2] + b + c)/2)
# So m(1)/m(0) = exp(c/2)
# True percentage change = exp(delta) * m(1)/m(0) - 1 = exp(delta + c/2) - 1
true_elasticity_x3 = np.exp(delta + c/2) - 1
print(f"  True elasticity: {true_elasticity_x3:.4f}")
print(f"    (exp(δ={delta:.2f}) * exp(c/2={c/2:.2f}) - 1)")

# Plotting diagnostic for continuous variables (X1 and X2)
fig, axes = plt.subplots(2, n_folds, figsize=(12, 6))
fig.suptitle('Diagnostics for X1 (Continuous Variable)')

for i in range(n_folds):
    fold_result = results.fold_results[i]
    test_idx = fold_result['test_idx']
    X_test = X[test_idx]
    
    # Panel 1: m(x) for X1
    m_results = fold_result['m_results']
    m_hat = m_results.predict(X_test)
    # True m(x) now includes effect of X3
    m_true = np.exp((a * X_test[:, 0]**2 + b + c * X_test[:, 2])/2)
    
    axes[0, i].scatter(X_test[:, 0], m_hat, alpha=0.4, label='m_hat(x)', s=1)
    axes[0, i].scatter(X_test[:, 0], m_true, alpha=0.4, label='m_true(x)', s=1)
    axes[0, i].set_ylim(0, min(100, max(m_hat.max(), m_true.max()) * 1.1))
    axes[0, i].set_xlabel('X1')
    axes[0, i].set_ylabel('m(x)')
    axes[0, i].set_title(f"m(x) vs X1, fold {i+1}")
    axes[0, i].legend()
    
    # Panel 2: score for X1
    f_results = fold_result['f_results']
    score_hat = f_results.alpha_weight(X_test, [0])[:, 0]
    score_true = -(X_test[:, 0] - mu_x)/4
    
    axes[1, i].scatter(score_true, score_hat, alpha=0.4, s=1)
    axes[1, i].plot([score_true.min(), score_true.max()], 
                    [score_true.min(), score_true.max()], 
                    'r--', alpha=0.5, label='45°')
    axes[1, i].set_xlabel('True score')
    axes[1, i].set_ylabel('Estimated score')
    axes[1, i].set_title(f"Score for X1, fold {i+1}")
    axes[1, i].legend()

plt.tight_layout()
plt.show()

# Separate diagnostic plots for binary X3
fig, axes = plt.subplots(2, n_folds, figsize=(12, 6))
fig.suptitle('Diagnostics for X3 (Binary Variable)')

for i in range(n_folds):
    fold_result = results.fold_results[i]
    test_idx = fold_result['test_idx']
    X_test = X[test_idx]
    
    # Panel 1: m(x) by X3 groups
    m_results = fold_result['m_results']
    m_hat = m_results.predict(X_test)
    
    # Split by X3 values
    x3_0_mask = X_test[:, 2] == 0
    x3_1_mask = X_test[:, 2] == 1
    
    # Box plot comparing m(x) for X3=0 vs X3=1
    axes[0, i].boxplot([m_hat[x3_0_mask], m_hat[x3_1_mask]], 
                       labels=['X3=0', 'X3=1'])
    axes[0, i].set_ylabel('m_hat(x)')
    axes[0, i].set_title(f"m(x) by X3, fold {i+1}")
    axes[0, i].grid(True, alpha=0.3)
    
    # Panel 2: Predicted probabilities P(X3=1|X)
    f_results = fold_result['f_results']
    # For binary variables, alpha_weight returns P(X3=x3|X)
    prob_x3 = f_results.alpha_weight(X_test, [2])[:, 0]
    
    # Plot predicted vs actual probabilities binned by X2
    x2_bins = np.percentile(X_test[:, 1], np.linspace(0, 100, 11))
    bin_centers = []
    actual_probs = []
    pred_probs = []
    
    for j in range(len(x2_bins)-1):
        mask = (X_test[:, 1] >= x2_bins[j]) & (X_test[:, 1] < x2_bins[j+1])
        if mask.sum() > 0:
            bin_centers.append((x2_bins[j] + x2_bins[j+1])/2)
            actual_probs.append(X_test[mask, 2].mean())
            pred_probs.append(prob_x3[mask].mean())
    
    axes[1, i].scatter(bin_centers, actual_probs, label='Actual P(X3=1)', s=50)
    axes[1, i].scatter(bin_centers, pred_probs, label='Predicted P(X3=1)', s=50)
    axes[1, i].set_xlabel('X2 (binned)')
    axes[1, i].set_ylabel('P(X3=1)')
    axes[1, i].set_title(f"P(X3=1|X2), fold {i+1}")
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional plot: Elasticity estimates across folds
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('Elasticity Estimates Across Folds')

# Extract theta_x from each fold for each variable
for var_idx, var_name in enumerate(['X1', 'X2', 'X3']):
    theta_folds = []
    for fold_result in results.fold_results:
        theta_x = fold_result['theta_x']
        if theta_x.size > 0:
            theta_folds.append(theta_x[:, var_idx])
    
    # Box plot of elasticity estimates
    if theta_folds:
        axes[var_idx].boxplot(theta_folds, labels=[f'Fold {i+1}' for i in range(len(theta_folds))])
        axes[var_idx].axhline(y=results.elasticities[var_idx] if not results._is_pandas 
                              else results.elasticities.loc[var_name.lower(), 'estimate'], 
                              color='r', linestyle='--', label='Final estimate')
        axes[var_idx].set_title(f'{var_name} Elasticity')
        axes[var_idx].set_ylabel('Elasticity')
        axes[var_idx].legend()
        axes[var_idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary
print("\n" + "="*60)
results.summary()
