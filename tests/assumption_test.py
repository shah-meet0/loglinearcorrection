from loglinearcorrection.data_generating_processes import DGP, MVNDataGenerator, ConstantGenerator, CombinedDataGenerator, IndependentNormErrorGenerator
from loglinearcorrection.dependence_funcs import independent_absolute, constant_mean, independent_squared
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


# THIS MODULE IS FOR PRELIMINARY TESTING OF THE PPML Assumption
def consistency(dgp, n):
    y, x, u = dgp.generate(n)
    model_ppml = sm.GLM(y, x, family=sm.families.Poisson())
    results_ppml = model_ppml.fit()
    yhat = model_ppml.predict(results_ppml.params, x)
    resids = y/yhat
    return np.mean(resids), results_ppml.params[0]



# Error Generator such that E[e^u|X] = 1

error_ppml = IndependentNormErrorGenerator(mean_fn=lambda X: -1 * independent_absolute(X),
                                           var_fn=lambda X: 2 * independent_absolute(X))

# Error Generator such that E[u|X] = 0

error_ols = IndependentNormErrorGenerator(mean_fn=constant_mean(0), var_fn=lambda X: 2 * independent_absolute(X))


# DGP for two cases, Y = exp(2X + 1 + u)
constant_generator = ConstantGenerator(1)
x_generator = MVNDataGenerator(means=[0], sigma=[[1]])
combined_generator = CombinedDataGenerator([x_generator, constant_generator])
betas = np.array([0.5, 1])

dgp_ppml = DGP(combined_generator, betas, error_ppml, exponential=True)
dgp_ols = DGP(combined_generator, betas, error_ols, exponential=True)

n_sim = np.linspace(50, 10000, 1000).astype(int)
results_ppml = []
results_ols = []

for n in n_sim:
    print(f'Calculating n = {n}')
    results_ppml.append(consistency(dgp_ppml, n))
    results_ols.append(consistency(dgp_ols, n))

results_ols = np.array(results_ols)
results_ppml = np.array(results_ppml)




fig, ax = plt.subplots(2,1, figsize=(12, 8))
sns.lineplot(x=n_sim[0:len(results_ols)], y=results_ols[:,0], ax=ax[0], label=r'$\mathbb{E}[e^u|X] \neq 1$')
sns.lineplot(x=n_sim[0:len(results_ppml)], y=results_ppml[:,0], ax=ax[0], label=r'$\mathbb{E}[e^u|X] = 1$')
ax[0].set_xlabel('')
ax[0].set_ylabel(r'$\frac{1}{n}\sum_i^n \frac{y_i}{\hat{y}_i}$', rotation=0, labelpad=8, fontsize=15)
ax[0].hlines(1, n_sim[0], n_sim[-1], color='red', linestyle='--', label='1')
ax[0].set_title('Test Statistic')
ax[0].yaxis.set_label_coords(-0.1, 0.5)
ax[0].set_ylim(0.5, 2.5)

sns.lineplot(x=n_sim[0:len(results_ols)], y=results_ols[:,1] - 0.5, ax=ax[1], label=r'$\mathbb{E}[e^u|X] \neq 1$')
sns.lineplot(x=n_sim[0:len(results_ppml)], y=results_ppml[:,1] - 0.5, ax=ax[1], label=r'$\mathbb{E}[e^u|X] = 1$')
ax[1].set_xlabel('Sample size n')
ax[1].set_ylabel(r'$\hat{\beta} - \beta$', rotation=0, labelpad=8, fontsize=15)
ax[1].hlines(0, n_sim[0], n_sim[-1], color='red', linestyle='--', label='0')
ax[1].set_title('Estimated Coefficient')
ax[1].yaxis.set_label_coords(-0.1, 0.5)
ax[1].set_ylim(-1, 1)

plt.savefig('./figures/convergence_test_coefficient.pdf', dpi=1000)