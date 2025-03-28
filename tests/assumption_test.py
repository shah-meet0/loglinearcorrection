from loglinearcorrection.data_generating_processes import DGP, MVNDataGenerator, ConstantGenerator, CombinedDataGenerator, IndependentNormErrorGenerator
from loglinearcorrection.dependence_funcs import independent_absolute, constant_mean, independent_squared
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# THIS MODULE IS FOR PRELIMINARY TESTING OF THE PPML Assumption
def consistency(dgp, n):
    y, x, u = dgp.generate(n)
    model_ppml = sm.GLM(y, x, family=sm.families.Poisson())
    results_ppml = model_ppml.fit()
    yhat = model_ppml.predict(results_ppml.params, x)
    resids = y/yhat
    return np.mean(resids)



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

n_sim = np.linspace(100, 100000, 10000).astype(int)
results_ppml = []
results_ols = []

for n in n_sim:
    print(f'Calculating n = {n}')
    results_ppml.append(consistency(dgp_ppml, n))
    results_ols.append(consistency(dgp_ols, n))

results_ols = np.array(results_ols) - 1
results_ppml = np.array(results_ppml) - 1


fig, ax = plt.subplots()
sns.lineplot(x=n_sim, y=results_ols, ax=ax, label=r'$\mathbb{E}[e^u|X] \neq 1$')
sns.lineplot(x=n_sim, y=results_ppml, ax=ax, label=r'$\mathbb{E}[e^u|X] = 1$')
ax.set_xlabel('Sample Size n')
ax.set_ylabel(r'$\dfrac{1}{n}\sum_i^n \dfrac{y_i}{\hat{y}_i} - 1$')
ax.hlines(1, 100, 30000, color='red', linestyle='--', label='1')
ax.set_title('Convergence of test statistic')
plt.show()