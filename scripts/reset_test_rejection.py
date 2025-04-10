from loglinearcorrection.data_generating_processes import DGP, MVNDataGenerator, ConstantGenerator, CombinedDataGenerator, IndependentNormErrorGenerator
from loglinearcorrection.dependence_funcs import independent_absolute, constant_mean, independent_squared
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

def reset_test(dgp, n):
    y, x, u = dgp.generate(n)
    model_ppml = sm.GLM(y, x, family=sm.families.Poisson())
    results_ppml = model_ppml.fit()
    yhat = results_ppml.mu.reshape(-1,1)
    model_ppml_RESET = sm.GLM(y, np.concatenate([x, yhat**2, yhat**3], axis=1), family=sm.families.Poisson())
    results_ppml_RESET = model_ppml_RESET.fit(cov_type='HC3')
    results_ppml_RESET.summary()
    resids = y/yhat
    return np.mean(resids), results_ppml.params[0]



# Error Generator such that E[e^u|X] = 1

error_ppml = IndependentNormErrorGenerator(mean_fn=lambda X: -1 * independent_absolute(X),
                                           var_fn=lambda X: 2 * independent_absolute(X))

# Error Generator such that E[u|X] = 0

error_ols = IndependentNormErrorGenerator(mean_fn=constant_mean(0), var_fn=lambda X: 2 * independent_absolute(X))


# DGP for two cases, Y = exp(2X + 1 + u)
constant_generator = ConstantGenerator(1)
x_generator = MVNDataGenerator(means=[2], sigma=[[1]])
combined_generator = CombinedDataGenerator([x_generator, constant_generator])
betas = np.array([0.5, 1])

dgp_ppml = DGP(combined_generator, betas, error_ppml, exponential=True)
dgp_ols = DGP(combined_generator, betas, error_ols, exponential=True)

n_sim = np.linspace(50, 10000, 1000).astype(int)
results_ppml = []
results_ols = []

for n in n_sim:
    print(f'Calculating n = {n}')
    results_ppml.append(reset_test(dgp_ppml, n))
    results_ols.append(reset_test(dgp_ols, n))

results_ols = np.array(results_ols)
results_ppml = np.array(results_ppml)


