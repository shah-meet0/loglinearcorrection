from loglinearcorrection.data_generating_processes import DGP, MVNDataGenerator, ConstantGenerator, CombinedDataGenerator, IndependentNormErrorGenerator
from loglinearcorrection.dependence_funcs import independent_absolute, constant_mean, independent_squared, independent_absolute_mean
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


# Testing a model for appropriate model specification

def test_model(dgp, n):
    y,x, u = dgp.generate(n)
    model_ppml = sm.GLM(y, x, family=sm.families.Poisson())
    results_ppml = model_ppml.fit()
    yhat = model_ppml.predict(results_ppml.params, x)
    resids = y/yhat

    poly = PolynomialFeatures(degree=3, include_bias=False)
    x_poly = poly.fit_transform(x[:,0].reshape(-1,1))
    test_x = sm.tools.add_constant(x_poly, prepend=False)
    model_ols = sm.OLS(resids, test_x)
    results_ols = model_ols.fit()
    r_mat = np.eye(len(x_poly[0]) + 1)
    q = np.zeros(len(x_poly[0]) + 1)
    q[-1] = 1
    wald_test = results_ols.f_test((r_mat,q))

    return wald_test.pvalue, np.mean(np.exp(u))


# Error Generator such that E[e^u|X] = 1

error_ppml = IndependentNormErrorGenerator(mean_fn=lambda X: -1 * independent_absolute_mean(X),
                                           var_fn=lambda X: 2 * independent_absolute(X))

# Error Generator such that E[u|X] = 0

error_ols = IndependentNormErrorGenerator(mean_fn=constant_mean(0), var_fn=lambda X: 2 * independent_absolute(X))


# DGP for two cases, Y = exp(2X + 1 + u)
constant_generator = ConstantGenerator(1)
x_generator = MVNDataGenerator(means=[0], sigma=[[1]])
combined_generator = CombinedDataGenerator([x_generator, constant_generator])
betas = np.array([2, 1])

dgp_ppml = DGP(combined_generator, betas, error_ppml, exponential=True)
dgp_ols = DGP(combined_generator, betas, error_ols, exponential=True)

n_sim = 1000
n_data = 1000
results_ppml = []
results_ols = []



for i in range(n_sim):
    results_ppml.append(test_model(dgp_ppml, n_data))
    results_ols.append(test_model(dgp_ols, n_data))

results_ols = np.array(results_ols)
results_ppml = np.array(results_ppml)

print(f"OLS: {np.mean(results_ols[:,0] < 0.05)}")
print(f"PPML: {np.mean(results_ppml[:,0] < 0.05)}")
