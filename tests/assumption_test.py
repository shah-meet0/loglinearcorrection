from loglinearcorrection.data_generating_processes import DGP, MVNDataGenerator, ConstantGenerator, CombinedDataGenerator, IndependentNormErrorGenerator
from loglinearcorrection.dependence_funcs import independent_absolute, constant_mean, independent_squared, independent_absolute_mean
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
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

error_ppml = IndependentNormErrorGenerator(mean_fn=lambda X: -1 * independent_absolute_mean(X),
                                           var_fn=lambda X: 2 * independent_absolute(X))

# Error Generator such that E[u|X] = 0

error_ols = IndependentNormErrorGenerator(mean_fn=constant_mean(0), var_fn=lambda X: 2 * independent_absolute(X))


# DGP for two cases, Y = exp(2X + 1 + u)
constant_generator = ConstantGenerator(1)
x_generator = MVNDataGenerator(means=[0], sigma=[[1]])
combined_generator = CombinedDataGenerator([x_generator, constant_generator])
betas = np.array([2, 3])

dgp_ppml = DGP(combined_generator, betas, error_ppml, exponential=True)
dgp_ols = DGP(combined_generator, betas, error_ols, exponential=True)

n_sim = 10
n_data = 50000
results_ppml = []
results_ols = []



for i in range(n_sim):
    try:
        print(i)
        # results_ppml.append(consistency(dgp_ppml, n_data))
        results_ols.append(consistency(dgp_ols, n_data))
    except:
        pass

results_ols = np.array(results_ols)
results_ppml = np.array(results_ppml)


