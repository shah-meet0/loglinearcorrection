from loglinearcorrection.data_generating_processes import DGP, MVNDataGenerator, ConstantGenerator, CombinedDataGenerator, IndependentNormErrorGenerator
from loglinearcorrection.dependence_funcs import independent_absolute, constant_mean, independent_squared

import numpy as np
from loglinearcorrection.correction_estimator import CorrectedEstimator

# SET UP DGP

error = IndependentNormErrorGenerator(mean_fn=constant_mean(0), var_fn=lambda X: 2 * independent_absolute(X))
constant_generator = ConstantGenerator(1)
x_generator = MVNDataGenerator(means=[0], sigma=[[1]])
combined_generator = CombinedDataGenerator([x_generator, constant_generator])
betas = np.array([0.5, 1])

dgp = DGP(combined_generator, betas, error, exponential=True)
n_data = 10000
y, x, u = dgp.generate(n_data)

model = CorrectedEstimator(y, x, correction_model_type='nn', interest=0)
res = model.fit()
res.plot_dist_semi_elasticity()
res.average_semi_elasticity()
res.semi_elasticiy_at_average()
print(res.test_ppml())