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

print(x)
###########################

import pandas as pd

# Read the CSV file
file_path="../replications/data/Acemoglu_Restrepo/Acemoglu_Restrepo_Table2_logDV_data.csv"
df = pd.read_csv(file_path)
df = df.dropna()


wgt = df['wage_earners_1990']
print(df)

# Extract the columns you want initially
X_new = df[['expof_euro5_qo93_07', 'division', 'group_id']]

# Create dummy variables for 'division' and convert to integers
division_dummies = pd.get_dummies(X_new['division'], prefix='division', drop_first=True).astype(int)

# Create dummy variables for 'group_id' and convert to integers
group_id_dummies = pd.get_dummies(X_new['group_id'], prefix='group_id', drop_first=True).astype(int)

# Remove the original 'division' and 'group_id' columns
X_new = X_new.drop(['division', 'group_id'], axis=1)

# Concatenate the dummy variables with X_new
X_new = pd.concat([X_new, division_dummies, group_id_dummies], axis=1)

# Add a column of 1's named 'constant'
X_new['constant'] = 1

# Display the first few rows of X_new
print(X_new.head())

# Check the shape to see how many columns total
print(f"\nShape of X_new: {X_new.shape}")

# List all columns to verify all variables including the constant
print("\nColumns in X_new:")
print(X_new.columns.tolist())

Y_new = df[['d_hrwage_ln_1990_2008']]

y=Y_new
yexp=np.exp(y)
x=X_new

print(len(Y_new))
print(x)
print(len(wgt))


# Check that there are no NaN values left
print("NaN in x:", x.isnull().sum().sum())

#########################
# OLS:

import statsmodels.api as sm


# Run the OLS regression
model = sm.WLS(y, x, weights=wgt)
results = model.fit()

# Print the summary
print(results.summary())

print(x)

###################
# FOR SEMI_ELASTICITIES
# X needs to include intercept if needed
# X should be of the form such that log y = Beta * X + u, that is transformations must be applied to it such that log y is linear in the transforms.

model = CorrectedEstimator(yexp, x, correction_model_type='nn', interest=0)
res = model.fit(
    # params_dict={'degree':2}, 
    # weights=wgt
)
res.average_semi_elasticity()
res.semi_elasticity_at_average()
print(res.test_ppml())


# FOR SEMI_ELASTICITIES
# X needs to include intercept if needed
# X should be of the form such that log y = Beta * X + u, that is transformations must be applied to it such that log y is linear in the transforms.


# model = CorrectedEstimator(y, x, correction_model_type='ols', interest=0)
# res = model.fit({'degree':10})
# res.plot_dist_semi_elasticity()
# res.average_semi_elasticity()
# res.semi_elasticity_at_average()
# res.plot_eu_grad()
# res.plot_eu()
# print(res.test_ppml())
# 
# # FOR ELASTICITIES
# # X needs to include intercept if needed
# # X should be of the form such that log y = Beta * X + u, that is transformations must be applied to it such that log y is linear in the transforms.
# # In particular, X of interest should already be expressed as log(x)
# 
# model = CorrectedEstimator(y, x, correction_model_type='nn', interest=0, log_x=True)
# res = model.fit()
# res.plot_dist_elasticity()
# res.average_elasticity()
# res.elasticity_at_average()
# ols_results = res.get_ols_results()
# res.plot_eu()
# res.plot_eu_grad()
# res.print_ols_results()
# print(res.test_ppml())

