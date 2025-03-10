from utils.data_generating_processes import DGP, NormalErrorGenerator, MVNDataGenerator, ConstantGenerator, CombinedDataGenerator
from utils.dependence_funcs import independent_absolute, constant_mean
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# THIS MODULE IS FOR PRELIMNARY TESTING OF THE CONTINUOUS CORRECTION ALGORITHM
# THIS WILL PROBABLY BE DELETED IN A FUTURE PATCH, AS MORE FUNCTIONS WILL DEPRECATE ITS QUALITY

x_generator = MVNDataGenerator(means=[0], sigma=[[1]])
cons_generator = ConstantGenerator(1)
error_generator = NormalErrorGenerator(mean_fn=constant_mean(0), cov_fn=independent_absolute)
combined_generator = CombinedDataGenerator([x_generator, cons_generator])

betas = np.array([2, 1])

dgp = DGP(combined_generator, betas, error_generator, exponential=True)

x,y,u = dgp.generate(10000)


entries = []

# OLS MODEL

model_OLS = sm.OLS(np.log(y), x)
results_OLS = model_OLS.fit()
residuals = results_OLS.resid
print(results_OLS.summary())

# POISSON REGRESSION MODEL
model_poisson = sm.GLM(y, x, family=sm.families.Poisson())
results_poisson = model_poisson.fit()
print(results_poisson.summary())

# FIT USING DIFFERENT METHODS

target = np.exp(residuals)
input = x

validation_input = sm.tools.add_constant(np.linspace(-1.5, 1.5, 1000).reshape(-1, 1), prepend=False)
validation_target = np.exp(np.abs(validation_input[:, 0])/2)

# Linear Regression
model_OLS = LinearRegression()
model_OLS.fit(input, target)
preds_ols = model_OLS.predict(validation_input)

mse_ols = mean_squared_error(validation_target, preds_ols)
r2_ols = r2_score(validation_target, preds_ols)
print("Linear Regression")
print(f"MSE: {mse_ols:.3f}")
print(f"R2: {r2_ols:.3f}")
entries.append(["OLS", mse_ols, r2_ols])

# Random Forest
model_RF = RandomForestRegressor()
param_grid_rf = {
    "n_estimators": [100, 200, 250, 300, 350, 400],
    "max_depth": [3,4,5,6]
}
grid_search_rf = GridSearchCV(model_RF, param_grid_rf, cv=5, n_jobs=-1)
grid_search_rf.fit(input, target)
preds_rf = grid_search_rf.predict(validation_input)

mse_rf = mean_squared_error(validation_target, preds_rf)
r2_rf = r2_score(validation_target, preds_rf)

print("Random Forest")
print(f"MSE: {mse_rf:.3f}")
print(f"R2: {r2_rf:.3f}")
entries.append(["RF", mse_rf, r2_rf])


# Gradient Boosting

model_GB = GradientBoostingRegressor()
param_grid_gb = {
    'n_estimators': [50, 100, 150],            # Number of trees
    'learning_rate': [0.005, 0.01, 0.05, 0.1],         # Learning rate
    'max_depth': [4, 5, 6],                     # Maximum depth of individual trees
    'min_samples_split': [5, 10, 15, 20],            # Minimum samples to split a node
}

grid_search_gb = GridSearchCV(model_GB, param_grid_gb, cv=5, n_jobs=-1)
grid_search_gb.fit(input, target)
preds_gb = grid_search_gb.predict(validation_input)

mse_gb = mean_squared_error(validation_target, preds_gb)
r2_gb = r2_score(validation_target, preds_gb)

print("Gradient Boosting")
print(f"MSE: {mse_gb:.3f}")
print(f"R2: {r2_gb:.3f}")
entries.append(["GB", mse_gb, r2_gb])

# Neural Network

model_NN = MLPRegressor(early_stopping=True)
param_grid_nn = {
    'hidden_layer_sizes': [(128, 128, 128, 128), (200, 200, 200, 200), (256, 256, 256, 256)],
    'activation': ['relu'],
    'learning_rate': ['adaptive'],
}

grid_search_nn = GridSearchCV(model_NN, param_grid_nn, cv=5, n_jobs=-1)
grid_search_nn.fit(input, target)

preds_nn = grid_search_nn.predict(validation_input)

mse_nn = mean_squared_error(validation_target, preds_nn)
r2_nn = r2_score(validation_target, preds_nn)

print("Neural Network")
print(f"MSE: {mse_nn:.3f}")
print(f"R2: {r2_nn:.3f}")
entries.append(["NN", mse_nn, r2_nn])


df = pd.DataFrame(entries, columns=["Model", "MSE", "R2"])
df.to_latex()


# PLOT RESULTS

fig, ax = plt.subplots(2,2, sharey=True, sharex=True)
for axis in ax.flatten():
    sns.lineplot(x=validation_input[:, 0], y=validation_target, ax=axis, label='True', color='green')

sns.lineplot(x=validation_input[:, 0], y=preds_ols, ax=ax[0][0], label='OLS', color='red')
sns.lineplot(x=validation_input[:, 0], y=preds_rf, ax=ax[0][1], label='RF', color='red')
sns.lineplot(x=validation_input[:, 0], y=preds_gb, ax=ax[1][0], label='GB', color='red')
sns.lineplot(x=validation_input[:, 0], y=preds_nn, ax=ax[1][1], label='NN', color='red')
ax[0][0].set_title(f'OLS, MSE: {mse_ols:.3f}')
ax[0][1].set_title(f'Random Forest, MSE: {mse_rf:.3f}')
ax[1][0].set_title(f'Gradient Boosting, MSE: {mse_gb:.3f}')
ax[1][1].set_title(f'Neural Network, MSE: {mse_nn:.3f}')

ax[0][0].set_ylim(1, 3)
fig.savefig('./figures/continuous_correction.pdf', bbox_inches='tight', dpi=300, transparent=True)
