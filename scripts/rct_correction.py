import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm
import numpy as np
import seaborn as sns
data = pd.read_csv('./data/plfs_2017.csv')
data['Age2'] = data['Age']**2

# Drop rows with missing values
data_wages = data.query('Wage > 0')

X = data_wages[['Age','Age2', 'Education', 'Sex', 'Industry_Sector']]
Y = data_wages['Wage'].values

# One hot encode the categorical variables

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first', sparse_output=False), ['Education', 'Sex','Industry_Sector'])], remainder='passthrough')
X = ct.fit_transform(X)
X = pd.DataFrame(X, columns=ct.get_feature_names_out())
# Fit the model
model = sm.OLS(np.log(Y), sm.tools.add_constant(X))
results = model.fit()
print(results.summary())

sm.Poisson(Y, sm.tools.add_constant(X)).fit().summary()

# Residuals
residuals = results.resid
target = np.exp(residuals)
input = X

# Neural Network
from sklearn.neural_network import MLPRegressor

model_nn = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter=1000, early_stopping=True)
model_nn.fit(input, target)

sns.scatterplot(x=data_wages['Age'], y=model_nn.predict(input), style=data_wages['Industry_Sector'], hue=data_wages['Education'])
sns.scatterplot(x=data_wages['Age'], y=target, style=data_wages['Industry_Sector'], hue=data_wages['Education'])
sns.scatterplot(x=data_wages['Age'], y=residuals, style=data_wages['Industry_Sector'], hue=data_wages['Education'])