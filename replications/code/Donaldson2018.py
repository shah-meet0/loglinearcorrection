
import numpy as np
from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator

import pandas as pd
# SET UP DGP


file_path = "./replications/data/Donaldson_Railroads/Dave_Donaldson_Railroads_tbl4.csv"

df = pd.read_csv(file_path)

df = df.dropna()

print(len(df))

# Extract the columns you want initially
X_new = df[['distid','RAIL',  'year']]

# Create dummy variables for 'division' and convert to integers
dist_dummies = pd.get_dummies(X_new['distid'], prefix='distid', drop_first=True).astype(int)

# Create dummy variables for 'group_id' and convert to integers
year_dummies = pd.get_dummies(X_new['year'], prefix='year', drop_first=True).astype(int)

# Remove the original 'division' and 'group_id' columns
X_new = X_new.drop(['distid', 'year'], axis=1)

# Concatenate the dummy variables with X_new
X_new = pd.concat([X_new, dist_dummies, year_dummies], axis=1)

# Add a column of 1's named 'constant'
X_new['constant'] = 1

# Display the first few rows of X_new
print(X_new)

# Check the shape to see how many columns total
print(f"\nShape of X_new: {X_new.shape}")

# List all columns to verify all variables including the constant
print("\nColumns in X_new:")
print(X_new.columns.tolist())

Y_new = df[['ln_realincome']]

y=Y_new
yexp=np.exp(y)
x=X_new

print(len(Y_new))
print(x)


# Check that there are no NaN values left
print("NaN in x:", x.isnull().sum().sum())
print("NaN in y:", y.isnull().sum().sum())


#########################
# OLS:

import statsmodels.api as sm


# Run the OLS regression
model = sm.WLS(y, x)
results = model.fit()
model = sm.GLM(yexp, x, family=sm.families.Poisson())
results = model.fit()


# Print the summary
print(results.summary())

print(x)

###################
# FOR SEMI_ELASTICITIES
# X needs to include intercept if needed
# X should be of the form such that log y = Beta * X + u, that is transformations must be applied to it such that log y is linear in the transforms.

model = DoublyRobustElasticityEstimator(yexp, X_new, estimator_type='nn', interest=1, fe=[0,2])
res = model.fit()

res.summary()
print(res.test_ppml())

print("hello")



######
# Save standard errors
pd.DataFrame({'standard_errors': res.se}).to_csv('standard_errors_backup.csv', index=False)
print("Standard errors saved to standard_errors_backup.csv")

######

import plotly.express as px
import pandas as pd

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'X_first_column': x[:, 0] if isinstance(x, np.ndarray) else x.iloc[:, 0],
    'Standard_Errors': res.se
})

# Create the scatter plot with Plotly
fig = px.scatter(
    plot_data, 
    x='X_first_column', 
    y='Standard_Errors',
    title='Standard Errors vs First Column of X',
    opacity=0.7,
    trendline='ols'  # Adds a trend line
)

# Customize the layout
fig.update_layout(
    xaxis_title='First Column of X',
    yaxis_title='Semi-Elasticity',
    template='plotly_white',
    title='Donaldson (2018) Table 4 Column 1 Bias Correction',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    width=800,
    height=600
)

# Show the plot
fig.show()

##################

import matplotlib.pyplot as plt

# Extract the first column
if isinstance(x, np.ndarray):
    x_first_column = x[:, 0]
else:
    x_first_column = x.iloc[:, 0]

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(x_first_column, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Histogram of First Column of X')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

######################################

# Save x values and standard errors to a CSV file
data_to_save = pd.DataFrame({
    'X_first_column': x[:, 0] if isinstance(x, np.ndarray) else x.iloc[:, 0],
    'Standard_Errors': res.se
})

data_to_save.to_csv('x_and_se_values.csv', index=False)
print("Data saved to x_and_se_values.csv")
plt.tight_layout()
plt.show()


################
