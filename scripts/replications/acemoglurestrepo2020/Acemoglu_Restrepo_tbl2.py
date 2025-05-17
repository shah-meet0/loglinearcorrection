import os
from loglinearcorrection.data_generating_processes import DGP, MVNDataGenerator, ConstantGenerator, CombinedDataGenerator, IndependentNormErrorGenerator
from loglinearcorrection.dependence_funcs import independent_absolute, constant_mean, independent_squared

import numpy as np
from loglinearcorrection.correction_estimator import CorrectedEstimator

import pandas as pd

# SET UP DGP
file_path = "./replications/data/Donaldson_Railroads/Dave_Donaldson_Railroads_tbl4.csv"
file_path="./replications/data/Acemoglu_Restrepo/Acemoglu_Restrepo_Table2_logDV_data.csv"

df = pd.read_csv(file_path)

df = df.dropna()

print(len(df))

print(df.columns)

# Extract the columns you want initially
#X_reg1 = df[['expof_euro5_qo93_07', 'division', 'groupid']]
X=df[['expof_euro5_qo93_07', 'division', 'group_id', 'ipums_female_1990', 'ipums_hispanic_1990', 'ipums_white_1990', 'ipums_black_1990', 'ipums_asian_1990', 'ipums_highschool_1990', 'ipums_college_1990', 'ipums_masters_1990', 'ipums_above65_1990', 'ipums_logpop_1990']]
Y = df['d_hrwage_ln_1990_2008']
weights=df['wage_earners_1990']

# Create dummy variables for 'division' and convert to integers
dummies1 = pd.get_dummies(X['division'], prefix='division', drop_first=True).astype(int)

# Create dummy variables for 'group_id' and convert to integers
dummies2 = pd.get_dummies(X['group_id'], prefix='group_id', drop_first=True).astype(int)

# Remove the original 'division' and 'group_id' columns
X = X.drop(['division', 'group_id'], axis=1)

# Concatenate the dummy variables with X_new
X = pd.concat([X, dummies1, dummies2], axis=1)

# Add a column of 1's named 'constant'
X['constant'] = 1

# Split into Regressions 1 and 2
print(X.columns)

X_reg1=X.drop(['ipums_female_1990', 'ipums_hispanic_1990', 'ipums_white_1990', 'ipums_black_1990', 'ipums_asian_1990', 'ipums_highschool_1990', 'ipums_college_1990', 'ipums_masters_1990', 'ipums_above65_1990', 'ipums_logpop_1990'], axis=1)

X_reg2=X

# Display the first few rows of X_new
print(X_reg1)

# Check the shape to see how many columns total
print(f"\nShape of X_new: {X_reg1.shape}")

# List all columns to verify all variables including the constant
print("\nColumns in X_new:")
print(X_reg1.columns.tolist())


y=Y
yexp=np.exp(y)
x=X_reg1

print(len(y))
print(x)


# Check that there are no NaN values left
print("NaN in x:", x.isnull().sum().sum())
print("NaN in y:", y.isnull().sum().sum())


#########################
# OLS:

import statsmodels.api as sm


# Run the OLS regression
model = sm.WLS(y, x, weights=weights)
results = model.fit()

# Print the summary
print(results.summary())

print(x)

###################
# FOR SEMI_ELASTICITIES
# X needs to include intercept if needed
# X should be of the form such that log y = Beta * X + u, that is transformations must be applied to it such that log y is linear in the transforms.

model = CorrectedEstimator(yexp, x, correction_model_type='binary', interest=0)
res = model.fit(params_dict={"degree":3}, weights=weights)

res.corrected_percentage_change()
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
