from loglinearcorrection.correction_estimator import CorrectedEstimator

import os
import sys
import pickle

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.formula.api import ols

from data_processing import load_and_process_data, prepare_stacked_data

data_path = "./raw-data/suarezserratoWhoBenefitsState2016/ILS_3_shocks_6_parm_housing.dta"  # Path to your Stata file

df = load_and_process_data(data_path)
df_stacked, outcomes =  prepare_stacked_data(df)

#------------------------------------
# Run their loglinear model (Table 4)
#------------------------------------

covars = [f"d_bus_dom2_{out}" for out in outcomes] + \
         [f"bartik_{out}" for out in outcomes]

# DECISION: remove empty weights
df_stacked = df_stacked.dropna(subset=['epop'])

# Create fixed effects
df_stacked['fe_outcome'] = df_stacked['fe_group'].astype(str) + '_' + \
                           df_stacked['outcome_idx'].astype(str)
df_stacked['year_outcome'] = df_stacked['year'].astype(str) + '_' + \
                             df_stacked['outcome_idx'].astype(str)

# Set up for PanelOLS (requires entity and time variables)
#df_stacked = df_stacked.set_index(['fips_state', 'year_outcome'])

# Create design matrix
X = df_stacked[covars]

# Add entity effects manually (since PanelOLS might not handle complex FE)
# Create dummy variables for fe_outcome
fe_dummies = pd.get_dummies(df_stacked['fe_outcome'], prefix='fe', drop_first=True, dtype=np.float64)
year_dummies = pd.get_dummies(df_stacked['year_outcome'], prefix='year', drop_first=True, dtype=np.float64).iloc[:,:-3]

# Add FE dummies to design matrix
X_with_fe = pd.concat([X, fe_dummies, year_dummies], axis=1)

# Run regression
y = df_stacked['outcome']
weights = df_stacked['epop'].values

# Use statsmodels WLS with cluster-robust standard errors
model = sm.WLS(y, X_with_fe, weights=weights)
results = model.fit(cov_type='cluster', cov_kwds={'groups': df_stacked.reset_index()['fips_state']})

# Extract coefficients and covariance matrix
coeffs = results.params[covars]  # Only keep the main covariates
cov_matrix = results.cov_params().loc[covars, covars]

#-------------------------
# Run PPML model
#-------------------------

yexp = np.exp(y)

model_poisson = sm.GLM(yexp, X_with_fe, weights=weights, family=sm.families.Poisson())
results_poisson = model_poisson.fit(cov_type='cluster', cov_kwds={'groups': df_stacked['fips_state']})
preds_poisson = results_poisson.predict(X_with_fe)

coeffs_poisson = results_poisson.params[covars]
cov_matrix_poisson = results_poisson.cov_params().loc[covars, covars]

#-------------------------------
# Run log-linear corrected model
#-------------------------------

def demean_for_fixed_effect(df, covariates, fe_col):
    result = df[covariates].copy()
    for var in covariates:
        group_means = df.groupby(fe_col)[var].transform('mean')
        result[var] = df[var] - group_means
    return result


def demean_multiple_fixed_effects(df, covariates, weights, fe_cols=None):
    """
    Simultaneously demean covariates for each unique combination of fixed effects with weights.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing covariates and fixed effects
    covariates : list
        List of covariate column names to demean
    weights : array-like
        Weights to use for demeaning
    fe_cols : list
        List of fixed effect column names
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with weighted-demeaned covariates
    """

    result = df[covariates].copy()
    
    # Add weights to dataframe for grouped operations
    df_with_weights = df.copy()
    df_with_weights['_weights'] = weights
    
    # Perform weighted demeaning for each covariate within each FE group
    for var in covariates:
        # Calculate weighted means for each unique combination of fixed effects
        weighted_sum = df_with_weights.groupby(fe_cols).apply(
            lambda x: (x[var] * x['_weights']).sum())
        weight_sum = df_with_weights.groupby(fe_cols)['_weights'].sum()
        weighted_means = weighted_sum / weight_sum
        
        # Create a mapping key for each row based on its FE values
        # This creates a unique identifier for each combination of FE values
        mapping_key = df_with_weights[fe_cols].apply(tuple, axis=1)
        
        # Map the weighted means back to the original rows
        # We convert the Series index to tuples to match our mapping key
        means_dict = dict(zip(weighted_means.index, weighted_means.values))
        mapped_means = mapping_key.map(means_dict)
        
        # Subtract weighted means
        result[var] = df[var] - mapped_means
    
    return result

demeanX = demean_multiple_fixed_effects(df_stacked, covars, weights=weights, fe_cols = ["fe_outcome","year_outcome"])

model_llc = CorrectedEstimator(yexp, demeanX, correction_model_type='ols', log_x=True, interest=3)

res_llc = model_llc.fit(weights=weights)

print(f"Standard log-log: {results.params[3]}")
print(f"PPML: {results_poisson.params[3]}")
print(f"Bias-corrected log-log: {res_llc.average_elasticity()}")

pd.DataFrame({
    'paper':'suarezserratoWhoBenefitsState2016',
    'loglog':results.params[3],
    'ppml':results_poisson.params[3],
    'bc':res_llc.average_elasticity(),
},index=[0]).to_csv('./output/replications/suarezserratoWhoBenefitsState2016.csv')

