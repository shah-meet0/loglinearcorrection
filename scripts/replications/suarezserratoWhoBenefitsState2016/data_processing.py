import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.formula.api import ols
from linearmodels import PanelOLS
import os

import pdb


# data_processing.py
# Replicate the Stata analysis from SZ 2016 to generate the required Excel files

def load_and_process_data(data_path):
    """
    Load the Stata dataset and perform initial processing
    """
    # Load the Stata file
    df = pd.read_stata(data_path)
    
    # Create fe_group variable (rust belt states in the 1980s)
    df['fe_group'] = df['fips_state'].copy()
    
    # Set fe_group to 0 for specific conditions
    conditions = [
        df['year'] > 1990,
        df['census_div'] == 1,
        df['census_div'] == 2,
        df['census_div'] == 5,
        df['census_div'] == 6,
        df['census_div'] == 7,
        df['census_div'] == 8,
        df['census_div'] == 9
    ]
    
    # Apply conditions
    for condition in conditions:
        df.loc[condition, 'fe_group'] = 0
    
    return df

def run_suest_equivalent(df, output_path):
    """
    Run individual regressions for each dependent variable
    This is equivalent to the suest approach in Stata
    """
    outcomes = ['dpop', 'dadjlwage', 'dadjlrent', 'dest']
    covariates = ['d_bus_dom2', 'bartik', 'd_esrate']
    
    # Run individual regressions
    results = {}
    for outcome in outcomes:
        # Create formula
        formula = f"{outcome} ~ " + " + ".join(covariates) + " + C(year) + C(fe_group)"
        
        # Run weighted regression
        model = ols(formula, data=df, weights=df['epop']).fit(
            cov_type='cluster', 
            cov_kwds={'groups': df['fips_state']}
        )
        results[outcome] = model
    
    return results

def prepare_stacked_data(df):
    """
    Reshape data for stacked regression approach
    """
    # Keep only necessary variables
    keep_vars = ['dpop', 'dadjlwage', 'dadjlrent', 'dest', 
                 'd_bus_dom2', 'bartik', 'd_esrate', 'year', 
                 'fe_group', 'epop', 'fips_state']
    
    df_subset = df[keep_vars].copy()
    
    # Rename for consistency
    df_subset.rename(columns={'dadjlwage': 'dwage', 'dadjlrent': 'drent'}, inplace=True)
    
    outcomes = ['dpop', 'dwage', 'drent', 'dest']
    covariates = ['d_bus_dom2', 'bartik', 'd_esrate', 'year', 'fe_group']
    
    # Create stacked dataset
    stacked_data = []
    
    for i, outcome in enumerate(outcomes):
        df_temp = df_subset.copy()
        df_temp['outcome'] = df_temp[outcome]
        df_temp['outcome_idx'] = i
        
        # Create interaction terms for each outcome
        for var in covariates:
            for j, out in enumerate(outcomes):
                df_temp[f"{var}_{out}"] = df_temp[var] if j == i else 0
        
        stacked_data.append(df_temp)
    
    # Combine all datasets
    df_stacked = pd.concat(stacked_data, ignore_index=True)
    
    return df_stacked, outcomes

def run_stacked_regression(df_stacked, outcomes, model_type, output_path):
    """
    Run stacked regression with fixed effects
    """
    # Define covariates based on model type
    if model_type == 'base':
        covars = [f"d_bus_dom2_{out}" for out in outcomes]
    elif model_type == 'bartik':
        covars = [f"d_bus_dom2_{out}" for out in outcomes] + \
                 [f"bartik_{out}" for out in outcomes]
    elif model_type == 'full':
        covars = [f"d_bus_dom2_{out}" for out in outcomes] + \
                 [f"bartik_{out}" for out in outcomes] + \
                 [f"d_esrate_{out}" for out in outcomes]

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
    weights = df_stacked['epop']
    
    # Use statsmodels WLS with cluster-robust standard errors
    pdb.set_trace()
    print(np.asarray(X_with_fe))
    model = sm.WLS(y, X_with_fe, weights=weights)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df_stacked.reset_index()['fips_state']})
    
    # Extract coefficients and covariance matrix
    coeffs = results.params[covars]  # Only keep the main covariates
    cov_matrix = results.cov_params().loc[covars, covars]
    
    # Export to Excel
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame([coeffs.values]).to_excel(writer, sheet_name='Sheet1', 
                                               index=False, header=False, startrow=0)
        cov_matrix.to_excel(writer, sheet_name='Sheet1', 
                           index=False, header=False, startrow=1)
    
    return coeffs, cov_matrix

def main(data_path, output_dir):
    """
    Main function to process data and run all estimations
    """
    print("Loading and processing data...")
    df = load_and_process_data(data_path)
    
    print("Running individual regressions (suest equivalent)...")
    # suest_results = run_suest_equivalent(df, output_dir)
    
    print("Preparing stacked data...")
    df_stacked, outcomes = prepare_stacked_data(df)
    
    print("Running base model...")
    coeffs_base, cov_base = run_stacked_regression(
        df_stacked.copy(), outcomes, 'base', 
        os.path.join(output_dir, 'base_est.xlsx')
    )
    
    print("Running base + Bartik model...")
    coeffs_bartik, cov_bartik = run_stacked_regression(
        df_stacked.copy(), outcomes, 'bartik', 
        os.path.join(output_dir, 'bartik_est.xlsx')
    )
    
    print("Running full model...")
    coeffs_full, cov_full = run_stacked_regression(
        df_stacked.copy(), outcomes, 'full', 
        os.path.join(output_dir, 'full_est.xlsx')
    )
    
    print("Data processing complete!")
    
    return {
        'base': (coeffs_base, cov_base),
        'bartik': (coeffs_bartik, cov_bartik),
        'full': (coeffs_full, cov_full)
    }

if __name__ == "__main__":
    # Set paths
    data_path = "ILS_3_shocks_6_parm_housing.dta"  # Path to your Stata file
    output_dir = "./estimates"  # Directory for output Excel files
    
    # Run the analysis
    results = main(data_path, output_dir)
    
    # Display summary
    print("\nModel Results Summary:")
    print("=====================")
    for model_name, (coeffs, cov) in results.items():
        print(f"\n{model_name.upper()} MODEL:")
        print(f"  Number of coefficients: {len(coeffs)}")
        print(f"  Covariance matrix shape: {cov.shape}")
        print(f"  First few coefficients:")
        for i, (var, coef) in enumerate(coeffs.head().items()):
            print(f"    {var}: {coef:.6f}")

