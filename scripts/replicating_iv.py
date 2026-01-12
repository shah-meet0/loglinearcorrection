import os
from datetime import datetime
import numpy as np
from loglinearcorrection.iv_model import IVDoublyRobustElasticityEstimatorModel as IVDREEM
import json
import pandas as pd

INPUT_DIR = os.path.expanduser("~/Downloads/retrep/outputdata")
output = os.path.expanduser("~/Downloads/retrep/estimates")

# Placeholder: list of IV paper IDs to replicate
paper_list = np.array(['023', '050', '055', '079', '103', '125', '126', '127', '129','136', '143', '144', '164'])

out = os.path.join(output, "iv_coef_diff_test.csv")
unable = []


def run_replications(DIR):
    files = os.listdir(DIR)

    for file in files:
        if file.endswith('.ini'):
            continue

        folder_path = os.path.join(DIR, file)

        results_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if not f.endswith('.ini')]

        for result_path in results_paths:
            paper_id = os.path.basename(os.path.dirname(result_path))
            panel_id = os.path.basename(result_path)
            print(f"Processing paper {paper_id}, {panel_id}")
            
            if paper_id not in paper_list:
                print(f"Skipping paper {paper_id} - not in paper_list.")
                continue
            
            # Skip if no z.parquet (not an IV specification)
            if not os.path.exists(os.path.join(result_path, 'z.parquet')):
                print(f"Skipping {panel_id} - no z.parquet found.")
                continue
            
            try:
                metadata, X, y, Z = process_paper(result_path)
            except Exception as e:
                print(f"Error loading data for {result_path}: {e}")
                continue

            try:
                results, mod = replicate(X, y, Z, metadata)
                results['paper'] = paper_id
                results['panel'] = panel_id
                print(results)
                
                if os.path.exists(out):
                    df = pd.read_csv(out)
                    df = pd.concat([df, results.to_frame().T], ignore_index=True)
                else:
                    df = results.to_frame().T

                df.to_csv(out, index=False)
            except Exception as e:
                print(e)
                print(f"Error replicating for {result_path}")
                unable.append(paper_id + "_" + panel_id)


def process_paper(result_path):
    """Load metadata, X, y, and Z from result_path."""
    metadata_path = os.path.join(result_path, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(metadata)
    
    X = pd.read_parquet(os.path.join(result_path, 'X.parquet'), engine='pyarrow')
    y = pd.read_parquet(os.path.join(result_path, 'y.parquet'), engine='pyarrow')
    Z = pd.read_parquet(os.path.join(result_path, 'z.parquet'), engine='pyarrow')
    
    return metadata, X, y, Z


def replicate(X, y, Z, metadata):
    """Run IV-DREEM replication."""

    # Get endogenous regressor indices/names
    endogenous_regressors = metadata.get('endogenous_regressors', [])

    # Identify endogenous vs exogenous columns
    all_cols = list(X.columns)
    if isinstance(endogenous_regressors[0], str):
        endog_cols = endogenous_regressors
    else:
        endog_cols = [all_cols[i] for i in endogenous_regressors]

    # Get interest variable name
    original_interest = metadata.get('interest')
    if isinstance(original_interest, list):
        original_interest = original_interest[0]
    if isinstance(original_interest, str):
        interest_col_name = original_interest
    else:
        interest_col_name = all_cols[original_interest]

    # Check if interest is in endogenous - if not, add it
    if interest_col_name not in endog_cols:
        print(f"  Note: Interest '{interest_col_name}' not in endogenous, adding it")
        endog_cols = [interest_col_name] + endog_cols

    # Get FE column names (convert from indices if needed)
    fe_spec = metadata.get('fe', None)
    fe_col_names = None
    if fe_spec is not None:
        if isinstance(fe_spec[0], str):
            # Already column names - could be string numbers like '1', '2'
            # Check if they're numeric strings that should be indices
            try:
                fe_indices = [int(f) for f in fe_spec]
                fe_col_names = [all_cols[i] for i in fe_indices]
            except (ValueError, IndexError):
                # They're actual column names
                fe_col_names = fe_spec
        else:
            # Integer indices
            fe_col_names = [all_cols[i] for i in fe_spec]

    # Determine which columns go where:
    # - exog: endogenous regressors + FE columns (for demeaning)
    # - exog_control: other exogenous controls

    if fe_col_names is not None:
        # Include FE columns in exog so the model can demean them
        exog_cols = list(dict.fromkeys(endog_cols + fe_col_names))  # preserve order, remove duplicates
        control_cols = [c for c in all_cols if c not in exog_cols]
    else:
        exog_cols = endog_cols
        control_cols = [c for c in all_cols if c not in exog_cols]

    X_exog = X[exog_cols]
    X_control = X[control_cols] if len(control_cols) > 0 else None

    # Find interest index in new exog
    new_interest = list(X_exog.columns).index(interest_col_name)

    # Fixed effects as column names in X_exog
    fe_for_model = fe_col_names if fe_col_names is not None else None

    n = X.shape[0]

    # Initialize model
    model = IVDREEM(
        endog=y,
        exog=X_exog,
        instruments=Z,
        exog_control=X_control,
        fixed_effects=fe_for_model,
        interest=[new_interest]
    )
    
    # Adaptive hyperparameters
    n_folds = 3 if n > 5000 else 5
    
    depth = max(4, int(round(np.log(n))))
    width = 20 * int(np.clip(round(n ** (1 / 6)), 5, 16))
    
    # First stage params (g(Z) estimation)
    first_stage_params = {
        'arch_params': {
            'hidden_layers': [width] * depth,
            'dropout': 0.1,
            'output_activation': 'identity'
        },
        'fit_params': {
            'epochs': 200,
            'batch_size': max(n // 50, 64),
            'learning_rate': 1e-4,
            'weight_decay': 1e-3,
            'patience': 30,
            'verbose': False
        }
    }
    
    # m(X, V) nuisance params
    m_params = {
        'arch_params': {
            'hidden_layers': [width] * depth,
            'dropout': 0.2,
            'output_activation': 'softplus'
        },
        'fit_params': {
            'epochs': 300,
            'batch_size': max(n // 50, 64),
            'learning_rate': 1e-4,
            'weight_decay': 1e-3,
            'patience': 40,
            'val_frac': 0.2 if n > 2000 else 0.1,
            'num_workers': 0 if n < 5000 else 4,
            'verbose': False
        }
    }
    
    # Density/score params
    density_params = {
        'arch_params': {
            'shared': {'hidden_layers': [width] * depth}
        },
        'fit_params': {
            'weight_decay': 1e-3,
            'num_workers': 0 if n < 5000 else 4,
            'epochs': 200,
            'verbose': False
        }
    }
    
    # Omega (density ratio) params
    omega_params = {
        'arch_params': {
            'hidden_layers': [width // 2] * (depth - 1)
        },
        'fit_params': {
            'epochs': 100,
            'n_permutations': 5,
            'verbose': False
        }
    }
    
    # Lambda (Riesz representer) params
    lambda_params = {
        'arch_params': {
            'hidden_layers': [width // 4] * (depth - 2) if depth > 2 else [width // 4]
        },
        'fit_params': {
            'epochs': 50,
            'verbose': False
        }
    }
    
    # Fit model
    results = model.fit(
        n_folds=n_folds,
        first_stage_params=first_stage_params,
        m_params=m_params,
        density_params=density_params,
        omega_params=omega_params,
        lambda_params=lambda_params,
        n_mc_samples=300
    )
    
    return ivdreem_summary_idx0_with_diff_se(results), results


def ivdreem_summary_idx0_with_diff_se(res) -> pd.Series:
    """Extract summary statistics for the first interest variable."""
    
    j = 0
    k = len(res.interest_indices)
    var_idx = res.interest_indices[j]
    name = res.exog_names[var_idx]
    
    # Point estimates
    elast, se_elast = res.get_elasticity(name)
    beta = float(res.beta[0] if res.beta.shape[0] == k else res.beta[var_idx])
    rho = float(res.rho[0]) if res.rho is not None else np.nan
    
    # Variance matrix scaled by n
    if res._variance_matrix is None:
        res.compute_variances()
    Vn = np.asarray(res._variance_matrix, float) / float(res.nobs)
    
    # Contrast vector for (elasticity - beta)
    r_els_beta = np.zeros(Vn.shape[0])
    r_els_beta[j] = 1.0
    r_els_beta[k + j] = -1.0
    
    # SE of difference
    se_diff_els_beta = float(np.sqrt(r_els_beta @ Vn @ r_els_beta))
    
    # Hypothesis test
    test = res.difference_test(verbose=False)
    r_beta = test.results[j]
    
    return pd.Series(
        {
            "variable_type": res.variable_types.get(var_idx, "continuous"),
            "elasticity": float(elast),
            "elasticity_se": float(se_elast),
            "beta_coef": beta,
            "rho": rho,
            
            "diff_elast_minus_beta": float(elast - beta),
            "se_diff_elast_minus_beta": se_diff_els_beta,
            "test_elast_vs_beta_stat": float(r_beta.statistic),
            "test_elast_vs_beta_p": float(r_beta.p_value),
            
            "nobs": int(res.nobs),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        name=name,
    )


def main():
    return run_replications(INPUT_DIR)


if __name__ == "__main__":
    main()
