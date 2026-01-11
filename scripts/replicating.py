import os
from datetime import datetime
import numpy as np
from loglinearcorrection.model import DoublyRobustElasticityEstimatorModel as DREEM
import json
import pandas as pd

INPUT_DIR = # add input here, google drive outputdata folder
output = # add output here
paper_list = np.load(r"\retrep\non_iv_reps.npy", allow_pickle=True) # i will give this to you separately
out = os.path.join(output, "coef_diff_test_high_dropout.csv")
unable = []


def run_replications(DIR):
    files = os.listdir(DIR)

    for file in files:
        if file.endswith('.ini'):
            continue

        folder_path = os.path.join(DIR, file)

        results_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if not f.endswith('.ini')]

        for result_path in results_paths:
            stripped_path = result_path.split('\\')
            print(f"Processing paper {stripped_path[-2]}, {stripped_path[-1]}")
            if stripped_path[-2] in ['018'] or stripped_path[-2] not in paper_list:
                print(f"Skipping paper {stripped_path[-2]} for now due to data issues.")
                continue
            try:
                metadata, X, y = process_paper(result_path)
            except Exception as e:
                print(f"Error loading data for {result_path}: {e}")
                continue

            try:
                results, mod = replicate(X, y, metadata)
                results['paper'] = stripped_path[-2]
                results['panel'] = stripped_path[-1]
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
                unable.append(stripped_path[-2] + "_" + stripped_path[-1])



def process_paper(result_path):
    metadata = os.path.join(result_path, 'metadata.json')
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    print(metadata)
    X = pd.read_parquet(os.path.join(result_path, 'X.parquet'))
    y = pd.read_parquet(os.path.join(result_path, 'y.parquet'))
    return metadata, X, y



def replicate(X, y, metadata):
    fe = metadata.get('fe', None)
    fe = [int(fixed_effect) for fixed_effect in fe] if fe is not None else None
    n= X.shape[0]
    model = DREEM(
        endog=y,
        exog=X,
        fixed_effects=fe,
        interest=metadata.get('interest')
    )

    n_folds = 3 if n > 5000 else 10

    depth = max(4, int(round(np.log(n))))  # ~ log(n)
    width = 20 * int(np.clip(round(n ** (1 / 6)), 5, 16))  # n^(1/6) so that H^2 ~ n^(1/3)


    arch_params_m = {
        'hidden_layers':  [width] * int(depth),
        'dropout': 0.2,
        'output_activation': 'identity'
    }

    fit_params_m = {
        "epochs": 300,
        "batch_size": max(n//50, 64),
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
        "patience": 40,
        "val_frac": 0.2 if n>2000 else 0.1,
        "num_workers": 0 if n < 5000 else 4
    }

    params = {
        'fit_params': {'weight_decay': 1e-3, "num_workers": 0 if n < 5000 else 4, 'epochs':200},
        'arch_params': {}
    }

    results = model.fit(n_folds=n_folds, m_params={'fit_params': fit_params_m, 'arch_params': arch_params_m},
                        density_params=params, fit_ppml=True)
    return dreem_summary_idx0_with_diff_se(results), results


def dreem_summary_idx0_with_diff_se(res) -> pd.Series:
    # resolve j=0 variable
    j = 0
    k = len(res.interest_indices)
    var_idx = res.interest_indices[j]
    name = res.exog_names[var_idx]
    # point estimates
    elast, se_elast = res.get_elasticity(name)
    ols = float(res.beta[0] if res.beta.shape[0] == k else res.beta[var_idx])
    has_ppml = getattr(res, "_ppml_fit", False) and (res.gamma is not None)
    ppml = float(
        res.gamma[0] if (has_ppml and res.gamma.shape[0] == k)
        else (res.gamma[var_idx] if has_ppml else np.nan)
    )

    # vcov scaled by n (same scaling as your tests)
    if res._variance_matrix is None:
        res.compute_variances()
    Vn = np.asarray(res._variance_matrix, float) / float(res.nobs)
    # contrast vectors r for (elast_j - ols_j), (elast_j - ppml_j), (ols_j - ppml_j)
    r_els_ols = np.zeros(Vn.shape[0])
    r_els_ols[j] = 1.0
    r_els_ols[k + j] = -1.0

    if has_ppml:
        r_els_ppml = np.zeros(Vn.shape[0])
        r_els_ppml[j] = 1.0
        r_els_ppml[2 * k + j] = -1.0

        r_ols_ppml = np.zeros(Vn.shape[0])
        r_ols_ppml[k + j] = 1.0
        r_ols_ppml[2 * k + j] = -1.0
    else:
        r_els_ppml = None
        r_ols_ppml = None

    # SEs of differences: sqrt(r' V r)
    se_diff_els_ols = float(np.sqrt(r_els_ols @ Vn @ r_els_ols))
    se_diff_els_ppml = float(np.sqrt(r_els_ppml @ Vn @ r_els_ppml)) if has_ppml else np.nan
    se_diff_ols_ppml = float(np.sqrt(r_ols_ppml @ Vn @ r_ols_ppml)) if has_ppml else np.nan

    # also grab the test stats/p-values you already compute
    test = res.difference_test(verbose=False)
    r_ols = test.results[j]           # Elasticity vs OLS
    if has_ppml:
        r_ppml = test.results[k + j]      # Elasticity vs PPML
        r_ols_ppml_test = test.results[2 * k + j]  # OLS vs PPML

    return pd.Series(
        {
            "variable_type": res.variable_types[var_idx],
            "elasticity": float(elast),
            "elasticity_se": float(se_elast),
            "ols_coef": ols,
            "ppml_coef": ppml if has_ppml else np.nan,

            "diff_elast_minus_ols": float(elast - ols),
            "se_diff_elast_minus_ols": se_diff_els_ols,
            "test_elast_vs_ols_stat": float(r_ols.statistic),
            "test_elast_vs_ols_p": float(r_ols.p_value),

            "diff_elast_minus_ppml": float(elast - ppml) if has_ppml else np.nan,
            "se_diff_elast_minus_ppml": se_diff_els_ppml,
            "test_elast_vs_ppml_stat": float(r_ppml.statistic) if has_ppml else np.nan,
            "test_elast_vs_ppml_p": float(r_ppml.p_value) if has_ppml else np.nan,

            "diff_ols_minus_ppml": float(ols - ppml) if has_ppml else np.nan,
            "se_diff_ols_minus_ppml": se_diff_ols_ppml,
            "test_ols_vs_ppml_stat": float(r_ols_ppml_test.statistic) if has_ppml else np.nan,
            "test_ols_vs_ppml_p": float(r_ols_ppml_test.p_value) if has_ppml else np.nan,

            "nobs": int(res.nobs),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        name=name,
    )

def main():
    return run_replications(INPUT_DIR)

df, res = main()