import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf # For formula-based GLM
from statsmodels.regression.linear_model import RegressionResultsWrapper
# from statsmodels.genmod.families import Poisson # Not needed directly
# from statsmodels.genmod.families.links import Log # Not needed directly
import patsy
from scipy import sparse # Not actively used in this version, but kept from original
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# For high-dimensional fixed effects (used in run_ddd_regressions)
try:
    import linearmodels
    from linearmodels.panel import PanelOLS
    linearmodels_available = True
except ImportError:
    print("Warning: linearmodels package not available. Install with 'pip install linearmodels' for OLS DDD with HDFE.")
    linearmodels_available = False

# For Log-Linear Corrected Estimator
try:
    from loglinearcorrection.correction_estimator import CorrectedEstimator
    loglinearcorrection_available = True
except ImportError:
    print("Warning: loglinearcorrection package not available. Install it to run the corrected estimator.")
    loglinearcorrection_available = False


# Define directory structure
dir_base = "./"
dir_raw_data_base = os.path.join(dir_base, "raw-data/besedesPhaseOutTariffs2020") # For actual raw inputs
dir_estimates_base = os.path.join(dir_base, "output/replications/besedesPhaseOutTariffs2020")

# Specific output subdirectories
dir_processed_data = os.path.join(dir_estimates_base, "processed_data")
dir_figures_tables = os.path.join(dir_estimates_base, "figuresTables")
# Appendix data directory (assuming it's an input like raw_data)
dir_appendix = os.path.join(dir_raw_data_base, "appendix")


# Create directories if they don't exist
for directory in [dir_raw_data_base, dir_estimates_base, dir_processed_data, dir_figures_tables, dir_appendix]:
    os.makedirs(directory, exist_ok=True)

# Function to prepare data for DDD analyses
def prepare_ddd_data(file_type):
    """
    Prepare data for difference-in-difference-in-differences analyses
    Saves processed files to dir_processed_data.

    Args:
        file_type: String, either 'rawCodes' or 'consistentCodes'
    """
    print(f"Preparing DDD data with {file_type}...")

    # Load the main dataset from raw_data directory
    raw_input_file_path = os.path.join(dir_raw_data_base, f"data_{file_type}.dta")
    try:
        df = pd.read_stata(raw_input_file_path)
    except FileNotFoundError:
        print(f"Error: File data_{file_type}.dta not found in {dir_raw_data_base}")
        print(f"Creating dummy data for testing prepare_ddd_data for {file_type}...")
        data = {
            'country': ['Mexico', 'Canada', 'USA', 'Germany', 'China', 'Australia', 'Bahrain', 'Chile', 'Colombia', 'Israel'] * 10,
            'year': np.repeat(range(1990, 2000), 10),
            'hs8': [f'prod{i}' for i in range(100)],
            'r_cv': np.random.rand(100) * 1000,
            'q': np.random.rand(100) * 100,
            'catEffMex': np.random.choice(['A', 'B', 'C', 'D', 'GSP', 'Mixed phasein (incl D)'], 100),
            'catEffCan': np.random.choice(['A', 'B', 'C', 'D', 'Mixed phasein (no D)'], 100)
        }
        df = pd.DataFrame(data)
        df['r_cv'] = df['r_cv'].abs() + 1
        df['q'] = df['q'].abs() + 1
        # Save this dummy to raw_data to allow script to proceed if it was originally expected there
        # This part is tricky; ideally, the script shouldn't write back to raw_data.
        # However, if this function IS the first step creating it from an even rawer source, this might be okay.
        # For this replication, we assume data_{file_type}.dta is a given input.
        # So, if not found, the dummy is just for in-memory processing.
        # To be safe, we won't save the dummy back to raw_input_file_path here.

    # Identify non-NAFTA FTA partners
    non_nafta_countries = [
        "Australia", "Bahrain", "Chile", "Colombia", "Israel", "Jordan",
        "Korea", "Morocco", "Oman", "Panama", "Peru", "Singapore",
        "Dominican Rep", "Costa Rica", "Honduras", "Nicaragua",
        "Guatemala", "El Salvador"
    ]
    df['nonNAFTA_FTApartner'] = df['country'].isin(non_nafta_countries).astype(int)

    # Identify Canada's FTA partners
    canada_fta_partners = [
        "Israel", "Chile", "Costa Rica", "Iceland", "Liechtenstein", "Norway",
        "Switzerland", "Peru", "Colombia", "Jordan", "Panama", "Honduras", "Korea"
    ]
    df['CanadaFTApartner'] = df['country'].isin(canada_fta_partners).astype(int)

    # Identify Mexico's FTA partners
    mexico_fta_partners = [
        "Colombia", "Chile", "Israel", "Iceland", "Liechtenstein", "Norway",
        "Switzerland", "Uruguay", "Japan", "Peru", "Panama", "Costa Rica",
        "Nicaragua", "Honduras", "El Salvador", "Belize", "Guatemala",
        "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
        "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
        "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
        "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
        "Spain", "Sweden", "United Kingdom"
    ]
    df['MexicoFTApartner'] = df['country'].isin(mexico_fta_partners).astype(int)

    for x in ['Mex', 'MexC', 'MexF', 'Can', 'CanC', 'CanF']:
        df[f'{x}_cv'] = df['r_cv']
        df[f'{x}_q'] = df['q']

    df.loc[df['nonNAFTA_FTApartner'] == 1, 'Mex_cv'] = np.nan
    df.loc[df['country'] == 'Canada', 'Mex_cv'] = np.nan
    df.loc[df['nonNAFTA_FTApartner'] == 1, 'Mex_q'] = np.nan
    df.loc[df['country'] == 'Canada', 'Mex_q'] = np.nan

    df.loc[df['nonNAFTA_FTApartner'] == 1, 'MexC_cv'] = np.nan
    df.loc[df['country'] == 'Canada', 'MexC_cv'] = np.nan
    df.loc[df['country'] == 'China', 'MexC_cv'] = np.nan
    df.loc[df['nonNAFTA_FTApartner'] == 1, 'MexC_q'] = np.nan
    df.loc[df['country'] == 'Canada', 'MexC_q'] = np.nan
    df.loc[df['country'] == 'China', 'MexC_q'] = np.nan

    df.loc[df['nonNAFTA_FTApartner'] == 1, 'MexF_cv'] = np.nan
    df.loc[df['country'] == 'Canada', 'MexF_cv'] = np.nan
    df.loc[df['MexicoFTApartner'] == 1, 'MexF_cv'] = np.nan
    df.loc[df['nonNAFTA_FTApartner'] == 1, 'MexF_q'] = np.nan
    df.loc[df['country'] == 'Canada', 'MexF_q'] = np.nan
    df.loc[df['MexicoFTApartner'] == 1, 'MexF_q'] = np.nan

    df.loc[df['nonNAFTA_FTApartner'] == 1, 'Can_cv'] = np.nan
    df.loc[df['country'] == 'Mexico', 'Can_cv'] = np.nan
    df.loc[df['nonNAFTA_FTApartner'] == 1, 'Can_q'] = np.nan
    df.loc[df['country'] == 'Mexico', 'Can_q'] = np.nan

    df.loc[df['nonNAFTA_FTApartner'] == 1, 'CanC_cv'] = np.nan
    df.loc[df['country'] == 'Mexico', 'CanC_cv'] = np.nan
    df.loc[df['country'] == 'China', 'CanC_cv'] = np.nan
    df.loc[df['nonNAFTA_FTApartner'] == 1, 'CanC_q'] = np.nan
    df.loc[df['country'] == 'Mexico', 'CanC_q'] = np.nan
    df.loc[df['country'] == 'China', 'CanC_q'] = np.nan

    df.loc[df['nonNAFTA_FTApartner'] == 1, 'CanF_cv'] = np.nan
    df.loc[df['country'] == 'Mexico', 'CanF_cv'] = np.nan
    df.loc[df['CanadaFTApartner'] == 1, 'CanF_cv'] = np.nan
    df.loc[df['nonNAFTA_FTApartner'] == 1, 'CanF_q'] = np.nan
    df.loc[df['country'] == 'Mexico', 'CanF_q'] = np.nan
    df.loc[df['CanadaFTApartner'] == 1, 'CanF_q'] = np.nan

    for x in ['Mex', 'MexC', 'MexF', 'Can', 'CanC', 'CanF']:
        df[f'{x}_uv'] = np.divide(df[f'{x}_cv'], df[f'{x}_q'], out=np.full_like(df[f'{x}_cv'], np.nan, dtype=np.float64), where=df[f'{x}_q']!=0)
        df[f'{x}_uv'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Save processed data to dir_processed_data
    df.to_pickle(os.path.join(dir_processed_data, f"data2_{file_type}.pkl"))
    df_stata = df.copy()
    for col in df_stata.columns:
        if df_stata[col].dtype in [np.float32, np.float64]:
            df_stata[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    try:
        df_stata.to_stata(os.path.join(dir_processed_data, f"data2_{file_type}.dta"), write_index=False, version=118)
        print(f"Successfully saved Stata file data2_{file_type}.dta to {dir_processed_data}")
    except ValueError as e:
        print(f"Warning: Could not save data2_{file_type}.dta to Stata format due to: {e}. Using pickle format primarily.")

    mex_df = df.copy()
    mex_df = mex_df[~mex_df['catEffMex'].isin(['Mixed phasein (incl D)', 'Mixed phasein (no D)',
                                             'missing code (see notes)', 'see notes'])]
    mex_df = mex_df[~mex_df['catEffMex'].isna()]
    columns_to_drop_mex = [col for col in mex_df.columns if 'Can' in col and col not in ['CanadaFTApartner', 'catEffCan']]
    mex_df = mex_df.drop(columns=columns_to_drop_mex)
    mex_df['treatCDF'] = (mex_df['catEffMex'] == 'D').astype(int)
    mex_df['treatNonCDF'] = (mex_df['catEffMex'] != 'D').astype(int)
    mex_df['treatImmed'] = (mex_df['catEffMex'] == 'A').astype(int)
    mex_df['treatGSP'] = (mex_df['catEffMex'] == 'GSP').astype(int)
    mex_df['treatPhase5'] = ((mex_df['catEffMex'] == 'B') | (mex_df['catEffMex'] == 'B6')).astype(int)
    mex_df['treatPhase10'] = ((mex_df['catEffMex'] == 'C') | (mex_df['catEffMex'] == 'C+') |
                             (mex_df['catEffMex'] == 'C10')).astype(int)
    mex_df['NAFTA'] = (mex_df['country'] == 'Mexico').astype(int)
    mex_df['cntry'] = pd.factorize(mex_df['country'])[0] + 1
    mex_df.to_pickle(os.path.join(dir_processed_data, f"data_Mex_{file_type}.pkl"))
    try:
        mex_stata = mex_df.copy()
        for col in mex_stata.columns:
            if mex_stata[col].dtype in [np.float32, np.float64]:
                mex_stata[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        mex_stata.to_stata(os.path.join(dir_processed_data, f"data_Mex_{file_type}.dta"), write_index=False, version=118)
    except ValueError as e:
        print(f"Warning: Could not save Mexico data to Stata format due to: {e}")

    can_df = df.copy()
    can_df = can_df[~can_df['catEffCan'].isin(['Mixed phasein (incl D)', 'Mixed phasein (no D)',
                                             'missing code (see notes)', 'see notes'])]
    can_df = can_df[~can_df['catEffCan'].isna()]
    columns_to_drop_can = [col for col in can_df.columns if 'Mex' in col and col not in ['MexicoFTApartner', 'catEffMex']]
    can_df = can_df.drop(columns=columns_to_drop_can)
    can_df['treatCDF'] = (can_df['catEffCan'] == 'D').astype(int)
    can_df['treatNonCDF'] = (can_df['catEffCan'] != 'D').astype(int)
    can_df['treatImmed'] = (can_df['catEffCan'] == 'A').astype(int)
    can_df['treatPhase5'] = ((can_df['catEffCan'] == 'B') | (can_df['catEffCan'] == 'B6')).astype(int)
    can_df['treatPhase10'] = ((can_df['catEffCan'] == 'C') | (can_df['catEffCan'] == 'C+') |
                             (can_df['catEffCan'] == 'C10')).astype(int)
    can_df['treatGSP'] = 0
    can_df['NAFTA'] = (can_df['country'] == 'Canada').astype(int)
    can_df['cntry'] = pd.factorize(can_df['country'])[0] + 1
    can_df.to_pickle(os.path.join(dir_processed_data, f"data_Can_{file_type}.pkl"))
    try:
        can_stata = can_df.copy()
        for col in can_stata.columns:
            if can_stata[col].dtype in [np.float32, np.float64]:
                can_stata[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        can_stata.to_stata(os.path.join(dir_processed_data, f"data_Can_{file_type}.dta"), write_index=False, version=118)
    except ValueError as e:
        print(f"Warning: Could not save Canada data to Stata format due to: {e}")

    # Rename files to match Stata convention, in dir_processed_data
    if file_type == 'rawCodes':
        for ext in ['.pkl', '.dta']:
            old_path = os.path.join(dir_processed_data, f"data2_{file_type}{ext}")
            if os.path.exists(old_path): os.rename(old_path, os.path.join(dir_processed_data, f"dataRC{ext}"))
            old_path_mex = os.path.join(dir_processed_data, f"data_Mex_{file_type}{ext}")
            if os.path.exists(old_path_mex): os.rename(old_path_mex, os.path.join(dir_processed_data, f"dataRC_Mex{ext}"))
            old_path_can = os.path.join(dir_processed_data, f"data_Can_{file_type}{ext}")
            if os.path.exists(old_path_can): os.rename(old_path_can, os.path.join(dir_processed_data, f"dataRC_Can{ext}"))
    elif file_type == 'consistentCodes':
        for ext in ['.pkl', '.dta']:
            old_path = os.path.join(dir_processed_data, f"data2_{file_type}{ext}")
            if os.path.exists(old_path): os.rename(old_path, os.path.join(dir_processed_data, f"dataCC{ext}"))
            old_path_mex = os.path.join(dir_processed_data, f"data_Mex_{file_type}{ext}")
            if os.path.exists(old_path_mex): os.rename(old_path_mex, os.path.join(dir_processed_data, f"dataCC_Mex{ext}"))
            old_path_can = os.path.join(dir_processed_data, f"data_Can_{file_type}{ext}")
            if os.path.exists(old_path_can): os.rename(old_path_can, os.path.join(dir_processed_data, f"dataCC_Can{ext}"))
    print(f"Finished preparing DDD data for {file_type}. Outputs in {dir_processed_data}.")


# Function for t-test (means, SEs, N)
def run_ttest(data, value_col, filter_cond, group_col):
    subset = data[filter_cond & data[value_col].notna()].copy()
    group1 = subset[subset[group_col] == 0][value_col]
    group2 = subset[subset[group_col] == 1][value_col]

    if len(group1) < 2 or len(group2) < 2:
        return {
            'mean1': group1.mean() if len(group1) > 0 else np.nan,
            'mean2': group2.mean() if len(group2) > 0 else np.nan,
            'diff': (group2.mean() - group1.mean()) if (len(group1) > 0 and len(group2) > 0) else np.nan,
            'se1': group1.std() / np.sqrt(len(group1)) if len(group1) > 0 else np.nan,
            'se2': group2.std() / np.sqrt(len(group2)) if len(group2) > 0 else np.nan,
            'se_diff': np.nan,
            'n1': len(group1), 'n2': len(group2)
        }
    tt_res = stats.ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
    se1 = group1.std(ddof=1) / np.sqrt(len(group1)) if len(group1) > 0 else np.nan # ddof=1 for sample std
    se2 = group2.std(ddof=1) / np.sqrt(len(group2)) if len(group2) > 0 else np.nan
    se_diff = np.sqrt(se1**2 + se2**2) if not np.isnan(se1) and not np.isnan(se2) else np.nan
    return {
        'mean1': group1.mean(), 'mean2': group2.mean(),
        'diff': group2.mean() - group1.mean(),
        'se1': se1, 'se2': se2, 'se_diff': se_diff,
        'n1': len(group1), 'n2': len(group2)
    }

# Function for running OLS DD regression
def run_ols_dd_regression(data, y_log, treat_var, time_var, filter_cond=None):
    subset = data.copy()
    if filter_cond is not None:
        subset = subset[filter_cond]
    subset = subset.dropna(subset=[y_log, treat_var, time_var]).copy()
    if subset.empty or subset[y_log].isna().all() or len(subset) < 4 : # Need enough data for const, treat, time, interaction
        return {'coef': np.nan, 'se': np.nan}
    subset['treat_time'] = subset[treat_var].astype(float) * subset[time_var].astype(float)
    X = subset[[treat_var, time_var, 'treat_time']].astype(float)
    X = sm.add_constant(X)
    y_vals = subset[y_log].astype(float)
    try:
        model = sm.OLS(y_vals, X)
        results = model.fit(cov_type='HC1')
        return {'coef': results.params['treat_time'], 'se': results.bse['treat_time']}
    except Exception as e:
        print(f"Error in OLS DD regression ({y_log} ~ {treat_var}*{time_var}): {e}")
        return {'coef': np.nan, 'se': np.nan}

# Function for running OLS DDD regression
def run_ols_ddd_regression(data, y_log, treat_var, time_var, diff_var):
    data_reg = data.dropna(subset=[y_log, treat_var, time_var, diff_var]).copy()
    if data_reg.empty or data_reg[y_log].isna().all() or len(data_reg) < 8: # Min obs for DDD
        return {'coef': np.nan, 'se': np.nan}
    formula = f"{y_log} ~ {treat_var} * {time_var} * {diff_var}"
    try:
        model = smf.ols(formula=formula, data=data_reg)
        results = model.fit(cov_type='HC1')
        interaction_term = None
        possible_terms = [ # Patsy can order terms differently
            f"{treat_var}:{time_var}:{diff_var}", f"{treat_var}:{diff_var}:{time_var}",
            f"{time_var}:{treat_var}:{diff_var}", f"{time_var}:{diff_var}:{treat_var}",
            f"{diff_var}:{treat_var}:{time_var}", f"{diff_var}:{time_var}:{treat_var}"
        ]
        # Also handle patsy's T. for categorical
        cat_treat = f"C({treat_var})[T.1.0]" if isinstance(data_reg[treat_var].dtype, pd.CategoricalDtype) or len(data_reg[treat_var].unique()) > 2 else treat_var
        cat_time = f"C({time_var})[T.1.0]" if isinstance(data_reg[time_var].dtype, pd.CategoricalDtype) or len(data_reg[time_var].unique()) > 2 else time_var
        cat_diff = f"C({diff_var})[T.1.0]" if isinstance(data_reg[diff_var].dtype, pd.CategoricalDtype) or len(data_reg[diff_var].unique()) > 2 else diff_var

        possible_terms_cat = [
            f"{cat_treat}:{cat_time}:{cat_diff}", f"{cat_treat}:{cat_diff}:{cat_time}",
            f"{cat_time}:{cat_treat}:{cat_diff}", f"{cat_time}:{cat_diff}:{cat_treat}",
            f"{cat_diff}:{cat_treat}:{cat_time}", f"{cat_diff}:{cat_time}:{cat_treat}"
        ]
        all_possible_terms = possible_terms + possible_terms_cat

        for term in all_possible_terms:
            if term in results.params.index:
                interaction_term = term
                break
        if not interaction_term: # More general fallback
            for param_name in results.params.index:
                if treat_var in param_name and time_var in param_name and diff_var in param_name and param_name.count(':') == 2:
                    interaction_term = param_name
                    break
        if interaction_term:
            return {'coef': results.params[interaction_term],'se': results.bse[interaction_term]}
        else:
            print(f"Warning: OLS DDD interaction term not found for {formula}. Params: {results.params.index}")
            return {'coef': np.nan, 'se': np.nan}
    except Exception as e:
        print(f"Error in OLS DDD regression ({formula}): {e}")
        return {'coef': np.nan, 'se': np.nan}

# Function for running PPML DD regression
def run_ppml_dd_regression(data, y_level, treat_var, time_var, filter_cond=None):
    subset = data.copy()
    if filter_cond is not None:
        subset = subset[filter_cond]
    subset = subset[subset[y_level] >= 0] # PPML requires non-negative y
    subset = subset.dropna(subset=[y_level, treat_var, time_var]).copy()
    if subset.empty or subset[y_level].sum() == 0 or len(subset) < 4:
        return {'coef': np.nan, 'se': np.nan}
    # Ensure vars are 0/1 for formula interpretation without C() if not already categorical
    formula = f"{y_level} ~ {treat_var} * {time_var}"
    try:
        model = smf.glm(formula=formula, data=subset, family=sm.families.Poisson(link=sm.families.links.Log()))
        results = model.fit(cov_type='HC1', disp=0) # disp=0 for Poisson
        interaction_term_name = None
        # Try to find interaction: treat_var:time_var (or similar if patsy renames/treats as categorical)
        possible_terms = [f"{treat_var}:{time_var}", f"{time_var}:{treat_var}"]
        # Add C() wrapped versions for robustness
        possible_terms += [f"C({treat_var})[T.1.0]:C({time_var})[T.1.0]", f"C({time_var})[T.1.0]:C({treat_var})[T.1.0]"]


        for term in possible_terms:
            if term in results.params.index:
                interaction_term_name = term
                break
        if not interaction_term_name: # More general fallback
             for param_name in results.params.index:
                if treat_var in param_name and time_var in param_name and ":" in param_name and not treat_var == param_name and not time_var == param_name :
                    interaction_term_name = param_name
                    break

        if interaction_term_name:
            return {'coef': results.params[interaction_term_name], 'se': results.bse[interaction_term_name]}
        else:
            print(f"Warning: PPML DD interaction term not found for {formula}. Params: {results.params.index}")
            return {'coef': np.nan, 'se': np.nan}
    except Exception as e:
        print(f"Error in PPML DD regression ({formula}): {e}")
        return {'coef': np.nan, 'se': np.nan}

# Function for running PPML DDD regression
def run_ppml_ddd_regression(data, y_level, treat_var, time_var, diff_var):
    data_reg = data[data[y_level] >= 0]
    data_reg = data_reg.dropna(subset=[y_level, treat_var, time_var, diff_var]).copy()
    if data_reg.empty or data_reg[y_level].sum() == 0 or len(data_reg) < 8:
        return {'coef': np.nan, 'se': np.nan}
    formula = f"{y_level} ~ {treat_var} * {time_var} * {diff_var}"
    try:
        model = smf.glm(formula=formula, data=data_reg, family=sm.families.Poisson(link=sm.families.links.Log()))
        results = model.fit(cov_type='HC1', disp=0)
        interaction_term_name = None
        # Same logic as OLS DDD for finding the term
        possible_terms = [
            f"{treat_var}:{time_var}:{diff_var}", f"{treat_var}:{diff_var}:{time_var}",
            f"{time_var}:{treat_var}:{diff_var}", f"{time_var}:{diff_var}:{treat_var}",
            f"{diff_var}:{treat_var}:{time_var}", f"{diff_var}:{time_var}:{treat_var}"
        ]
        # Add C() wrapped versions for robustness
        cat_treat = f"C({treat_var})[T.1.0]"
        cat_time = f"C({time_var})[T.1.0]"
        cat_diff = f"C({diff_var})[T.1.0]"
        possible_terms_cat = [
            f"{cat_treat}:{cat_time}:{cat_diff}", f"{cat_treat}:{cat_diff}:{cat_time}",
            f"{cat_time}:{cat_treat}:{cat_diff}", f"{cat_time}:{cat_diff}:{cat_treat}",
            f"{cat_diff}:{cat_treat}:{cat_time}", f"{cat_diff}:{cat_time}:{cat_treat}"
        ]
        all_possible_terms = possible_terms + possible_terms_cat

        for term in all_possible_terms:
            if term in results.params.index:
                interaction_term_name = term
                break
        if not interaction_term_name: # More general fallback
            for param_name in results.params.index:
                if treat_var in param_name and time_var in param_name and diff_var in param_name and param_name.count(':') == 2:
                    interaction_term_name = param_name
                    break
        if interaction_term_name:
            return {'coef': results.params[interaction_term_name], 'se': results.bse[interaction_term_name]}
        else:
            print(f"Warning: PPML DDD interaction term not found for {formula}. Params: {results.params.index}")
            return {'coef': np.nan, 'se': np.nan}
    except Exception as e:
        print(f"Error in PPML DDD regression ({formula}): {e}")
        return {'coef': np.nan, 'se': np.nan}

# Function for running Log-Linear Corrected DD regression
def run_corrected_dd_regression(data, y_level_col, treat_var, time_var, filter_cond=None):
    if not loglinearcorrection_available:
        print("Loglinearcorrection package not available. Skipping corrected DD regression.")
        return {'coef': np.nan, 'se': np.nan}

    subset = data.copy()
    if filter_cond is not None:
        subset = subset[filter_cond]

    subset = subset[subset[y_level_col] >= 0]
    subset = subset.dropna(subset=[y_level_col, treat_var, time_var]).copy()

    if subset.empty or subset[y_level_col].sum() == 0 or len(subset) < 20: # Min obs for stability
        # print(f"Skipping corrected DD for {y_level_col}: not enough data after filtering.")
        return {'coef': np.nan, 'se': np.nan}

    y_val = subset[y_level_col].astype(float).values # CorrectedEstimator expects numpy array typically
    
    X_df = pd.DataFrame()
    X_df[treat_var] = subset[treat_var].astype(float).values
    X_df[time_var] = subset[time_var].astype(float).values
    X_df['interaction_term'] = X_df[treat_var] * X_df[time_var]
    
    # Add constant and ensure all variables are part of the final X matrix fed to estimator
    # The estimator itself might handle intercept, or it should be included.
    # The example showed X_new['constant'] = 1, so we include it.
    X_for_estimator = sm.add_constant(X_df[[treat_var, time_var, 'interaction_term']], prepend=True)
    # Columns are: 'const', treat_var, time_var, 'interaction_term'
    
    try:
        # The 'interest' parameter is the column index of the variable of interest in X_for_estimator
        interest_idx = X_for_estimator.columns.get_loc('interaction_term')
        
        model = CorrectedEstimator(y_val, X_for_estimator.values, correction_model_type='ols', interest=interest_idx)
        res = model.fit(params_dict={"degree": 1})
        return {'coef': res.average_semi_elasticity(), 'se': np.nan} # SE not available
    except Exception as e:
        print(f"Error in CorrectedEstimator DD regression ({y_level_col} ~ {treat_var}*{time_var}): {e}")
        # import traceback
        # traceback.print_exc()
        return {'coef': np.nan, 'se': np.nan}

# Function for running Log-Linear Corrected DDD regression
def run_corrected_ddd_regression(data, y_level_col, treat_var, time_var, diff_var):
    if not loglinearcorrection_available:
        print("Loglinearcorrection package not available. Skipping corrected DDD regression.")
        return {'coef': np.nan, 'se': np.nan}

    data_reg = data.copy()
    data_reg = data_reg[data_reg[y_level_col] >= 0]
    data_reg = data_reg.dropna(subset=[y_level_col, treat_var, time_var, diff_var]).copy()

    if data_reg.empty or data_reg[y_level_col].sum() == 0 or len(data_reg) < 30: # Min obs
        # print(f"Skipping corrected DDD for {y_level_col}: not enough data.")
        return {'coef': np.nan, 'se': np.nan}

    y_val = data_reg[y_level_col].astype(float).values
    
    # Use patsy to create the design matrix including intercept and all interactions
    # Ensure variables are treated by their names as they are already 0/1
    formula = f"~ {treat_var} * {time_var} * {diff_var}" 
    X_patsy = patsy.dmatrix(formula, data_reg, return_type='dataframe')

    # Find the triple interaction term name (robustly)
    interaction_term_name_found = None
    # Try specific combinations first (patsy often sorts terms alphabetically within an interaction)
    term_elements = sorted([treat_var, time_var, diff_var])
    specific_patsy_term = f"{term_elements[0]}:{term_elements[1]}:{term_elements[2]}"

    if specific_patsy_term in X_patsy.columns:
        interaction_term_name_found = specific_patsy_term
    else: # Fallback search
        for col_name in X_patsy.columns:
            is_treat_present = treat_var in col_name
            is_time_present = time_var in col_name
            is_diff_present = diff_var in col_name
            
            # For C() wrapped variables if patsy created them
            is_cat_treat_present = f"C({treat_var})[T." in col_name or treat_var == col_name
            is_cat_time_present = f"C({time_var})[T." in col_name or time_var == col_name
            is_cat_diff_present = f"C({diff_var})[T." in col_name or diff_var == col_name
            
            if (is_treat_present and is_time_present and is_diff_present and col_name.count(':') == 2) or \
               (is_cat_treat_present and is_cat_time_present and is_cat_diff_present and col_name.count(':') == 2) :
                interaction_term_name_found = col_name
                break
    
    if not interaction_term_name_found:
        print(f"Error: Could not find triple interaction term for {treat_var}*{time_var}*{diff_var} in X_patsy columns: {X_patsy.columns}")
        return {'coef': np.nan, 'se': np.nan}

    try:
        interest_idx = X_patsy.columns.get_loc(interaction_term_name_found)
        model = CorrectedEstimator(y_val, X_patsy.values, correction_model_type='ols', interest=interest_idx)
        res = model.fit(params_dict={"degree": 1})
        return {'coef': res.average_semi_elasticity(), 'se': np.nan} # SE not available
    except Exception as e:
        print(f"Error in CorrectedEstimator DDD regression ({y_level_col} ~ {treat_var}*{time_var}*{diff_var}): {e}")
        # import traceback
        # traceback.print_exc()
        return {'coef': np.nan, 'se': np.nan}


# Function to create Table 2
def create_table_2(file_suffix='CC'):
    """
    Creates Table 2: Time-invariant DDD estimates of NAFTA (OLS, PPML, and Corrected Log-Linear)
    Reads data from dir_processed_data. Saves table to dir_figures_tables.

    Args:
        file_suffix: String, file suffix to use (e.g., 'CC' for consistent codes)
    """
    print(f"Creating Table 2 (OLS, PPML, Corrected) with {file_suffix} data...")

    excel_file_path = os.path.join(dir_figures_tables, f"table_2_{file_suffix}.xlsx")
    writer = pd.ExcelWriter(excel_file_path, engine='openpyxl')

    # Define common row indices based on the detailed OLS header structure for easier mapping
    # These are 0-indexed for DataFrame loc, corresponding to Excel rows if header is row 1
    # Panel A: NAFTA vs ROW approach
    # A1. Phase-out products
    row_a1_nafta_mean = 7; row_a1_nafta_se = 8; row_a1_nafta_n = 9
    row_a1_row_mean = 11; row_a1_row_se = 12; row_a1_row_n = 13 # Adjusted to match typical table layout
    row_a1_dd_coef = 14; row_a1_dd_se = 15

    # A2. CDF products
    row_a2_nafta_mean = 24; row_a2_nafta_se = 25; row_a2_nafta_n = 26 # Adjusted from original script's example based on typical Table 2 structure
    row_a2_row_mean = 28; row_a2_row_se = 29; row_a2_row_n = 30
    row_a2_dd_coef = 31; row_a2_dd_se = 32
    row_a_ddd_coef = 33; row_a_ddd_se = 34 # DDD for Panel A

    # Panel B: Phase-out vs CDF-products approach
    # (Using an offset for Panel B based on typical Table 2 structure. Original script had hardcoded '37' for 'b1_phase' mean)
    # Let's assume Panel B starts around row 41 in the Excel sheet (index 40 for df.loc)
    # Adjust these if the exact header structure dictates different absolute row numbers.
    # For consistency, let's use the header_ols.shape[0] to verify. The example headers_ols has 60 lines.
    # The original script used:
    # Panel B, B1. NAFTA partner
    row_b1_phase_mean = 41; row_b1_phase_se = 42; row_b1_phase_n = 43 # Phase-out products
    row_b1_cdf_mean = 44; row_b1_cdf_se = 45; row_b1_cdf_n = 46       # CDF products
    row_b1_dd_coef = 47; row_b1_dd_se = 48                           # DD (Phase-out vs CDF for NAFTA partner)

    # Panel B, B2. ROW
    row_b2_phase_mean = 52; row_b2_phase_se = 53; row_b2_phase_n = 54 # Phase-out products
    row_b2_cdf_mean = 55; row_b2_cdf_se = 56; row_b2_cdf_n = 57       # CDF products
    row_b2_dd_coef = 58; row_b2_dd_se = 59                           # DD (Phase-out vs CDF for ROW)
    row_b_ddd_coef = 60; row_b_ddd_se = 61                           # DDD for Panel B

    # --- OLS Section ---
    print("Processing OLS estimates for Table 2...")
    headers_ols = pd.DataFrame({ # Structure from original, adjust if needed
        'A': ["Table 2: Time-invariant DDD estimates of NAFTA (OLS on log values)", "", "Dependent Variable: Log of Customs Value", "", "Panel A: NAFTA vs ROW approach", "", "A1. Phase-out products", "Mexico", "Pre-NAFTA", "Post-NAFTA", "Growth", "NAFTA partner", "", "", "", "ROW", "", "", "", "DD (OLS)", "", "", "A2. CDF products", "Mexico", "Pre-NAFTA", "Post-NAFTA", "Growth", "NAFTA partner", "", "", "", "ROW", "", "", "", "DD (OLS)", "", "DDD (OLS)", "", "(continued on next page)", "", "Table 2 (continued)", "", "Panel B: Phase-out vs CDF-products approach", "", "B1. NAFTA partner", "Mexico", "Pre-NAFTA", "Post-NAFTA", "Growth", "Phase-out products", "", "", "", "CDF products", "", "", "", "DD (OLS)", "", "", "B2. ROW", "Mexico", "Pre-NAFTA", "Post-NAFTA", "Growth", "Phase-out products", "", "", "", "CDF products", "", "", "", "DD (OLS)", "", "DDD (OLS)", ""],
        'E': ["", "", "", "", "", "", "", "Canada", "Pre-NAFTA", "Post-NAFTA", "Growth", "NAFTA partner", "", "", "", "ROW", "", "", "", "DD (OLS)", "", "", "", "Canada", "Pre-NAFTA", "Post-NAFTA", "Growth", "NAFTA partner", "", "", "", "ROW", "", "", "", "DD (OLS)", "", "DDD (OLS)", "", "", "", "", "", "", "", "", "Canada", "Pre-NAFTA", "Post-NAFTA", "Growth", "Phase-out products", "", "", "", "CDF products", "", "", "", "DD (OLS)", "", "", "", "Canada", "Pre-NAFTA", "Post-NAFTA", "Growth", "Phase-out products", "", "", "", "CDF products", "", "", "", "DD (OLS)", "", "DDD (OLS)", ""]
    }) # This header has 63 rows. Max index 62.
    # Let's re-verify the indices based on the user's "original" output example structure which is more compact.
    # The actual excel filling might be less dense than this header implies or the header needs manual mapping.
    # For now, I will use the indices that match the user's visual output example for OLS,
    # then replicate that pattern for PPML and Corrected.

    # Updated row indices based on typical compact Table 2 visual output (Panel A, A1 has 8 lines of data + DD/DDD)
    idx_map = {
        'a1_nafta_mean': 7, 'a1_nafta_se': 8, 'a1_nafta_n': 9,
        'a1_row_mean': 10, 'a1_row_se': 11, 'a1_row_n': 12,
        'a1_dd_coef': 13, 'a1_dd_se': 14,

        'a2_nafta_mean': 18, 'a2_nafta_se': 19, 'a2_nafta_n': 20,
        'a2_row_mean': 21, 'a2_row_se': 22, 'a2_row_n': 23,
        'a2_dd_coef': 24, 'a2_dd_se': 25,
        'a_ddd_coef': 26, 'a_ddd_se': 27,

        'b1_phase_mean': 37, 'b1_phase_se': 38, 'b1_phase_n': 39,
        'b1_cdf_mean': 40, 'b1_cdf_se': 41, 'b1_cdf_n': 42,
        'b1_dd_coef': 43, 'b1_dd_se': 44,

        'b2_phase_mean': 48, 'b2_phase_se': 49, 'b2_phase_n': 50,
        'b2_cdf_mean': 51, 'b2_cdf_se': 52, 'b2_cdf_n': 53,
        'b2_dd_coef': 54, 'b2_dd_se': 55,
        'b_ddd_coef': 56, 'b_ddd_se': 57,
    }
    max_rows_needed = max(idx_map.values()) + 2 # A bit of buffer
    results_ols_df = pd.DataFrame(index=range(max_rows_needed), columns=list('ABCDEFGHIJ'))


    headers_ols.to_excel(writer, sheet_name=f'Table2_OLS_Input{file_suffix}', index=False, header=False)
    results_ols = {}

    # Process Mexico OLS data (same as before)
    try:
        mexico_file_path_pkl = os.path.join(dir_processed_data, f"data{file_suffix}_Mex.pkl")
        mexico_file_path_dta = os.path.join(dir_processed_data, f"data{file_suffix}_Mex.dta")
        if os.path.exists(mexico_file_path_pkl): mex_df = pd.read_pickle(mexico_file_path_pkl)
        elif os.path.exists(mexico_file_path_dta): mex_df = pd.read_stata(mexico_file_path_dta)
        else: raise FileNotFoundError("Mexico data file not found.")
        
        mex_df['NAFTA'] = (mex_df['country'] == 'Mexico').astype(int); mex_df['ROW'] = (~mex_df['NAFTA'].astype(bool)).astype(int)
        mex_df['phase'] = (mex_df['catEffMex'] != 'D').astype(int); mex_df['CDF'] = (mex_df['catEffMex'] == 'D').astype(int)
        mex_df['pre'] = (mex_df['year'] < 1993).astype(int); mex_df['post'] = (mex_df['year'] >= 1993).astype(int)
        mex_df['lnMex_cv'] = np.log(mex_df['Mex_cv'].replace(0, np.nan)); mex_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        results_ols['mex'] = {
            'a1_nafta': run_ttest(mex_df, 'lnMex_cv', (mex_df['phase'] == 1) & (mex_df['NAFTA'] == 1), 'post'),
            'a1_row': run_ttest(mex_df, 'lnMex_cv', (mex_df['phase'] == 1) & (mex_df['ROW'] == 1), 'post'),
            'a1_dd': run_ols_dd_regression(mex_df, 'lnMex_cv', 'NAFTA', 'post', (mex_df['phase'] == 1)),
            'a2_nafta': run_ttest(mex_df, 'lnMex_cv', (mex_df['CDF'] == 1) & (mex_df['NAFTA'] == 1), 'post'),
            'a2_row': run_ttest(mex_df, 'lnMex_cv', (mex_df['CDF'] == 1) & (mex_df['ROW'] == 1), 'post'),
            'a2_dd': run_ols_dd_regression(mex_df, 'lnMex_cv', 'NAFTA', 'post', (mex_df['CDF'] == 1)),
            'a_ddd': run_ols_ddd_regression(mex_df, 'lnMex_cv', 'NAFTA', 'post', 'phase'),
            'b1_phase': run_ttest(mex_df, 'lnMex_cv', (mex_df['phase'] == 1) & (mex_df['NAFTA'] == 1), 'post'), # same as a1_nafta
            'b1_cdf': run_ttest(mex_df, 'lnMex_cv', (mex_df['CDF'] == 1) & (mex_df['NAFTA'] == 1), 'post'),   # same as a2_nafta
            'b1_dd': run_ols_dd_regression(mex_df, 'lnMex_cv', 'phase', 'post', (mex_df['NAFTA'] == 1)),
            'b2_phase': run_ttest(mex_df, 'lnMex_cv', (mex_df['phase'] == 1) & (mex_df['ROW'] == 1), 'post'),     # same as a1_row
            'b2_cdf': run_ttest(mex_df, 'lnMex_cv', (mex_df['CDF'] == 1) & (mex_df['ROW'] == 1), 'post'),       # same as a2_row
            'b2_dd': run_ols_dd_regression(mex_df, 'lnMex_cv', 'phase', 'post', (mex_df['ROW'] == 1)),
            'b_ddd': run_ols_ddd_regression(mex_df, 'lnMex_cv', 'NAFTA', 'post', 'phase') # Same as a_ddd
        }
    except Exception as e: print(f"Error processing Mexico OLS data: {e}"); results_ols['mex'] = {}

    # Process Canada OLS data (same as before)
    try:
        canada_file_path_pkl = os.path.join(dir_processed_data, f"data{file_suffix}_Can.pkl")
        canada_file_path_dta = os.path.join(dir_processed_data, f"data{file_suffix}_Can.dta")
        if os.path.exists(canada_file_path_pkl): can_df = pd.read_pickle(canada_file_path_pkl)
        elif os.path.exists(canada_file_path_dta): can_df = pd.read_stata(canada_file_path_dta)
        else: raise FileNotFoundError("Canada data file not found.")

        can_df['NAFTA'] = (can_df['country'] == 'Canada').astype(int); can_df['ROW'] = (~can_df['NAFTA'].astype(bool)).astype(int)
        can_df['phase'] = (can_df['catEffCan'] != 'D').astype(int); can_df['CDF'] = (can_df['catEffCan'] == 'D').astype(int)
        can_df['pre'] = (can_df['year'] < 1993).astype(int); can_df['post'] = (can_df['year'] >= 1993).astype(int)
        can_df['lnCan_cv'] = np.log(can_df['Can_cv'].replace(0, np.nan)); can_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        results_ols['can'] = {
            'a1_nafta': run_ttest(can_df, 'lnCan_cv', (can_df['phase'] == 1) & (can_df['NAFTA'] == 1), 'post'),
            'a1_row': run_ttest(can_df, 'lnCan_cv', (can_df['phase'] == 1) & (can_df['ROW'] == 1), 'post'),
            'a1_dd': run_ols_dd_regression(can_df, 'lnCan_cv', 'NAFTA', 'post', (can_df['phase'] == 1)),
            'a2_nafta': run_ttest(can_df, 'lnCan_cv', (can_df['CDF'] == 1) & (can_df['NAFTA'] == 1), 'post'),
            'a2_row': run_ttest(can_df, 'lnCan_cv', (can_df['CDF'] == 1) & (can_df['ROW'] == 1), 'post'),
            'a2_dd': run_ols_dd_regression(can_df, 'lnCan_cv', 'NAFTA', 'post', (can_df['CDF'] == 1)),
            'a_ddd': run_ols_ddd_regression(can_df, 'lnCan_cv', 'NAFTA', 'post', 'phase'),
            'b1_phase': run_ttest(can_df, 'lnCan_cv', (can_df['phase'] == 1) & (can_df['NAFTA'] == 1), 'post'),
            'b1_cdf': run_ttest(can_df, 'lnCan_cv', (can_df['CDF'] == 1) & (can_df['NAFTA'] == 1), 'post'),
            'b1_dd': run_ols_dd_regression(can_df, 'lnCan_cv', 'phase', 'post', (can_df['NAFTA'] == 1)),
            'b2_phase': run_ttest(can_df, 'lnCan_cv', (can_df['phase'] == 1) & (can_df['ROW'] == 1), 'post'),
            'b2_cdf': run_ttest(can_df, 'lnCan_cv', (can_df['CDF'] == 1) & (can_df['ROW'] == 1), 'post'),
            'b2_dd': run_ols_dd_regression(can_df, 'lnCan_cv', 'phase', 'post', (can_df['ROW'] == 1)),
            'b_ddd': run_ols_ddd_regression(can_df, 'lnCan_cv', 'NAFTA', 'post', 'phase')
        }
    except Exception as e: print(f"Error processing Canada OLS data: {e}"); results_ols['can'] = {}

    if 'mex' in results_ols and results_ols['mex'] and 'can' in results_ols and results_ols['can']:
        def format_val(val, format_str="{:.3f}"):
            if pd.isna(val) or val == "": return ""
            try: return format_str.format(float(val))
            except (ValueError, TypeError): return str(val)
        
        # Mexico OLS
        m = results_ols['mex']
        results_ols_df.loc[idx_map['a1_nafta_mean'], 'B'] = format_val(m['a1_nafta']['mean1']); results_ols_df.loc[idx_map['a1_nafta_mean'], 'C'] = format_val(m['a1_nafta']['mean2']); results_ols_df.loc[idx_map['a1_nafta_mean'], 'D'] = format_val(m['a1_nafta']['diff'])
        results_ols_df.loc[idx_map['a1_nafta_se'], 'B'] = f"({format_val(m['a1_nafta']['se1'])})"; results_ols_df.loc[idx_map['a1_nafta_se'], 'C'] = f"({format_val(m['a1_nafta']['se2'])})"; results_ols_df.loc[idx_map['a1_nafta_se'], 'D'] = f"({format_val(m['a1_nafta']['se_diff'])})"
        results_ols_df.loc[idx_map['a1_nafta_n'], 'B'] = f"[{m['a1_nafta']['n1']}]"; results_ols_df.loc[idx_map['a1_nafta_n'], 'C'] = f"[{m['a1_nafta']['n2']}]"
        results_ols_df.loc[idx_map['a1_row_mean'], 'B'] = format_val(m['a1_row']['mean1']); results_ols_df.loc[idx_map['a1_row_mean'], 'C'] = format_val(m['a1_row']['mean2']); results_ols_df.loc[idx_map['a1_row_mean'], 'D'] = format_val(m['a1_row']['diff'])
        results_ols_df.loc[idx_map['a1_row_se'], 'B'] = f"({format_val(m['a1_row']['se1'])})"; results_ols_df.loc[idx_map['a1_row_se'], 'C'] = f"({format_val(m['a1_row']['se2'])})"; results_ols_df.loc[idx_map['a1_row_se'], 'D'] = f"({format_val(m['a1_row']['se_diff'])})"
        results_ols_df.loc[idx_map['a1_row_n'], 'B'] = f"[{m['a1_row']['n1']}]"; results_ols_df.loc[idx_map['a1_row_n'], 'C'] = f"[{m['a1_row']['n2']}]"
        results_ols_df.loc[idx_map['a1_dd_coef'], 'D'] = format_val(m['a1_dd']['coef']); results_ols_df.loc[idx_map['a1_dd_se'], 'D'] = f"({format_val(m['a1_dd']['se'])})"
        results_ols_df.loc[idx_map['a2_nafta_mean'], 'B'] = format_val(m['a2_nafta']['mean1']); results_ols_df.loc[idx_map['a2_nafta_mean'], 'C'] = format_val(m['a2_nafta']['mean2']); results_ols_df.loc[idx_map['a2_nafta_mean'], 'D'] = format_val(m['a2_nafta']['diff'])
        results_ols_df.loc[idx_map['a2_nafta_se'], 'B'] = f"({format_val(m['a2_nafta']['se1'])})"; results_ols_df.loc[idx_map['a2_nafta_se'], 'C'] = f"({format_val(m['a2_nafta']['se2'])})"; results_ols_df.loc[idx_map['a2_nafta_se'], 'D'] = f"({format_val(m['a2_nafta']['se_diff'])})"
        results_ols_df.loc[idx_map['a2_nafta_n'], 'B'] = f"[{m['a2_nafta']['n1']}]"; results_ols_df.loc[idx_map['a2_nafta_n'], 'C'] = f"[{m['a2_nafta']['n2']}]"
        results_ols_df.loc[idx_map['a2_row_mean'], 'B'] = format_val(m['a2_row']['mean1']); results_ols_df.loc[idx_map['a2_row_mean'], 'C'] = format_val(m['a2_row']['mean2']); results_ols_df.loc[idx_map['a2_row_mean'], 'D'] = format_val(m['a2_row']['diff'])
        results_ols_df.loc[idx_map['a2_row_se'], 'B'] = f"({format_val(m['a2_row']['se1'])})"; results_ols_df.loc[idx_map['a2_row_se'], 'C'] = f"({format_val(m['a2_row']['se2'])})"; results_ols_df.loc[idx_map['a2_row_se'], 'D'] = f"({format_val(m['a2_row']['se_diff'])})"
        results_ols_df.loc[idx_map['a2_row_n'], 'B'] = f"[{m['a2_row']['n1']}]"; results_ols_df.loc[idx_map['a2_row_n'], 'C'] = f"[{m['a2_row']['n2']}]"
        results_ols_df.loc[idx_map['a2_dd_coef'], 'D'] = format_val(m['a2_dd']['coef']); results_ols_df.loc[idx_map['a2_dd_se'], 'D'] = f"({format_val(m['a2_dd']['se'])})"
        results_ols_df.loc[idx_map['a_ddd_coef'], 'D'] = format_val(m['a_ddd']['coef']); results_ols_df.loc[idx_map['a_ddd_se'], 'D'] = f"({format_val(m['a_ddd']['se'])})"
        results_ols_df.loc[idx_map['b1_phase_mean'], 'B'] = format_val(m['b1_phase']['mean1']); results_ols_df.loc[idx_map['b1_phase_mean'], 'C'] = format_val(m['b1_phase']['mean2']); results_ols_df.loc[idx_map['b1_phase_mean'], 'D'] = format_val(m['b1_phase']['diff'])
        results_ols_df.loc[idx_map['b1_phase_se'], 'B'] = f"({format_val(m['b1_phase']['se1'])})"; results_ols_df.loc[idx_map['b1_phase_se'], 'C'] = f"({format_val(m['b1_phase']['se2'])})"; results_ols_df.loc[idx_map['b1_phase_se'], 'D'] = f"({format_val(m['b1_phase']['se_diff'])})"
        results_ols_df.loc[idx_map['b1_phase_n'], 'B'] = f"[{m['b1_phase']['n1']}]"; results_ols_df.loc[idx_map['b1_phase_n'], 'C'] = f"[{m['b1_phase']['n2']}]"
        results_ols_df.loc[idx_map['b1_cdf_mean'], 'B'] = format_val(m['b1_cdf']['mean1']); results_ols_df.loc[idx_map['b1_cdf_mean'], 'C'] = format_val(m['b1_cdf']['mean2']); results_ols_df.loc[idx_map['b1_cdf_mean'], 'D'] = format_val(m['b1_cdf']['diff'])
        results_ols_df.loc[idx_map['b1_cdf_se'], 'B'] = f"({format_val(m['b1_cdf']['se1'])})"; results_ols_df.loc[idx_map['b1_cdf_se'], 'C'] = f"({format_val(m['b1_cdf']['se2'])})"; results_ols_df.loc[idx_map['b1_cdf_se'], 'D'] = f"({format_val(m['b1_cdf']['se_diff'])})"
        results_ols_df.loc[idx_map['b1_cdf_n'], 'B'] = f"[{m['b1_cdf']['n1']}]"; results_ols_df.loc[idx_map['b1_cdf_n'], 'C'] = f"[{m['b1_cdf']['n2']}]"
        results_ols_df.loc[idx_map['b1_dd_coef'], 'D'] = format_val(m['b1_dd']['coef']); results_ols_df.loc[idx_map['b1_dd_se'], 'D'] = f"({format_val(m['b1_dd']['se'])})"
        results_ols_df.loc[idx_map['b2_phase_mean'], 'B'] = format_val(m['b2_phase']['mean1']); results_ols_df.loc[idx_map['b2_phase_mean'], 'C'] = format_val(m['b2_phase']['mean2']); results_ols_df.loc[idx_map['b2_phase_mean'], 'D'] = format_val(m['b2_phase']['diff'])
        results_ols_df.loc[idx_map['b2_phase_se'], 'B'] = f"({format_val(m['b2_phase']['se1'])})"; results_ols_df.loc[idx_map['b2_phase_se'], 'C'] = f"({format_val(m['b2_phase']['se2'])})"; results_ols_df.loc[idx_map['b2_phase_se'], 'D'] = f"({format_val(m['b2_phase']['se_diff'])})"
        results_ols_df.loc[idx_map['b2_phase_n'], 'B'] = f"[{m['b2_phase']['n1']}]"; results_ols_df.loc[idx_map['b2_phase_n'], 'C'] = f"[{m['b2_phase']['n2']}]"
        results_ols_df.loc[idx_map['b2_cdf_mean'], 'B'] = format_val(m['b2_cdf']['mean1']); results_ols_df.loc[idx_map['b2_cdf_mean'], 'C'] = format_val(m['b2_cdf']['mean2']); results_ols_df.loc[idx_map['b2_cdf_mean'], 'D'] = format_val(m['b2_cdf']['diff'])
        results_ols_df.loc[idx_map['b2_cdf_se'], 'B'] = f"({format_val(m['b2_cdf']['se1'])})"; results_ols_df.loc[idx_map['b2_cdf_se'], 'C'] = f"({format_val(m['b2_cdf']['se2'])})"; results_ols_df.loc[idx_map['b2_cdf_se'], 'D'] = f"({format_val(m['b2_cdf']['se_diff'])})"
        results_ols_df.loc[idx_map['b2_cdf_n'], 'B'] = f"[{m['b2_cdf']['n1']}]"; results_ols_df.loc[idx_map['b2_cdf_n'], 'C'] = f"[{m['b2_cdf']['n2']}]"
        results_ols_df.loc[idx_map['b2_dd_coef'], 'D'] = format_val(m['b2_dd']['coef']); results_ols_df.loc[idx_map['b2_dd_se'], 'D'] = f"({format_val(m['b2_dd']['se'])})"
        results_ols_df.loc[idx_map['b_ddd_coef'], 'D'] = format_val(m['b_ddd']['coef']); results_ols_df.loc[idx_map['b_ddd_se'], 'D'] = f"({format_val(m['b_ddd']['se'])})"
        
        # Canada OLS
        c = results_ols['can']
        results_ols_df.loc[idx_map['a1_nafta_mean'], 'F'] = format_val(c['a1_nafta']['mean1']); results_ols_df.loc[idx_map['a1_nafta_mean'], 'G'] = format_val(c['a1_nafta']['mean2']); results_ols_df.loc[idx_map['a1_nafta_mean'], 'H'] = format_val(c['a1_nafta']['diff'])
        results_ols_df.loc[idx_map['a1_nafta_se'], 'F'] = f"({format_val(c['a1_nafta']['se1'])})"; results_ols_df.loc[idx_map['a1_nafta_se'], 'G'] = f"({format_val(c['a1_nafta']['se2'])})"; results_ols_df.loc[idx_map['a1_nafta_se'], 'H'] = f"({format_val(c['a1_nafta']['se_diff'])})"
        results_ols_df.loc[idx_map['a1_nafta_n'], 'F'] = f"[{c['a1_nafta']['n1']}]"; results_ols_df.loc[idx_map['a1_nafta_n'], 'G'] = f"[{c['a1_nafta']['n2']}]"
        results_ols_df.loc[idx_map['a1_row_mean'], 'F'] = format_val(c['a1_row']['mean1']); results_ols_df.loc[idx_map['a1_row_mean'], 'G'] = format_val(c['a1_row']['mean2']); results_ols_df.loc[idx_map['a1_row_mean'], 'H'] = format_val(c['a1_row']['diff'])
        results_ols_df.loc[idx_map['a1_row_se'], 'F'] = f"({format_val(c['a1_row']['se1'])})"; results_ols_df.loc[idx_map['a1_row_se'], 'G'] = f"({format_val(c['a1_row']['se2'])})"; results_ols_df.loc[idx_map['a1_row_se'], 'H'] = f"({format_val(c['a1_row']['se_diff'])})"
        results_ols_df.loc[idx_map['a1_row_n'], 'F'] = f"[{c['a1_row']['n1']}]"; results_ols_df.loc[idx_map['a1_row_n'], 'G'] = f"[{c['a1_row']['n2']}]"
        results_ols_df.loc[idx_map['a1_dd_coef'], 'H'] = format_val(c['a1_dd']['coef']); results_ols_df.loc[idx_map['a1_dd_se'], 'H'] = f"({format_val(c['a1_dd']['se'])})"
        results_ols_df.loc[idx_map['a2_nafta_mean'], 'F'] = format_val(c['a2_nafta']['mean1']); results_ols_df.loc[idx_map['a2_nafta_mean'], 'G'] = format_val(c['a2_nafta']['mean2']); results_ols_df.loc[idx_map['a2_nafta_mean'], 'H'] = format_val(c['a2_nafta']['diff'])
        results_ols_df.loc[idx_map['a2_nafta_se'], 'F'] = f"({format_val(c['a2_nafta']['se1'])})"; results_ols_df.loc[idx_map['a2_nafta_se'], 'G'] = f"({format_val(c['a2_nafta']['se2'])})"; results_ols_df.loc[idx_map['a2_nafta_se'], 'H'] = f"({format_val(c['a2_nafta']['se_diff'])})"
        results_ols_df.loc[idx_map['a2_nafta_n'], 'F'] = f"[{c['a2_nafta']['n1']}]"; results_ols_df.loc[idx_map['a2_nafta_n'], 'G'] = f"[{c['a2_nafta']['n2']}]"
        results_ols_df.loc[idx_map['a2_row_mean'], 'F'] = format_val(c['a2_row']['mean1']); results_ols_df.loc[idx_map['a2_row_mean'], 'G'] = format_val(c['a2_row']['mean2']); results_ols_df.loc[idx_map['a2_row_mean'], 'H'] = format_val(c['a2_row']['diff'])
        results_ols_df.loc[idx_map['a2_row_se'], 'F'] = f"({format_val(c['a2_row']['se1'])})"; results_ols_df.loc[idx_map['a2_row_se'], 'G'] = f"({format_val(c['a2_row']['se2'])})"; results_ols_df.loc[idx_map['a2_row_se'], 'H'] = f"({format_val(c['a2_row']['se_diff'])})"
        results_ols_df.loc[idx_map['a2_row_n'], 'F'] = f"[{c['a2_row']['n1']}]"; results_ols_df.loc[idx_map['a2_row_n'], 'G'] = f"[{c['a2_row']['n2']}]"
        results_ols_df.loc[idx_map['a2_dd_coef'], 'H'] = format_val(c['a2_dd']['coef']); results_ols_df.loc[idx_map['a2_dd_se'], 'H'] = f"({format_val(c['a2_dd']['se'])})"
        results_ols_df.loc[idx_map['a_ddd_coef'], 'H'] = format_val(c['a_ddd']['coef']); results_ols_df.loc[idx_map['a_ddd_se'], 'H'] = f"({format_val(c['a_ddd']['se'])})"
        results_ols_df.loc[idx_map['b1_phase_mean'], 'F'] = format_val(c['b1_phase']['mean1']); results_ols_df.loc[idx_map['b1_phase_mean'], 'G'] = format_val(c['b1_phase']['mean2']); results_ols_df.loc[idx_map['b1_phase_mean'], 'H'] = format_val(c['b1_phase']['diff'])
        results_ols_df.loc[idx_map['b1_phase_se'], 'F'] = f"({format_val(c['b1_phase']['se1'])})"; results_ols_df.loc[idx_map['b1_phase_se'], 'G'] = f"({format_val(c['b1_phase']['se2'])})"; results_ols_df.loc[idx_map['b1_phase_se'], 'H'] = f"({format_val(c['b1_phase']['se_diff'])})"
        results_ols_df.loc[idx_map['b1_phase_n'], 'F'] = f"[{c['b1_phase']['n1']}]"; results_ols_df.loc[idx_map['b1_phase_n'], 'G'] = f"[{c['b1_phase']['n2']}]"
        results_ols_df.loc[idx_map['b1_cdf_mean'], 'F'] = format_val(c['b1_cdf']['mean1']); results_ols_df.loc[idx_map['b1_cdf_mean'], 'G'] = format_val(c['b1_cdf']['mean2']); results_ols_df.loc[idx_map['b1_cdf_mean'], 'H'] = format_val(c['b1_cdf']['diff'])
        results_ols_df.loc[idx_map['b1_cdf_se'], 'F'] = f"({format_val(c['b1_cdf']['se1'])})"; results_ols_df.loc[idx_map['b1_cdf_se'], 'G'] = f"({format_val(c['b1_cdf']['se2'])})"; results_ols_df.loc[idx_map['b1_cdf_se'], 'H'] = f"({format_val(c['b1_cdf']['se_diff'])})"
        results_ols_df.loc[idx_map['b1_cdf_n'], 'F'] = f"[{c['b1_cdf']['n1']}]"; results_ols_df.loc[idx_map['b1_cdf_n'], 'G'] = f"[{c['b1_cdf']['n2']}]"
        results_ols_df.loc[idx_map['b1_dd_coef'], 'H'] = format_val(c['b1_dd']['coef']); results_ols_df.loc[idx_map['b1_dd_se'], 'H'] = f"({format_val(c['b1_dd']['se'])})"
        results_ols_df.loc[idx_map['b2_phase_mean'], 'F'] = format_val(c['b2_phase']['mean1']); results_ols_df.loc[idx_map['b2_phase_mean'], 'G'] = format_val(c['b2_phase']['mean2']); results_ols_df.loc[idx_map['b2_phase_mean'], 'H'] = format_val(c['b2_phase']['diff'])
        results_ols_df.loc[idx_map['b2_phase_se'], 'F'] = f"({format_val(c['b2_phase']['se1'])})"; results_ols_df.loc[idx_map['b2_phase_se'], 'G'] = f"({format_val(c['b2_phase']['se2'])})"; results_ols_df.loc[idx_map['b2_phase_se'], 'H'] = f"({format_val(c['b2_phase']['se_diff'])})"
        results_ols_df.loc[idx_map['b2_phase_n'], 'F'] = f"[{c['b2_phase']['n1']}]"; results_ols_df.loc[idx_map['b2_phase_n'], 'G'] = f"[{c['b2_phase']['n2']}]"
        results_ols_df.loc[idx_map['b2_cdf_mean'], 'F'] = format_val(c['b2_cdf']['mean1']); results_ols_df.loc[idx_map['b2_cdf_mean'], 'G'] = format_val(c['b2_cdf']['mean2']); results_ols_df.loc[idx_map['b2_cdf_mean'], 'H'] = format_val(c['b2_cdf']['diff'])
        results_ols_df.loc[idx_map['b2_cdf_se'], 'F'] = f"({format_val(c['b2_cdf']['se1'])})"; results_ols_df.loc[idx_map['b2_cdf_se'], 'G'] = f"({format_val(c['b2_cdf']['se2'])})"; results_ols_df.loc[idx_map['b2_cdf_se'], 'H'] = f"({format_val(c['b2_cdf']['se_diff'])})"
        results_ols_df.loc[idx_map['b2_cdf_n'], 'F'] = f"[{c['b2_cdf']['n1']}]"; results_ols_df.loc[idx_map['b2_cdf_n'], 'G'] = f"[{c['b2_cdf']['n2']}]"
        results_ols_df.loc[idx_map['b2_dd_coef'], 'H'] = format_val(c['b2_dd']['coef']); results_ols_df.loc[idx_map['b2_dd_se'], 'H'] = f"({format_val(c['b2_dd']['se'])})"
        results_ols_df.loc[idx_map['b_ddd_coef'], 'H'] = format_val(c['b_ddd']['coef']); results_ols_df.loc[idx_map['b_ddd_se'], 'H'] = f"({format_val(c['b_ddd']['se'])})"

        results_ols_df.to_excel(writer, sheet_name=f'Table2_OLS_Results{file_suffix}', index=False, header=False)
        print(f"OLS results for Table 2 saved to sheet Table2_OLS_Results{file_suffix}")
    else:
        print(f"Skipping OLS results table generation due to missing Mexico or Canada data.")

    # --- PPML Section ---
    print("\nProcessing PPML estimates for Table 2...")
    headers_ppml = pd.DataFrame({ # Structure from original
        'A': ["Table 2: Time-invariant DDD estimates of NAFTA (PPML on level values)", "", "Dependent Variable: Customs Value (Levels)", "", "Panel A: NAFTA vs ROW approach", "", "A1. Phase-out products", "Mexico", "Pre-NAFTA", "Post-NAFTA", "Growth", "NAFTA partner", "", "", "", "ROW", "", "", "", "DD (PPML)", "", "", "A2. CDF products", "Mexico", "Pre-NAFTA", "Post-NAFTA", "Growth", "NAFTA partner", "", "", "", "ROW", "", "", "", "DD (PPML)", "", "DDD (PPML)", "", "(continued on next page)", "", "Table 2 (continued)", "", "Panel B: Phase-out vs CDF-products approach", "", "B1. NAFTA partner", "Mexico", "Pre-NAFTA", "Post-NAFTA", "Growth", "Phase-out products", "", "", "", "CDF products", "", "", "", "DD (PPML)", "", "", "B2. ROW", "Mexico", "Pre-NAFTA", "Post-NAFTA", "Growth", "Phase-out products", "", "", "", "CDF products", "", "", "", "DD (PPML)", "", "DDD (PPML)", ""],
        'E': ["", "", "", "", "", "", "", "Canada", "Pre-NAFTA", "Post-NAFTA", "Growth", "NAFTA partner", "", "", "", "ROW", "", "", "", "DD (PPML)", "", "", "", "Canada", "Pre-NAFTA", "Post-NAFTA", "Growth", "NAFTA partner", "", "", "", "ROW", "", "", "", "DD (PPML)", "", "DDD (PPML)", "", "", "", "", "", "", "", "", "Canada", "Pre-NAFTA", "Post-NAFTA", "Growth", "Phase-out products", "", "", "", "CDF products", "", "", "", "DD (PPML)", "", "", "", "Canada", "Pre-NAFTA", "Post-NAFTA", "Growth", "Phase-out products", "", "", "", "CDF products", "", "", "", "DD (PPML)", "", "DDD (PPML)", ""]
    })
    headers_ppml.to_excel(writer, sheet_name=f'Table2_PPML_Input{file_suffix}', index=False, header=False)
    results_ppml_df = pd.DataFrame(index=range(max_rows_needed), columns=list('ABCDEFGHIJ'))
    results_ppml = {}

    # Process Mexico PPML Data (same as before)
    try:
        if 'mex' not in results_ols or not results_ols['mex']:
            mexico_file_path_pkl = os.path.join(dir_processed_data, f"data{file_suffix}_Mex.pkl"); mexico_file_path_dta = os.path.join(dir_processed_data, f"data{file_suffix}_Mex.dta")
            if os.path.exists(mexico_file_path_pkl): mex_df_ppml_load = pd.read_pickle(mexico_file_path_pkl)
            elif os.path.exists(mexico_file_path_dta): mex_df_ppml_load = pd.read_stata(mexico_file_path_dta)
            else: raise FileNotFoundError("Mexico data file not found for PPML.")
        else: mex_df_ppml_load = mex_df.copy()

        mex_df_ppml = mex_df_ppml_load.copy()
        mex_df_ppml['NAFTA'] = (mex_df_ppml['country'] == 'Mexico').astype(int); mex_df_ppml['ROW'] = (~mex_df_ppml['NAFTA'].astype(bool)).astype(int)
        mex_df_ppml['phase'] = (mex_df_ppml['catEffMex'] != 'D').astype(int); mex_df_ppml['CDF'] = (mex_df_ppml['catEffMex'] == 'D').astype(int)
        mex_df_ppml['pre'] = (mex_df_ppml['year'] < 1993).astype(int); mex_df_ppml['post'] = (mex_df_ppml['year'] >= 1993).astype(int)
        mex_df_ppml['Mex_cv'] = mex_df_ppml['Mex_cv'].astype(float)
        if (mex_df_ppml['Mex_cv'] < 0).any(): mex_df_ppml.loc[mex_df_ppml['Mex_cv'] < 0, 'Mex_cv'] = 0

        results_ppml['mex'] = {
            'a1_nafta': run_ttest(mex_df_ppml, 'Mex_cv', (mex_df_ppml['phase'] == 1) & (mex_df_ppml['NAFTA'] == 1), 'post'),
            'a1_row': run_ttest(mex_df_ppml, 'Mex_cv', (mex_df_ppml['phase'] == 1) & (mex_df_ppml['ROW'] == 1), 'post'),
            'a1_dd': run_ppml_dd_regression(mex_df_ppml, 'Mex_cv', 'NAFTA', 'post', (mex_df_ppml['phase'] == 1)),
            'a2_nafta': run_ttest(mex_df_ppml, 'Mex_cv', (mex_df_ppml['CDF'] == 1) & (mex_df_ppml['NAFTA'] == 1), 'post'),
            'a2_row': run_ttest(mex_df_ppml, 'Mex_cv', (mex_df_ppml['CDF'] == 1) & (mex_df_ppml['ROW'] == 1), 'post'),
            'a2_dd': run_ppml_dd_regression(mex_df_ppml, 'Mex_cv', 'NAFTA', 'post', (mex_df_ppml['CDF'] == 1)),
            'a_ddd': run_ppml_ddd_regression(mex_df_ppml, 'Mex_cv', 'NAFTA', 'post', 'phase'),
            'b1_phase': run_ttest(mex_df_ppml, 'Mex_cv', (mex_df_ppml['phase'] == 1) & (mex_df_ppml['NAFTA'] == 1), 'post'),
            'b1_cdf': run_ttest(mex_df_ppml, 'Mex_cv', (mex_df_ppml['CDF'] == 1) & (mex_df_ppml['NAFTA'] == 1), 'post'),
            'b1_dd': run_ppml_dd_regression(mex_df_ppml, 'Mex_cv', 'phase', 'post', (mex_df_ppml['NAFTA'] == 1)),
            'b2_phase': run_ttest(mex_df_ppml, 'Mex_cv', (mex_df_ppml['phase'] == 1) & (mex_df_ppml['ROW'] == 1), 'post'),
            'b2_cdf': run_ttest(mex_df_ppml, 'Mex_cv', (mex_df_ppml['CDF'] == 1) & (mex_df_ppml['ROW'] == 1), 'post'),
            'b2_dd': run_ppml_dd_regression(mex_df_ppml, 'Mex_cv', 'phase', 'post', (mex_df_ppml['ROW'] == 1)),
            'b_ddd': run_ppml_ddd_regression(mex_df_ppml, 'Mex_cv', 'NAFTA', 'post', 'phase')
        }
    except Exception as e: print(f"Error processing Mexico PPML data: {e}"); results_ppml['mex'] = {}

    # Process Canada PPML Data (same as before)
    try:
        if 'can' not in results_ols or not results_ols['can']:
            canada_file_path_pkl = os.path.join(dir_processed_data, f"data{file_suffix}_Can.pkl"); canada_file_path_dta = os.path.join(dir_processed_data, f"data{file_suffix}_Can.dta")
            if os.path.exists(canada_file_path_pkl): can_df_ppml_load = pd.read_pickle(canada_file_path_pkl)
            elif os.path.exists(canada_file_path_dta): can_df_ppml_load = pd.read_stata(canada_file_path_dta)
            else: raise FileNotFoundError("Canada data file not found for PPML.")
        else: can_df_ppml_load = can_df.copy()

        can_df_ppml = can_df_ppml_load.copy()
        can_df_ppml['NAFTA'] = (can_df_ppml['country'] == 'Canada').astype(int); can_df_ppml['ROW'] = (~can_df_ppml['NAFTA'].astype(bool)).astype(int)
        can_df_ppml['phase'] = (can_df_ppml['catEffCan'] != 'D').astype(int); can_df_ppml['CDF'] = (can_df_ppml['catEffCan'] == 'D').astype(int)
        can_df_ppml['pre'] = (can_df_ppml['year'] < 1993).astype(int); can_df_ppml['post'] = (can_df_ppml['year'] >= 1993).astype(int)
        can_df_ppml['Can_cv'] = can_df_ppml['Can_cv'].astype(float)
        if (can_df_ppml['Can_cv'] < 0).any(): can_df_ppml.loc[can_df_ppml['Can_cv'] < 0, 'Can_cv'] = 0
        
        results_ppml['can'] = {
            'a1_nafta': run_ttest(can_df_ppml, 'Can_cv', (can_df_ppml['phase'] == 1) & (can_df_ppml['NAFTA'] == 1), 'post'),
            'a1_row': run_ttest(can_df_ppml, 'Can_cv', (can_df_ppml['phase'] == 1) & (can_df_ppml['ROW'] == 1), 'post'),
            'a1_dd': run_ppml_dd_regression(can_df_ppml, 'Can_cv', 'NAFTA', 'post', (can_df_ppml['phase'] == 1)),
            'a2_nafta': run_ttest(can_df_ppml, 'Can_cv', (can_df_ppml['CDF'] == 1) & (can_df_ppml['NAFTA'] == 1), 'post'),
            'a2_row': run_ttest(can_df_ppml, 'Can_cv', (can_df_ppml['CDF'] == 1) & (can_df_ppml['ROW'] == 1), 'post'),
            'a2_dd': run_ppml_dd_regression(can_df_ppml, 'Can_cv', 'NAFTA', 'post', (can_df_ppml['CDF'] == 1)),
            'a_ddd': run_ppml_ddd_regression(can_df_ppml, 'Can_cv', 'NAFTA', 'post', 'phase'),
            'b1_phase': run_ttest(can_df_ppml, 'Can_cv', (can_df_ppml['phase'] == 1) & (can_df_ppml['NAFTA'] == 1), 'post'),
            'b1_cdf': run_ttest(can_df_ppml, 'Can_cv', (can_df_ppml['CDF'] == 1) & (can_df_ppml['NAFTA'] == 1), 'post'),
            'b1_dd': run_ppml_dd_regression(can_df_ppml, 'Can_cv', 'phase', 'post', (can_df_ppml['NAFTA'] == 1)),
            'b2_phase': run_ttest(can_df_ppml, 'Can_cv', (can_df_ppml['phase'] == 1) & (can_df_ppml['ROW'] == 1), 'post'),
            'b2_cdf': run_ttest(can_df_ppml, 'Can_cv', (can_df_ppml['CDF'] == 1) & (can_df_ppml['ROW'] == 1), 'post'),
            'b2_dd': run_ppml_dd_regression(can_df_ppml, 'Can_cv', 'phase', 'post', (can_df_ppml['ROW'] == 1)),
            'b_ddd': run_ppml_ddd_regression(can_df_ppml, 'Can_cv', 'NAFTA', 'post', 'phase')
        }
    except Exception as e: print(f"Error processing Canada PPML data: {e}"); results_ppml['can'] = {}

    if 'mex' in results_ppml and results_ppml['mex'] and 'can' in results_ppml and results_ppml['can']:
        def format_val_level(val, format_str="{:.1f}"):
            if pd.isna(val) or val == "": return ""
            try: return format_str.format(float(val))
            except (ValueError, TypeError): return str(val)
        def format_val_coef(val, format_str="{:.3f}"):
            if pd.isna(val) or val == "": return ""
            try: return format_str.format(float(val))
            except (ValueError, TypeError): return str(val)

        # Mexico PPML
        m = results_ppml['mex']
        results_ppml_df.loc[idx_map['a1_nafta_mean'], 'B'] = format_val_level(m['a1_nafta']['mean1']); results_ppml_df.loc[idx_map['a1_nafta_mean'], 'C'] = format_val_level(m['a1_nafta']['mean2']); results_ppml_df.loc[idx_map['a1_nafta_mean'], 'D'] = format_val_level(m['a1_nafta']['diff'])
        results_ppml_df.loc[idx_map['a1_nafta_se'], 'B'] = f"({format_val_level(m['a1_nafta']['se1'])})"; results_ppml_df.loc[idx_map['a1_nafta_se'], 'C'] = f"({format_val_level(m['a1_nafta']['se2'])})"; results_ppml_df.loc[idx_map['a1_nafta_se'], 'D'] = f"({format_val_level(m['a1_nafta']['se_diff'])})"
        results_ppml_df.loc[idx_map['a1_nafta_n'], 'B'] = f"[{m['a1_nafta']['n1']}]"; results_ppml_df.loc[idx_map['a1_nafta_n'], 'C'] = f"[{m['a1_nafta']['n2']}]"
        results_ppml_df.loc[idx_map['a1_row_mean'], 'B'] = format_val_level(m['a1_row']['mean1']); results_ppml_df.loc[idx_map['a1_row_mean'], 'C'] = format_val_level(m['a1_row']['mean2']); results_ppml_df.loc[idx_map['a1_row_mean'], 'D'] = format_val_level(m['a1_row']['diff'])
        results_ppml_df.loc[idx_map['a1_row_se'], 'B'] = f"({format_val_level(m['a1_row']['se1'])})"; results_ppml_df.loc[idx_map['a1_row_se'], 'C'] = f"({format_val_level(m['a1_row']['se2'])})"; results_ppml_df.loc[idx_map['a1_row_se'], 'D'] = f"({format_val_level(m['a1_row']['se_diff'])})"
        results_ppml_df.loc[idx_map['a1_row_n'], 'B'] = f"[{m['a1_row']['n1']}]"; results_ppml_df.loc[idx_map['a1_row_n'], 'C'] = f"[{m['a1_row']['n2']}]"
        results_ppml_df.loc[idx_map['a1_dd_coef'], 'D'] = format_val_coef(m['a1_dd']['coef']); results_ppml_df.loc[idx_map['a1_dd_se'], 'D'] = f"({format_val_coef(m['a1_dd']['se'])})"
        results_ppml_df.loc[idx_map['a2_nafta_mean'], 'B'] = format_val_level(m['a2_nafta']['mean1']); results_ppml_df.loc[idx_map['a2_nafta_mean'], 'C'] = format_val_level(m['a2_nafta']['mean2']); results_ppml_df.loc[idx_map['a2_nafta_mean'], 'D'] = format_val_level(m['a2_nafta']['diff'])
        results_ppml_df.loc[idx_map['a2_nafta_se'], 'B'] = f"({format_val_level(m['a2_nafta']['se1'])})"; results_ppml_df.loc[idx_map['a2_nafta_se'], 'C'] = f"({format_val_level(m['a2_nafta']['se2'])})"; results_ppml_df.loc[idx_map['a2_nafta_se'], 'D'] = f"({format_val_level(m['a2_nafta']['se_diff'])})"
        results_ppml_df.loc[idx_map['a2_nafta_n'], 'B'] = f"[{m['a2_nafta']['n1']}]"; results_ppml_df.loc[idx_map['a2_nafta_n'], 'C'] = f"[{m['a2_nafta']['n2']}]"
        results_ppml_df.loc[idx_map['a2_row_mean'], 'B'] = format_val_level(m['a2_row']['mean1']); results_ppml_df.loc[idx_map['a2_row_mean'], 'C'] = format_val_level(m['a2_row']['mean2']); results_ppml_df.loc[idx_map['a2_row_mean'], 'D'] = format_val_level(m['a2_row']['diff'])
        results_ppml_df.loc[idx_map['a2_row_se'], 'B'] = f"({format_val_level(m['a2_row']['se1'])})"; results_ppml_df.loc[idx_map['a2_row_se'], 'C'] = f"({format_val_level(m['a2_row']['se2'])})"; results_ppml_df.loc[idx_map['a2_row_se'], 'D'] = f"({format_val_level(m['a2_row']['se_diff'])})"
        results_ppml_df.loc[idx_map['a2_row_n'], 'B'] = f"[{m['a2_row']['n1']}]"; results_ppml_df.loc[idx_map['a2_row_n'], 'C'] = f"[{m['a2_row']['n2']}]"
        results_ppml_df.loc[idx_map['a2_dd_coef'], 'D'] = format_val_coef(m['a2_dd']['coef']); results_ppml_df.loc[idx_map['a2_dd_se'], 'D'] = f"({format_val_coef(m['a2_dd']['se'])})"
        results_ppml_df.loc[idx_map['a_ddd_coef'], 'D'] = format_val_coef(m['a_ddd']['coef']); results_ppml_df.loc[idx_map['a_ddd_se'], 'D'] = f"({format_val_coef(m['a_ddd']['se'])})"
        results_ppml_df.loc[idx_map['b1_phase_mean'], 'B'] = format_val_level(m['b1_phase']['mean1']); results_ppml_df.loc[idx_map['b1_phase_mean'], 'C'] = format_val_level(m['b1_phase']['mean2']); results_ppml_df.loc[idx_map['b1_phase_mean'], 'D'] = format_val_level(m['b1_phase']['diff'])
        results_ppml_df.loc[idx_map['b1_phase_se'], 'B'] = f"({format_val_level(m['b1_phase']['se1'])})"; results_ppml_df.loc[idx_map['b1_phase_se'], 'C'] = f"({format_val_level(m['b1_phase']['se2'])})"; results_ppml_df.loc[idx_map['b1_phase_se'], 'D'] = f"({format_val_level(m['b1_phase']['se_diff'])})"
        results_ppml_df.loc[idx_map['b1_phase_n'], 'B'] = f"[{m['b1_phase']['n1']}]"; results_ppml_df.loc[idx_map['b1_phase_n'], 'C'] = f"[{m['b1_phase']['n2']}]"
        results_ppml_df.loc[idx_map['b1_cdf_mean'], 'B'] = format_val_level(m['b1_cdf']['mean1']); results_ppml_df.loc[idx_map['b1_cdf_mean'], 'C'] = format_val_level(m['b1_cdf']['mean2']); results_ppml_df.loc[idx_map['b1_cdf_mean'], 'D'] = format_val_level(m['b1_cdf']['diff'])
        results_ppml_df.loc[idx_map['b1_cdf_se'], 'B'] = f"({format_val_level(m['b1_cdf']['se1'])})"; results_ppml_df.loc[idx_map['b1_cdf_se'], 'C'] = f"({format_val_level(m['b1_cdf']['se2'])})"; results_ppml_df.loc[idx_map['b1_cdf_se'], 'D'] = f"({format_val_level(m['b1_cdf']['se_diff'])})"
        results_ppml_df.loc[idx_map['b1_cdf_n'], 'B'] = f"[{m['b1_cdf']['n1']}]"; results_ppml_df.loc[idx_map['b1_cdf_n'], 'C'] = f"[{m['b1_cdf']['n2']}]"
        results_ppml_df.loc[idx_map['b1_dd_coef'], 'D'] = format_val_coef(m['b1_dd']['coef']); results_ppml_df.loc[idx_map['b1_dd_se'], 'D'] = f"({format_val_coef(m['b1_dd']['se'])})"
        results_ppml_df.loc[idx_map['b2_phase_mean'], 'B'] = format_val_level(m['b2_phase']['mean1']); results_ppml_df.loc[idx_map['b2_phase_mean'], 'C'] = format_val_level(m['b2_phase']['mean2']); results_ppml_df.loc[idx_map['b2_phase_mean'], 'D'] = format_val_level(m['b2_phase']['diff'])
        results_ppml_df.loc[idx_map['b2_phase_se'], 'B'] = f"({format_val_level(m['b2_phase']['se1'])})"; results_ppml_df.loc[idx_map['b2_phase_se'], 'C'] = f"({format_val_level(m['b2_phase']['se2'])})"; results_ppml_df.loc[idx_map['b2_phase_se'], 'D'] = f"({format_val_level(m['b2_phase']['se_diff'])})"
        results_ppml_df.loc[idx_map['b2_phase_n'], 'B'] = f"[{m['b2_phase']['n1']}]"; results_ppml_df.loc[idx_map['b2_phase_n'], 'C'] = f"[{m['b2_phase']['n2']}]"
        results_ppml_df.loc[idx_map['b2_cdf_mean'], 'B'] = format_val_level(m['b2_cdf']['mean1']); results_ppml_df.loc[idx_map['b2_cdf_mean'], 'C'] = format_val_level(m['b2_cdf']['mean2']); results_ppml_df.loc[idx_map['b2_cdf_mean'], 'D'] = format_val_level(m['b2_cdf']['diff'])
        results_ppml_df.loc[idx_map['b2_cdf_se'], 'B'] = f"({format_val_level(m['b2_cdf']['se1'])})"; results_ppml_df.loc[idx_map['b2_cdf_se'], 'C'] = f"({format_val_level(m['b2_cdf']['se2'])})"; results_ppml_df.loc[idx_map['b2_cdf_se'], 'D'] = f"({format_val_level(m['b2_cdf']['se_diff'])})"
        results_ppml_df.loc[idx_map['b2_cdf_n'], 'B'] = f"[{m['b2_cdf']['n1']}]"; results_ppml_df.loc[idx_map['b2_cdf_n'], 'C'] = f"[{m['b2_cdf']['n2']}]"
        results_ppml_df.loc[idx_map['b2_dd_coef'], 'D'] = format_val_coef(m['b2_dd']['coef']); results_ppml_df.loc[idx_map['b2_dd_se'], 'D'] = f"({format_val_coef(m['b2_dd']['se'])})"
        results_ppml_df.loc[idx_map['b_ddd_coef'], 'D'] = format_val_coef(m['b_ddd']['coef']); results_ppml_df.loc[idx_map['b_ddd_se'], 'D'] = f"({format_val_coef(m['b_ddd']['se'])})"

        # Canada PPML
        c = results_ppml['can']
        results_ppml_df.loc[idx_map['a1_nafta_mean'], 'F'] = format_val_level(c['a1_nafta']['mean1']); results_ppml_df.loc[idx_map['a1_nafta_mean'], 'G'] = format_val_level(c['a1_nafta']['mean2']); results_ppml_df.loc[idx_map['a1_nafta_mean'], 'H'] = format_val_level(c['a1_nafta']['diff'])
        results_ppml_df.loc[idx_map['a1_nafta_se'], 'F'] = f"({format_val_level(c['a1_nafta']['se1'])})"; results_ppml_df.loc[idx_map['a1_nafta_se'], 'G'] = f"({format_val_level(c['a1_nafta']['se2'])})"; results_ppml_df.loc[idx_map['a1_nafta_se'], 'H'] = f"({format_val_level(c['a1_nafta']['se_diff'])})"
        results_ppml_df.loc[idx_map['a1_nafta_n'], 'F'] = f"[{c['a1_nafta']['n1']}]"; results_ppml_df.loc[idx_map['a1_nafta_n'], 'G'] = f"[{c['a1_nafta']['n2']}]"
        results_ppml_df.loc[idx_map['a1_row_mean'], 'F'] = format_val_level(c['a1_row']['mean1']); results_ppml_df.loc[idx_map['a1_row_mean'], 'G'] = format_val_level(c['a1_row']['mean2']); results_ppml_df.loc[idx_map['a1_row_mean'], 'H'] = format_val_level(c['a1_row']['diff'])
        results_ppml_df.loc[idx_map['a1_row_se'], 'F'] = f"({format_val_level(c['a1_row']['se1'])})"; results_ppml_df.loc[idx_map['a1_row_se'], 'G'] = f"({format_val_level(c['a1_row']['se2'])})"; results_ppml_df.loc[idx_map['a1_row_se'], 'H'] = f"({format_val_level(c['a1_row']['se_diff'])})"
        results_ppml_df.loc[idx_map['a1_row_n'], 'F'] = f"[{c['a1_row']['n1']}]"; results_ppml_df.loc[idx_map['a1_row_n'], 'G'] = f"[{c['a1_row']['n2']}]"
        results_ppml_df.loc[idx_map['a1_dd_coef'], 'H'] = format_val_coef(c['a1_dd']['coef']); results_ppml_df.loc[idx_map['a1_dd_se'], 'H'] = f"({format_val_coef(c['a1_dd']['se'])})"
        results_ppml_df.loc[idx_map['a2_nafta_mean'], 'F'] = format_val_level(c['a2_nafta']['mean1']); results_ppml_df.loc[idx_map['a2_nafta_mean'], 'G'] = format_val_level(c['a2_nafta']['mean2']); results_ppml_df.loc[idx_map['a2_nafta_mean'], 'H'] = format_val_level(c['a2_nafta']['diff'])
        results_ppml_df.loc[idx_map['a2_nafta_se'], 'F'] = f"({format_val_level(c['a2_nafta']['se1'])})"; results_ppml_df.loc[idx_map['a2_nafta_se'], 'G'] = f"({format_val_level(c['a2_nafta']['se2'])})"; results_ppml_df.loc[idx_map['a2_nafta_se'], 'H'] = f"({format_val_level(c['a2_nafta']['se_diff'])})"
        results_ppml_df.loc[idx_map['a2_nafta_n'], 'F'] = f"[{c['a2_nafta']['n1']}]"; results_ppml_df.loc[idx_map['a2_nafta_n'], 'G'] = f"[{c['a2_nafta']['n2']}]"
        results_ppml_df.loc[idx_map['a2_row_mean'], 'F'] = format_val_level(c['a2_row']['mean1']); results_ppml_df.loc[idx_map['a2_row_mean'], 'G'] = format_val_level(c['a2_row']['mean2']); results_ppml_df.loc[idx_map['a2_row_mean'], 'H'] = format_val_level(c['a2_row']['diff'])
        results_ppml_df.loc[idx_map['a2_row_se'], 'F'] = f"({format_val_level(c['a2_row']['se1'])})"; results_ppml_df.loc[idx_map['a2_row_se'], 'G'] = f"({format_val_level(c['a2_row']['se2'])})"; results_ppml_df.loc[idx_map['a2_row_se'], 'H'] = f"({format_val_level(c['a2_row']['se_diff'])})"
        results_ppml_df.loc[idx_map['a2_row_n'], 'F'] = f"[{c['a2_row']['n1']}]"; results_ppml_df.loc[idx_map['a2_row_n'], 'G'] = f"[{c['a2_row']['n2']}]"
        results_ppml_df.loc[idx_map['a2_dd_coef'], 'H'] = format_val_coef(c['a2_dd']['coef']); results_ppml_df.loc[idx_map['a2_dd_se'], 'H'] = f"({format_val_coef(c['a2_dd']['se'])})"
        results_ppml_df.loc[idx_map['a_ddd_coef'], 'H'] = format_val_coef(c['a_ddd']['coef']); results_ppml_df.loc[idx_map['a_ddd_se'], 'H'] = f"({format_val_coef(c['a_ddd']['se'])})"
        results_ppml_df.loc[idx_map['b1_phase_mean'], 'F'] = format_val_level(c['b1_phase']['mean1']); results_ppml_df.loc[idx_map['b1_phase_mean'], 'G'] = format_val_level(c['b1_phase']['mean2']); results_ppml_df.loc[idx_map['b1_phase_mean'], 'H'] = format_val_level(c['b1_phase']['diff'])
        results_ppml_df.loc[idx_map['b1_phase_se'], 'F'] = f"({format_val_level(c['b1_phase']['se1'])})"; results_ppml_df.loc[idx_map['b1_phase_se'], 'G'] = f"({format_val_level(c['b1_phase']['se2'])})"; results_ppml_df.loc[idx_map['b1_phase_se'], 'H'] = f"({format_val_level(c['b1_phase']['se_diff'])})"
        results_ppml_df.loc[idx_map['b1_phase_n'], 'F'] = f"[{c['b1_phase']['n1']}]"; results_ppml_df.loc[idx_map['b1_phase_n'], 'G'] = f"[{c['b1_phase']['n2']}]"
        results_ppml_df.loc[idx_map['b1_cdf_mean'], 'F'] = format_val_level(c['b1_cdf']['mean1']); results_ppml_df.loc[idx_map['b1_cdf_mean'], 'G'] = format_val_level(c['b1_cdf']['mean2']); results_ppml_df.loc[idx_map['b1_cdf_mean'], 'H'] = format_val_level(c['b1_cdf']['diff'])
        results_ppml_df.loc[idx_map['b1_cdf_se'], 'F'] = f"({format_val_level(c['b1_cdf']['se1'])})"; results_ppml_df.loc[idx_map['b1_cdf_se'], 'G'] = f"({format_val_level(c['b1_cdf']['se2'])})"; results_ppml_df.loc[idx_map['b1_cdf_se'], 'H'] = f"({format_val_level(c['b1_cdf']['se_diff'])})"
        results_ppml_df.loc[idx_map['b1_cdf_n'], 'F'] = f"[{c['b1_cdf']['n1']}]"; results_ppml_df.loc[idx_map['b1_cdf_n'], 'G'] = f"[{c['b1_cdf']['n2']}]"
        results_ppml_df.loc[idx_map['b1_dd_coef'], 'H'] = format_val_coef(c['b1_dd']['coef']); results_ppml_df.loc[idx_map['b1_dd_se'], 'H'] = f"({format_val_coef(c['b1_dd']['se'])})"
        results_ppml_df.loc[idx_map['b2_phase_mean'], 'F'] = format_val_level(c['b2_phase']['mean1']); results_ppml_df.loc[idx_map['b2_phase_mean'], 'G'] = format_val_level(c['b2_phase']['mean2']); results_ppml_df.loc[idx_map['b2_phase_mean'], 'H'] = format_val_level(c['b2_phase']['diff'])
        results_ppml_df.loc[idx_map['b2_phase_se'], 'F'] = f"({format_val_level(c['b2_phase']['se1'])})"; results_ppml_df.loc[idx_map['b2_phase_se'], 'G'] = f"({format_val_level(c['b2_phase']['se2'])})"; results_ppml_df.loc[idx_map['b2_phase_se'], 'H'] = f"({format_val_level(c['b2_phase']['se_diff'])})"
        results_ppml_df.loc[idx_map['b2_phase_n'], 'F'] = f"[{c['b2_phase']['n1']}]"; results_ppml_df.loc[idx_map['b2_phase_n'], 'G'] = f"[{c['b2_phase']['n2']}]"
        results_ppml_df.loc[idx_map['b2_cdf_mean'], 'F'] = format_val_level(c['b2_cdf']['mean1']); results_ppml_df.loc[idx_map['b2_cdf_mean'], 'G'] = format_val_level(c['b2_cdf']['mean2']); results_ppml_df.loc[idx_map['b2_cdf_mean'], 'H'] = format_val_level(c['b2_cdf']['diff'])
        results_ppml_df.loc[idx_map['b2_cdf_se'], 'F'] = f"({format_val_level(c['b2_cdf']['se1'])})"; results_ppml_df.loc[idx_map['b2_cdf_se'], 'G'] = f"({format_val_level(c['b2_cdf']['se2'])})"; results_ppml_df.loc[idx_map['b2_cdf_se'], 'H'] = f"({format_val_level(c['b2_cdf']['se_diff'])})"
        results_ppml_df.loc[idx_map['b2_cdf_n'], 'F'] = f"[{c['b2_cdf']['n1']}]"; results_ppml_df.loc[idx_map['b2_cdf_n'], 'G'] = f"[{c['b2_cdf']['n2']}]"
        results_ppml_df.loc[idx_map['b2_dd_coef'], 'H'] = format_val_coef(c['b2_dd']['coef']); results_ppml_df.loc[idx_map['b2_dd_se'], 'H'] = f"({format_val_coef(c['b2_dd']['se'])})"
        results_ppml_df.loc[idx_map['b_ddd_coef'], 'H'] = format_val_coef(c['b_ddd']['coef']); results_ppml_df.loc[idx_map['b_ddd_se'], 'H'] = f"({format_val_coef(c['b_ddd']['se'])})"

        results_ppml_df.to_excel(writer, sheet_name=f'Table2_PPML_Results{file_suffix}', index=False, header=False)
        print(f"PPML results for Table 2 saved to sheet Table2_PPML_Results{file_suffix}")
    else:
        print(f"Skipping PPML results table generation due to missing Mexico or Canada data for PPML.")


    # --- Log-Linear Corrected Section ---
    print("\nProcessing Log-Linear Corrected estimates for Table 2...")
    headers_corrected = pd.DataFrame({ # Based on PPML headers, change "PPML" to "Corrected"
        'A': ["Table 2: Time-invariant DDD estimates of NAFTA (Log-Linear Corrected on level values)", "", "Dependent Variable: Customs Value (Levels)", "", "Panel A: NAFTA vs ROW approach", "", "A1. Phase-out products", "Mexico", "Pre-NAFTA", "Post-NAFTA", "Growth", "NAFTA partner", "", "", "", "ROW", "", "", "", "DD (Corrected)", "", "", "A2. CDF products", "Mexico", "Pre-NAFTA", "Post-NAFTA", "Growth", "NAFTA partner", "", "", "", "ROW", "", "", "", "DD (Corrected)", "", "DDD (Corrected)", "", "(continued on next page)", "", "Table 2 (continued)", "", "Panel B: Phase-out vs CDF-products approach", "", "B1. NAFTA partner", "Mexico", "Pre-NAFTA", "Post-NAFTA", "Growth", "Phase-out products", "", "", "", "CDF products", "", "", "", "DD (Corrected)", "", "", "B2. ROW", "Mexico", "Pre-NAFTA", "Post-NAFTA", "Growth", "Phase-out products", "", "", "", "CDF products", "", "", "", "DD (Corrected)", "", "DDD (Corrected)", ""],
        'E': ["", "", "", "", "", "", "", "Canada", "Pre-NAFTA", "Post-NAFTA", "Growth", "NAFTA partner", "", "", "", "ROW", "", "", "", "DD (Corrected)", "", "", "", "Canada", "Pre-NAFTA", "Post-NAFTA", "Growth", "NAFTA partner", "", "", "", "ROW", "", "", "", "DD (Corrected)", "", "DDD (Corrected)", "", "", "", "", "", "", "", "", "Canada", "Pre-NAFTA", "Post-NAFTA", "Growth", "Phase-out products", "", "", "", "CDF products", "", "", "", "DD (Corrected)", "", "", "", "Canada", "Pre-NAFTA", "Post-NAFTA", "Growth", "Phase-out products", "", "", "", "CDF products", "", "", "", "DD (Corrected)", "", "DDD (Corrected)", ""]
    })
    headers_corrected.to_excel(writer, sheet_name=f'Table2_Corrected_Input{file_suffix}', index=False, header=False)
    results_corrected_df = pd.DataFrame(index=range(max_rows_needed), columns=list('ABCDEFGHIJ'))
    results_corrected = {}

    if not loglinearcorrection_available:
        print("Skipping Log-Linear Corrected results: loglinearcorrection package not found.")
    else:
        # Process Mexico Corrected Data
        try:
            # Re-use mex_df_ppml if available, else load. This df has level 'Mex_cv'
            if 'mex' not in results_ppml or not results_ppml['mex'] or mex_df_ppml.empty :
                mexico_file_path_pkl = os.path.join(dir_processed_data, f"data{file_suffix}_Mex.pkl"); mexico_file_path_dta = os.path.join(dir_processed_data, f"data{file_suffix}_Mex.dta")
                if os.path.exists(mexico_file_path_pkl): mex_df_corrected_load = pd.read_pickle(mexico_file_path_pkl)
                elif os.path.exists(mexico_file_path_dta): mex_df_corrected_load = pd.read_stata(mexico_file_path_dta)
                else: raise FileNotFoundError("Mexico data file not found for Corrected Estimator.")
            else: mex_df_corrected_load = mex_df_ppml.copy()

            mex_df_corrected = mex_df_corrected_load.copy()
            mex_df_corrected['NAFTA'] = (mex_df_corrected['country'] == 'Mexico').astype(int); mex_df_corrected['ROW'] = (~mex_df_corrected['NAFTA'].astype(bool)).astype(int)
            mex_df_corrected['phase'] = (mex_df_corrected['catEffMex'] != 'D').astype(int); mex_df_corrected['CDF'] = (mex_df_corrected['catEffMex'] == 'D').astype(int)
            mex_df_corrected['pre'] = (mex_df_corrected['year'] < 1993).astype(int); mex_df_corrected['post'] = (mex_df_corrected['year'] >= 1993).astype(int)
            mex_df_corrected['Mex_cv'] = mex_df_corrected['Mex_cv'].astype(float)
            if (mex_df_corrected['Mex_cv'] < 0).any(): mex_df_corrected.loc[mex_df_corrected['Mex_cv'] < 0, 'Mex_cv'] = 0

            results_corrected['mex'] = {
                'a1_nafta': results_ppml.get('mex', {}).get('a1_nafta', run_ttest(mex_df_corrected, 'Mex_cv', (mex_df_corrected['phase'] == 1) & (mex_df_corrected['NAFTA'] == 1), 'post')),
                'a1_row': results_ppml.get('mex', {}).get('a1_row', run_ttest(mex_df_corrected, 'Mex_cv', (mex_df_corrected['phase'] == 1) & (mex_df_corrected['ROW'] == 1), 'post')),
                'a1_dd': run_corrected_dd_regression(mex_df_corrected, 'Mex_cv', 'NAFTA', 'post', (mex_df_corrected['phase'] == 1)),
                'a2_nafta': results_ppml.get('mex', {}).get('a2_nafta', run_ttest(mex_df_corrected, 'Mex_cv', (mex_df_corrected['CDF'] == 1) & (mex_df_corrected['NAFTA'] == 1), 'post')),
                'a2_row': results_ppml.get('mex', {}).get('a2_row', run_ttest(mex_df_corrected, 'Mex_cv', (mex_df_corrected['CDF'] == 1) & (mex_df_corrected['ROW'] == 1), 'post')),
                'a2_dd': run_corrected_dd_regression(mex_df_corrected, 'Mex_cv', 'NAFTA', 'post', (mex_df_corrected['CDF'] == 1)),
                'a_ddd': run_corrected_ddd_regression(mex_df_corrected, 'Mex_cv', 'NAFTA', 'post', 'phase'),
                'b1_phase': results_ppml.get('mex', {}).get('b1_phase', run_ttest(mex_df_corrected, 'Mex_cv', (mex_df_corrected['phase'] == 1) & (mex_df_corrected['NAFTA'] == 1), 'post')),
                'b1_cdf': results_ppml.get('mex', {}).get('b1_cdf', run_ttest(mex_df_corrected, 'Mex_cv', (mex_df_corrected['CDF'] == 1) & (mex_df_corrected['NAFTA'] == 1), 'post')),
                'b1_dd': run_corrected_dd_regression(mex_df_corrected, 'Mex_cv', 'phase', 'post', (mex_df_corrected['NAFTA'] == 1)),
                'b2_phase': results_ppml.get('mex', {}).get('b2_phase', run_ttest(mex_df_corrected, 'Mex_cv', (mex_df_corrected['phase'] == 1) & (mex_df_corrected['ROW'] == 1), 'post')),
                'b2_cdf': results_ppml.get('mex', {}).get('b2_cdf', run_ttest(mex_df_corrected, 'Mex_cv', (mex_df_corrected['CDF'] == 1) & (mex_df_corrected['ROW'] == 1), 'post')),
                'b2_dd': run_corrected_dd_regression(mex_df_corrected, 'Mex_cv', 'phase', 'post', (mex_df_corrected['ROW'] == 1)),
                'b_ddd': run_corrected_ddd_regression(mex_df_corrected, 'Mex_cv', 'NAFTA', 'post', 'phase')
            }
        except Exception as e: print(f"Error processing Mexico Corrected Estimator data: {e}"); results_corrected['mex'] = {}

        # Process Canada Corrected Data
        try:
            if 'can' not in results_ppml or not results_ppml['can'] or can_df_ppml.empty:
                canada_file_path_pkl = os.path.join(dir_processed_data, f"data{file_suffix}_Can.pkl"); canada_file_path_dta = os.path.join(dir_processed_data, f"data{file_suffix}_Can.dta")
                if os.path.exists(canada_file_path_pkl): can_df_corrected_load = pd.read_pickle(canada_file_path_pkl)
                elif os.path.exists(canada_file_path_dta): can_df_corrected_load = pd.read_stata(canada_file_path_dta)
                else: raise FileNotFoundError("Canada data file not found for Corrected Estimator.")
            else: can_df_corrected_load = can_df_ppml.copy()

            can_df_corrected = can_df_corrected_load.copy()
            can_df_corrected['NAFTA'] = (can_df_corrected['country'] == 'Canada').astype(int); can_df_corrected['ROW'] = (~can_df_corrected['NAFTA'].astype(bool)).astype(int)
            can_df_corrected['phase'] = (can_df_corrected['catEffCan'] != 'D').astype(int); can_df_corrected['CDF'] = (can_df_corrected['catEffCan'] == 'D').astype(int)
            can_df_corrected['pre'] = (can_df_corrected['year'] < 1993).astype(int); can_df_corrected['post'] = (can_df_corrected['year'] >= 1993).astype(int)
            can_df_corrected['Can_cv'] = can_df_corrected['Can_cv'].astype(float)
            if (can_df_corrected['Can_cv'] < 0).any(): can_df_corrected.loc[can_df_corrected['Can_cv'] < 0, 'Can_cv'] = 0
            
            results_corrected['can'] = {
                'a1_nafta': results_ppml.get('can', {}).get('a1_nafta', run_ttest(can_df_corrected, 'Can_cv', (can_df_corrected['phase'] == 1) & (can_df_corrected['NAFTA'] == 1), 'post')),
                'a1_row': results_ppml.get('can', {}).get('a1_row', run_ttest(can_df_corrected, 'Can_cv', (can_df_corrected['phase'] == 1) & (can_df_corrected['ROW'] == 1), 'post')),
                'a1_dd': run_corrected_dd_regression(can_df_corrected, 'Can_cv', 'NAFTA', 'post', (can_df_corrected['phase'] == 1)),
                'a2_nafta': results_ppml.get('can', {}).get('a2_nafta', run_ttest(can_df_corrected, 'Can_cv', (can_df_corrected['CDF'] == 1) & (can_df_corrected['NAFTA'] == 1), 'post')),
                'a2_row': results_ppml.get('can', {}).get('a2_row', run_ttest(can_df_corrected, 'Can_cv', (can_df_corrected['CDF'] == 1) & (can_df_corrected['ROW'] == 1), 'post')),
                'a2_dd': run_corrected_dd_regression(can_df_corrected, 'Can_cv', 'NAFTA', 'post', (can_df_corrected['CDF'] == 1)),
                'a_ddd': run_corrected_ddd_regression(can_df_corrected, 'Can_cv', 'NAFTA', 'post', 'phase'),
                'b1_phase': results_ppml.get('can', {}).get('b1_phase', run_ttest(can_df_corrected, 'Can_cv', (can_df_corrected['phase'] == 1) & (can_df_corrected['NAFTA'] == 1), 'post')),
                'b1_cdf': results_ppml.get('can', {}).get('b1_cdf', run_ttest(can_df_corrected, 'Can_cv', (can_df_corrected['CDF'] == 1) & (can_df_corrected['NAFTA'] == 1), 'post')),
                'b1_dd': run_corrected_dd_regression(can_df_corrected, 'Can_cv', 'phase', 'post', (can_df_corrected['NAFTA'] == 1)),
                'b2_phase': results_ppml.get('can', {}).get('b2_phase', run_ttest(can_df_corrected, 'Can_cv', (can_df_corrected['phase'] == 1) & (can_df_corrected['ROW'] == 1), 'post')),
                'b2_cdf': results_ppml.get('can', {}).get('b2_cdf', run_ttest(can_df_corrected, 'Can_cv', (can_df_corrected['CDF'] == 1) & (can_df_corrected['ROW'] == 1), 'post')),
                'b2_dd': run_corrected_dd_regression(can_df_corrected, 'Can_cv', 'phase', 'post', (can_df_corrected['ROW'] == 1)),
                'b_ddd': run_corrected_ddd_regression(can_df_corrected, 'Can_cv', 'NAFTA', 'post', 'phase')
            }
        except Exception as e: print(f"Error processing Canada Corrected Estimator data: {e}"); results_corrected['can'] = {}

        if 'mex' in results_corrected and results_corrected['mex'] and 'can' in results_corrected and results_corrected['can']:
            # Mexico Corrected
            m = results_corrected['mex']
            results_corrected_df.loc[idx_map['a1_nafta_mean'], 'B'] = format_val_level(m['a1_nafta']['mean1']); results_corrected_df.loc[idx_map['a1_nafta_mean'], 'C'] = format_val_level(m['a1_nafta']['mean2']); results_corrected_df.loc[idx_map['a1_nafta_mean'], 'D'] = format_val_level(m['a1_nafta']['diff'])
            results_corrected_df.loc[idx_map['a1_nafta_se'], 'B'] = f"({format_val_level(m['a1_nafta']['se1'])})"; results_corrected_df.loc[idx_map['a1_nafta_se'], 'C'] = f"({format_val_level(m['a1_nafta']['se2'])})"; results_corrected_df.loc[idx_map['a1_nafta_se'], 'D'] = f"({format_val_level(m['a1_nafta']['se_diff'])})"
            results_corrected_df.loc[idx_map['a1_nafta_n'], 'B'] = f"[{m['a1_nafta']['n1']}]"; results_corrected_df.loc[idx_map['a1_nafta_n'], 'C'] = f"[{m['a1_nafta']['n2']}]"
            results_corrected_df.loc[idx_map['a1_row_mean'], 'B'] = format_val_level(m['a1_row']['mean1']); results_corrected_df.loc[idx_map['a1_row_mean'], 'C'] = format_val_level(m['a1_row']['mean2']); results_corrected_df.loc[idx_map['a1_row_mean'], 'D'] = format_val_level(m['a1_row']['diff'])
            results_corrected_df.loc[idx_map['a1_row_se'], 'B'] = f"({format_val_level(m['a1_row']['se1'])})"; results_corrected_df.loc[idx_map['a1_row_se'], 'C'] = f"({format_val_level(m['a1_row']['se2'])})"; results_corrected_df.loc[idx_map['a1_row_se'], 'D'] = f"({format_val_level(m['a1_row']['se_diff'])})"
            results_corrected_df.loc[idx_map['a1_row_n'], 'B'] = f"[{m['a1_row']['n1']}]"; results_corrected_df.loc[idx_map['a1_row_n'], 'C'] = f"[{m['a1_row']['n2']}]"
            results_corrected_df.loc[idx_map['a1_dd_coef'], 'D'] = format_val_coef(m['a1_dd']['coef']); results_corrected_df.loc[idx_map['a1_dd_se'], 'D'] = "" # No SE
            results_corrected_df.loc[idx_map['a2_nafta_mean'], 'B'] = format_val_level(m['a2_nafta']['mean1']); results_corrected_df.loc[idx_map['a2_nafta_mean'], 'C'] = format_val_level(m['a2_nafta']['mean2']); results_corrected_df.loc[idx_map['a2_nafta_mean'], 'D'] = format_val_level(m['a2_nafta']['diff'])
            results_corrected_df.loc[idx_map['a2_nafta_se'], 'B'] = f"({format_val_level(m['a2_nafta']['se1'])})"; results_corrected_df.loc[idx_map['a2_nafta_se'], 'C'] = f"({format_val_level(m['a2_nafta']['se2'])})"; results_corrected_df.loc[idx_map['a2_nafta_se'], 'D'] = f"({format_val_level(m['a2_nafta']['se_diff'])})"
            results_corrected_df.loc[idx_map['a2_nafta_n'], 'B'] = f"[{m['a2_nafta']['n1']}]"; results_corrected_df.loc[idx_map['a2_nafta_n'], 'C'] = f"[{m['a2_nafta']['n2']}]"
            results_corrected_df.loc[idx_map['a2_row_mean'], 'B'] = format_val_level(m['a2_row']['mean1']); results_corrected_df.loc[idx_map['a2_row_mean'], 'C'] = format_val_level(m['a2_row']['mean2']); results_corrected_df.loc[idx_map['a2_row_mean'], 'D'] = format_val_level(m['a2_row']['diff'])
            results_corrected_df.loc[idx_map['a2_row_se'], 'B'] = f"({format_val_level(m['a2_row']['se1'])})"; results_corrected_df.loc[idx_map['a2_row_se'], 'C'] = f"({format_val_level(m['a2_row']['se2'])})"; results_corrected_df.loc[idx_map['a2_row_se'], 'D'] = f"({format_val_level(m['a2_row']['se_diff'])})"
            results_corrected_df.loc[idx_map['a2_row_n'], 'B'] = f"[{m['a2_row']['n1']}]"; results_corrected_df.loc[idx_map['a2_row_n'], 'C'] = f"[{m['a2_row']['n2']}]"
            results_corrected_df.loc[idx_map['a2_dd_coef'], 'D'] = format_val_coef(m['a2_dd']['coef']); results_corrected_df.loc[idx_map['a2_dd_se'], 'D'] = "" # No SE
            results_corrected_df.loc[idx_map['a_ddd_coef'], 'D'] = format_val_coef(m['a_ddd']['coef']); results_corrected_df.loc[idx_map['a_ddd_se'], 'D'] = "" # No SE
            results_corrected_df.loc[idx_map['b1_phase_mean'], 'B'] = format_val_level(m['b1_phase']['mean1']); results_corrected_df.loc[idx_map['b1_phase_mean'], 'C'] = format_val_level(m['b1_phase']['mean2']); results_corrected_df.loc[idx_map['b1_phase_mean'], 'D'] = format_val_level(m['b1_phase']['diff'])
            results_corrected_df.loc[idx_map['b1_phase_se'], 'B'] = f"({format_val_level(m['b1_phase']['se1'])})"; results_corrected_df.loc[idx_map['b1_phase_se'], 'C'] = f"({format_val_level(m['b1_phase']['se2'])})"; results_corrected_df.loc[idx_map['b1_phase_se'], 'D'] = f"({format_val_level(m['b1_phase']['se_diff'])})"
            results_corrected_df.loc[idx_map['b1_phase_n'], 'B'] = f"[{m['b1_phase']['n1']}]"; results_corrected_df.loc[idx_map['b1_phase_n'], 'C'] = f"[{m['b1_phase']['n2']}]"
            results_corrected_df.loc[idx_map['b1_cdf_mean'], 'B'] = format_val_level(m['b1_cdf']['mean1']); results_corrected_df.loc[idx_map['b1_cdf_mean'], 'C'] = format_val_level(m['b1_cdf']['mean2']); results_corrected_df.loc[idx_map['b1_cdf_mean'], 'D'] = format_val_level(m['b1_cdf']['diff'])
            results_corrected_df.loc[idx_map['b1_cdf_se'], 'B'] = f"({format_val_level(m['b1_cdf']['se1'])})"; results_corrected_df.loc[idx_map['b1_cdf_se'], 'C'] = f"({format_val_level(m['b1_cdf']['se2'])})"; results_corrected_df.loc[idx_map['b1_cdf_se'], 'D'] = f"({format_val_level(m['b1_cdf']['se_diff'])})"
            results_corrected_df.loc[idx_map['b1_cdf_n'], 'B'] = f"[{m['b1_cdf']['n1']}]"; results_corrected_df.loc[idx_map['b1_cdf_n'], 'C'] = f"[{m['b1_cdf']['n2']}]"
            results_corrected_df.loc[idx_map['b1_dd_coef'], 'D'] = format_val_coef(m['b1_dd']['coef']); results_corrected_df.loc[idx_map['b1_dd_se'], 'D'] = "" # No SE
            results_corrected_df.loc[idx_map['b2_phase_mean'], 'B'] = format_val_level(m['b2_phase']['mean1']); results_corrected_df.loc[idx_map['b2_phase_mean'], 'C'] = format_val_level(m['b2_phase']['mean2']); results_corrected_df.loc[idx_map['b2_phase_mean'], 'D'] = format_val_level(m['b2_phase']['diff'])
            results_corrected_df.loc[idx_map['b2_phase_se'], 'B'] = f"({format_val_level(m['b2_phase']['se1'])})"; results_corrected_df.loc[idx_map['b2_phase_se'], 'C'] = f"({format_val_level(m['b2_phase']['se2'])})"; results_corrected_df.loc[idx_map['b2_phase_se'], 'D'] = f"({format_val_level(m['b2_phase']['se_diff'])})"
            results_corrected_df.loc[idx_map['b2_phase_n'], 'B'] = f"[{m['b2_phase']['n1']}]"; results_corrected_df.loc[idx_map['b2_phase_n'], 'C'] = f"[{m['b2_phase']['n2']}]"
            results_corrected_df.loc[idx_map['b2_cdf_mean'], 'B'] = format_val_level(m['b2_cdf']['mean1']); results_corrected_df.loc[idx_map['b2_cdf_mean'], 'C'] = format_val_level(m['b2_cdf']['mean2']); results_corrected_df.loc[idx_map['b2_cdf_mean'], 'D'] = format_val_level(m['b2_cdf']['diff'])
            results_corrected_df.loc[idx_map['b2_cdf_se'], 'B'] = f"({format_val_level(m['b2_cdf']['se1'])})"; results_corrected_df.loc[idx_map['b2_cdf_se'], 'C'] = f"({format_val_level(m['b2_cdf']['se2'])})"; results_corrected_df.loc[idx_map['b2_cdf_se'], 'D'] = f"({format_val_level(m['b2_cdf']['se_diff'])})"
            results_corrected_df.loc[idx_map['b2_cdf_n'], 'B'] = f"[{m['b2_cdf']['n1']}]"; results_corrected_df.loc[idx_map['b2_cdf_n'], 'C'] = f"[{m['b2_cdf']['n2']}]"
            results_corrected_df.loc[idx_map['b2_dd_coef'], 'D'] = format_val_coef(m['b2_dd']['coef']); results_corrected_df.loc[idx_map['b2_dd_se'], 'D'] = "" # No SE
            results_corrected_df.loc[idx_map['b_ddd_coef'], 'D'] = format_val_coef(m['b_ddd']['coef']); results_corrected_df.loc[idx_map['b_ddd_se'], 'D'] = "" # No SE

            # Canada Corrected
            c = results_corrected['can']
            results_corrected_df.loc[idx_map['a1_nafta_mean'], 'F'] = format_val_level(c['a1_nafta']['mean1']); results_corrected_df.loc[idx_map['a1_nafta_mean'], 'G'] = format_val_level(c['a1_nafta']['mean2']); results_corrected_df.loc[idx_map['a1_nafta_mean'], 'H'] = format_val_level(c['a1_nafta']['diff'])
            results_corrected_df.loc[idx_map['a1_nafta_se'], 'F'] = f"({format_val_level(c['a1_nafta']['se1'])})"; results_corrected_df.loc[idx_map['a1_nafta_se'], 'G'] = f"({format_val_level(c['a1_nafta']['se2'])})"; results_corrected_df.loc[idx_map['a1_nafta_se'], 'H'] = f"({format_val_level(c['a1_nafta']['se_diff'])})"
            results_corrected_df.loc[idx_map['a1_nafta_n'], 'F'] = f"[{c['a1_nafta']['n1']}]"; results_corrected_df.loc[idx_map['a1_nafta_n'], 'G'] = f"[{c['a1_nafta']['n2']}]"
            results_corrected_df.loc[idx_map['a1_row_mean'], 'F'] = format_val_level(c['a1_row']['mean1']); results_corrected_df.loc[idx_map['a1_row_mean'], 'G'] = format_val_level(c['a1_row']['mean2']); results_corrected_df.loc[idx_map['a1_row_mean'], 'H'] = format_val_level(c['a1_row']['diff'])
            results_corrected_df.loc[idx_map['a1_row_se'], 'F'] = f"({format_val_level(c['a1_row']['se1'])})"; results_corrected_df.loc[idx_map['a1_row_se'], 'G'] = f"({format_val_level(c['a1_row']['se2'])})"; results_corrected_df.loc[idx_map['a1_row_se'], 'H'] = f"({format_val_level(c['a1_row']['se_diff'])})"
            results_corrected_df.loc[idx_map['a1_row_n'], 'F'] = f"[{c['a1_row']['n1']}]"; results_corrected_df.loc[idx_map['a1_row_n'], 'G'] = f"[{c['a1_row']['n2']}]"
            results_corrected_df.loc[idx_map['a1_dd_coef'], 'H'] = format_val_coef(c['a1_dd']['coef']); results_corrected_df.loc[idx_map['a1_dd_se'], 'H'] = "" # No SE
            results_corrected_df.loc[idx_map['a2_nafta_mean'], 'F'] = format_val_level(c['a2_nafta']['mean1']); results_corrected_df.loc[idx_map['a2_nafta_mean'], 'G'] = format_val_level(c['a2_nafta']['mean2']); results_corrected_df.loc[idx_map['a2_nafta_mean'], 'H'] = format_val_level(c['a2_nafta']['diff'])
            results_corrected_df.loc[idx_map['a2_nafta_se'], 'F'] = f"({format_val_level(c['a2_nafta']['se1'])})"; results_corrected_df.loc[idx_map['a2_nafta_se'], 'G'] = f"({format_val_level(c['a2_nafta']['se2'])})"; results_corrected_df.loc[idx_map['a2_nafta_se'], 'H'] = f"({format_val_level(c['a2_nafta']['se_diff'])})"
            results_corrected_df.loc[idx_map['a2_nafta_n'], 'F'] = f"[{c['a2_nafta']['n1']}]"; results_corrected_df.loc[idx_map['a2_nafta_n'], 'G'] = f"[{c['a2_nafta']['n2']}]"
            results_corrected_df.loc[idx_map['a2_row_mean'], 'F'] = format_val_level(c['a2_row']['mean1']); results_corrected_df.loc[idx_map['a2_row_mean'], 'G'] = format_val_level(c['a2_row']['mean2']); results_corrected_df.loc[idx_map['a2_row_mean'], 'H'] = format_val_level(c['a2_row']['diff'])
            results_corrected_df.loc[idx_map['a2_row_se'], 'F'] = f"({format_val_level(c['a2_row']['se1'])})"; results_corrected_df.loc[idx_map['a2_row_se'], 'G'] = f"({format_val_level(c['a2_row']['se2'])})"; results_corrected_df.loc[idx_map['a2_row_se'], 'H'] = f"({format_val_level(c['a2_row']['se_diff'])})"
            results_corrected_df.loc[idx_map['a2_row_n'], 'F'] = f"[{c['a2_row']['n1']}]"; results_corrected_df.loc[idx_map['a2_row_n'], 'G'] = f"[{c['a2_row']['n2']}]"
            results_corrected_df.loc[idx_map['a2_dd_coef'], 'H'] = format_val_coef(c['a2_dd']['coef']); results_corrected_df.loc[idx_map['a2_dd_se'], 'H'] = "" # No SE
            results_corrected_df.loc[idx_map['a_ddd_coef'], 'H'] = format_val_coef(c['a_ddd']['coef']); results_corrected_df.loc[idx_map['a_ddd_se'], 'H'] = "" # No SE
            results_corrected_df.loc[idx_map['b1_phase_mean'], 'F'] = format_val_level(c['b1_phase']['mean1']); results_corrected_df.loc[idx_map['b1_phase_mean'], 'G'] = format_val_level(c['b1_phase']['mean2']); results_corrected_df.loc[idx_map['b1_phase_mean'], 'H'] = format_val_level(c['b1_phase']['diff'])
            results_corrected_df.loc[idx_map['b1_phase_se'], 'F'] = f"({format_val_level(c['b1_phase']['se1'])})"; results_corrected_df.loc[idx_map['b1_phase_se'], 'G'] = f"({format_val_level(c['b1_phase']['se2'])})"; results_corrected_df.loc[idx_map['b1_phase_se'], 'H'] = f"({format_val_level(c['b1_phase']['se_diff'])})"
            results_corrected_df.loc[idx_map['b1_phase_n'], 'F'] = f"[{c['b1_phase']['n1']}]"; results_corrected_df.loc[idx_map['b1_phase_n'], 'G'] = f"[{c['b1_phase']['n2']}]"
            results_corrected_df.loc[idx_map['b1_cdf_mean'], 'F'] = format_val_level(c['b1_cdf']['mean1']); results_corrected_df.loc[idx_map['b1_cdf_mean'], 'G'] = format_val_level(c['b1_cdf']['mean2']); results_corrected_df.loc[idx_map['b1_cdf_mean'], 'H'] = format_val_level(c['b1_cdf']['diff'])
            results_corrected_df.loc[idx_map['b1_cdf_se'], 'F'] = f"({format_val_level(c['b1_cdf']['se1'])})"; results_corrected_df.loc[idx_map['b1_cdf_se'], 'G'] = f"({format_val_level(c['b1_cdf']['se2'])})"; results_corrected_df.loc[idx_map['b1_cdf_se'], 'H'] = f"({format_val_level(c['b1_cdf']['se_diff'])})"
            results_corrected_df.loc[idx_map['b1_cdf_n'], 'F'] = f"[{c['b1_cdf']['n1']}]"; results_corrected_df.loc[idx_map['b1_cdf_n'], 'G'] = f"[{c['b1_cdf']['n2']}]"
            results_corrected_df.loc[idx_map['b1_dd_coef'], 'H'] = format_val_coef(c['b1_dd']['coef']); results_corrected_df.loc[idx_map['b1_dd_se'], 'H'] = "" # No SE
            results_corrected_df.loc[idx_map['b2_phase_mean'], 'F'] = format_val_level(c['b2_phase']['mean1']); results_corrected_df.loc[idx_map['b2_phase_mean'], 'G'] = format_val_level(c['b2_phase']['mean2']); results_corrected_df.loc[idx_map['b2_phase_mean'], 'H'] = format_val_level(c['b2_phase']['diff'])
            results_corrected_df.loc[idx_map['b2_phase_se'], 'F'] = f"({format_val_level(c['b2_phase']['se1'])})"; results_corrected_df.loc[idx_map['b2_phase_se'], 'G'] = f"({format_val_level(c['b2_phase']['se2'])})"; results_corrected_df.loc[idx_map['b2_phase_se'], 'H'] = f"({format_val_level(c['b2_phase']['se_diff'])})"
            results_corrected_df.loc[idx_map['b2_phase_n'], 'F'] = f"[{c['b2_phase']['n1']}]"; results_corrected_df.loc[idx_map['b2_phase_n'], 'G'] = f"[{c['b2_phase']['n2']}]"
            results_corrected_df.loc[idx_map['b2_cdf_mean'], 'F'] = format_val_level(c['b2_cdf']['mean1']); results_corrected_df.loc[idx_map['b2_cdf_mean'], 'G'] = format_val_level(c['b2_cdf']['mean2']); results_corrected_df.loc[idx_map['b2_cdf_mean'], 'H'] = format_val_level(c['b2_cdf']['diff'])
            results_corrected_df.loc[idx_map['b2_cdf_se'], 'F'] = f"({format_val_level(c['b2_cdf']['se1'])})"; results_corrected_df.loc[idx_map['b2_cdf_se'], 'G'] = f"({format_val_level(c['b2_cdf']['se2'])})"; results_corrected_df.loc[idx_map['b2_cdf_se'], 'H'] = f"({format_val_level(c['b2_cdf']['se_diff'])})"
            results_corrected_df.loc[idx_map['b2_cdf_n'], 'F'] = f"[{c['b2_cdf']['n1']}]"; results_corrected_df.loc[idx_map['b2_cdf_n'], 'G'] = f"[{c['b2_cdf']['n2']}]"
            results_corrected_df.loc[idx_map['b2_dd_coef'], 'H'] = format_val_coef(c['b2_dd']['coef']); results_corrected_df.loc[idx_map['b2_dd_se'], 'H'] = "" # No SE
            results_corrected_df.loc[idx_map['b_ddd_coef'], 'H'] = format_val_coef(c['b_ddd']['coef']); results_corrected_df.loc[idx_map['b_ddd_se'], 'H'] = "" # No SE

            results_corrected_df.to_excel(writer, sheet_name=f'Table2_Corrected_Results{file_suffix}', index=False, header=False)
            print(f"Corrected Log-Linear results for Table 2 saved to sheet Table2_Corrected_Results{file_suffix}")
        else:
            print(f"Skipping Corrected Log-Linear results table generation due to missing Mexico or Canada data for Corrected Estimator.")
    
    writer.close()
    print(f"Table 2 (OLS, PPML, Corrected) created and saved to {excel_file_path}")
    return True

# Function to create Table A3
def create_table_a3():
    """Generate data for Table A3: Tariff cuts by staging category.
    Reads schedulesUS.dta from dir_raw_data_base.
    Reads dataRC from dir_processed_data.
    Saves table to dir_figures_tables."""
    print("Creating Table A3...")
    schedules_path = os.path.join(dir_raw_data_base, "schedulesUS.dta") # Input from raw data
    if not os.path.exists(schedules_path):
        print(f"Warning: {schedules_path} not found. Creating dummy schedulesUS.dta in {dir_raw_data_base}.")
        dummy_schedules_data = {
            'hs8': [f'prod{i}' for i in range(100)], 'baseCUSFTA': np.random.rand(100)*20,
            'baseCanNAFTA': np.random.rand(100)*20, 'baseMexNAFTA': np.random.rand(100)*20,
            'catC': np.random.choice(['A', 'B', 'C', 'D', 'GSP', 'Mixed'], 100),
            'catEffC': np.random.choice(['A', 'B', 'C', 'D', 'GSP', 'Mixed'], 100),
            'catEffM': np.random.choice(['A', 'B', 'C', 'D', 'GSP', 'Mixed'], 100)
        }
        pd.DataFrame(dummy_schedules_data).to_stata(schedules_path, write_index=False)

    # Load processed raw codes data
    data_rc_path_pkl = os.path.join(dir_processed_data, "dataRC.pkl")
    data_rc_path_dta = os.path.join(dir_processed_data, "dataRC.dta")
    
    if os.path.exists(data_rc_path_pkl):
        df = pd.read_pickle(data_rc_path_pkl)
    elif os.path.exists(data_rc_path_dta):
        df = pd.read_stata(data_rc_path_dta)
    else:
        print(f"Error: dataRC.pkl or dataRC.dta not found in {dir_processed_data}. Cannot create Table A3.")
        print("Creating minimal dummy df for Table A3 to proceed...")
        df = pd.DataFrame({'hs8': [f'prod{i}' for i in range(10)], 
                           'catC': ['A']*10, 'catEffC': ['A']*10, 'catEffM': ['A']*10})
    
    schedules = pd.read_stata(schedules_path)
    df = pd.merge(df, schedules, on='hs8', how='left', suffixes=('', '_sched'))

    # Resolve duplicate columns if any, prioritize original df columns for cats
    for col_cat in ['catC', 'catEffC', 'catEffM']:
        if f"{col_cat}_sched" in df.columns and col_cat in df.columns:
             df[col_cat] = df[col_cat].fillna(df[f"{col_cat}_sched"]) # Fill if original is NaN
             df.drop(columns=[f"{col_cat}_sched"], inplace=True)
        elif f"{col_cat}_sched" in df.columns and col_cat not in df.columns:
             df.rename(columns={f"{col_cat}_sched": col_cat}, inplace=True)


    df = df.drop_duplicates(subset=['hs8'])

    for z in ['catC', 'catEffC', 'catEffM']:
        if z in df.columns:
            df.loc[df[z].isin(["missing code (see notes)", "see notes"]), z] = "Missing"
            df.loc[df[z].isin(["Mixed phasein (no D)", "Mixed phasein (incl D)"]), z] = "Mixed"
            if z == 'catC': df.loc[df['catC'] == "3 EQUAL ANNUAL STAGES", 'catC'] = "Missing"
        else: df[z] = "Missing" # Add column if missing (dummy data case)
    
    df['a'] = 1
    for base_col in ['baseCUSFTA', 'baseCanNAFTA', 'baseMexNAFTA']:
        if base_col not in df.columns: df[base_col] = 0 # For dummy data

    cusfta_agg = df.groupby('catC').agg(CUSFTAmean=('baseCUSFTA', 'mean'), CUSFTAobs=('a', 'sum')).reset_index().rename(columns={'catC': 'staging'})
    can_nafta_agg = df.groupby('catEffC').agg(CanNAFTAmean=('baseCanNAFTA', 'mean'), CanNAFTAobs=('a', 'sum')).reset_index().rename(columns={'catEffC': 'staging'})
    mex_nafta_agg = df.groupby('catEffM').agg(MexNAFTAmean=('baseMexNAFTA', 'mean'), MexNAFTAobs=('a', 'sum')).reset_index().rename(columns={'catEffM': 'staging'})

    combined = pd.merge(cusfta_agg, can_nafta_agg, on='staging', how='outer')
    combined = pd.merge(combined, mex_nafta_agg, on='staging', how='outer')

    combined['years'] = np.nan
    years_map = {'A': 1, 'B': 5, 'B6': 6, 'C': 10, 'C10': 10, 'C+': 15}
    combined['years'] = combined['staging'].map(years_map)

    for col in ['CUSFTAmean', 'CanNAFTAmean', 'MexNAFTAmean']:
        combined.loc[combined['staging'].isin(['D', 'Mixed', 'Missing']), col] = np.nan
    combined.loc[combined['staging'] == 'GSP', 'MexNAFTAmean'] = np.nan

    combined['CUSFTAannual'] = combined['CUSFTAmean'] / combined['years']
    combined['CanNAFTAannual'] = combined['CanNAFTAmean'] / combined['years']
    combined['MexNAFTAannual'] = combined['MexNAFTAmean'] / combined['years']

    combined = combined[combined['staging'] != ''] 
    combined = combined.dropna(subset=['staging'])
    combined = combined.sort_values(['years', 'staging'])
    combined = combined.drop(columns=['years'])

    final_cols = ['staging', 'CUSFTAobs', 'CUSFTAmean', 'CUSFTAannual', 'CanNAFTAobs', 'CanNAFTAmean', 'CanNAFTAannual', 'MexNAFTAobs', 'MexNAFTAmean', 'MexNAFTAannual']
    for fc in final_cols:
        if fc not in combined.columns: combined[fc] = np.nan
    combined = combined[final_cols]

    combined.to_excel(os.path.join(dir_figures_tables, "table_A3.xlsx"), sheet_name="TableA3RC_data", index=False)
    print("Table A3 created and saved.")


# Function to run DDD regressions (Original OLS with demeaning/HDFE)
def run_ddd_regressions(country, dependent_var_suffix, file_suffix):
    """
    Run OLS DDD regressions with high-dimensional fixed effects using linearmodels.
    Reads data from dir_processed_data.
    """
    print(f"Running OLS DDD (HDFE) regressions for {country}, DV suffix: {dependent_var_suffix}, file: {file_suffix}")
    if not linearmodels_available:
        print("linearmodels package not available. Skipping OLS DDD (HDFE) regressions.")
        return {"error": "linearmodels not available"}

    data_path_pkl = os.path.join(dir_processed_data, f"data{file_suffix}_{country}.pkl")
    data_path_dta = os.path.join(dir_processed_data, f"data{file_suffix}_{country}.dta")

    if os.path.exists(data_path_pkl): df = pd.read_pickle(data_path_pkl)
    elif os.path.exists(data_path_dta): df = pd.read_stata(data_path_dta)
    else:
        print(f"Error: Data file for {country} with suffix {file_suffix} not found in {dir_processed_data}.")
        dummy_data = { # Minimal dummy data
            'year': np.tile(range(1990,2000), 50), 'hs8': np.repeat([f'hs{i}' for i in range(50)], 10),
            'cntry': np.tile(np.arange(500) % 5, 10), # Dummy country groups
            f'{country}_cv': np.random.rand(500) * 1000 + 1, f'{country}_uv': np.random.rand(500) * 10 + 1,
            f'{country}_q': np.random.rand(500) * 100 + 1, 'NAFTA': np.random.randint(0,2,500),
            'treatCDF': np.random.randint(0,2,500), 'treatNonCDF': 1 - np.random.randint(0,2,500),
            'treatImmed': np.random.randint(0,2,500), 'treatGSP': np.random.randint(0,2,500) if country == 'Mex' else 0,
            'treatPhase5': np.random.randint(0,2,500), 'treatPhase10': np.random.randint(0,2,500),
        }
        df = pd.DataFrame(dummy_data)
        print("Using dummy data for OLS DDD (HDFE) regression.")

    y_col_original = f"{country}_{dependent_var_suffix}"
    y_var = f"ln_{y_col_original}"
    if y_col_original not in df.columns:
        print(f"Error: Original DV {y_col_original} not found for OLS HDFE.")
        return {"error": f"Original DV {y_col_original} not found"}
    
    df[y_var] = np.log(df[y_col_original].replace(0, np.nan))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['post'] = (df['year'] >= 1994).astype(int) # Paper uses 1994 for post-NAFTA

    interaction_cols = ['NAFTA', 'treatCDF', 'treatNonCDF', 'post', 'treatImmed', 'treatGSP', 'treatPhase5', 'treatPhase10']
    for col in interaction_cols:
        if col not in df.columns: df[col] = 0
        else: df[col] = df[col].astype(int)

    df['post_NAFTA'] = (df['post'] * df['NAFTA']).astype(int)
    df['post_treatCDF'] = (df['post'] * df['treatCDF']).astype(int)
    df['post_treatNonCDF'] = (df['post'] * df['treatNonCDF']).astype(int)
    df['NAFTA_treatCDF'] = (df['NAFTA'] * df['treatCDF']).astype(int)
    df['NAFTA_treatNonCDF'] = (df['NAFTA'] * df['treatNonCDF']).astype(int)
    df['post_NAFTA_treatCDF'] = (df['post'] * df['NAFTA'] * df['treatCDF']).astype(int)
    df['post_NAFTA_treatNonCDF'] = (df['post'] * df['NAFTA'] * df['treatNonCDF']).astype(int)

    treat_var_checks = []
    for treat in ['Immed', 'GSP', 'Phase5', 'Phase10']:
        if treat == 'GSP' and country == 'Can':
            treat_var_checks.append((treat, False)); continue
        treat_col = f'treat{treat}'
        if treat_col in df.columns:
            df[f'post_treat{treat}'] = (df['post'] * df[treat_col]).astype(int)
            df[f'NAFTA_treat{treat}'] = (df['NAFTA'] * df[treat_col]).astype(int)
            df[f'post_NAFTA_treat{treat}'] = (df['post'] * df['NAFTA'] * df[treat_col]).astype(int)
            treat_var_checks.append((treat, True))
        else: treat_var_checks.append((treat, False))

    fe_cols = ['hs8', 'cntry', 'year'] # product, country_group (exporter for US imports), year
    for fe_col in fe_cols:
        if fe_col not in df.columns: df[fe_col] = 0 # Dummy for missing FE col

    df_clean = df.dropna(subset=[y_var] + fe_cols).copy()
    df_clean = df_clean[~np.isinf(df_clean[y_var])]
    df_clean[y_var] = df_clean[y_var].astype(float)

    print(f"Clean data shape for OLS DDD (HDFE): {df_clean.shape}")
    if len(df_clean) < 100: print("Warning: Small sample size for OLS DDD (HDFE).")
    if len(df_clean) < 20: return {"error": "Sample too small for OLS DDD (HDFE)"}
    
    results = {}
    df_panel = df_clean.set_index(['hs8', 'cntry', 'year']) # Product-Exporter-Year FE

    # Spec 1: CDF vs Non-CDF
    exog_vars1 = ['post_NAFTA_treatCDF', 'post_NAFTA_treatNonCDF', 'post_NAFTA', 'post_treatCDF', 'post_treatNonCDF', 'NAFTA_treatCDF', 'NAFTA_treatNonCDF']
    missing_exog1 = [v for v in exog_vars1 if v not in df_panel.columns]
    if missing_exog1:
        print(f"Error: Missing exog variables for PanelOLS Spec1: {missing_exog1}")
        results['spec1_effects'] = {'CDF_effect': np.nan, 'NonCDF_effect': np.nan, 'Diff_CDF_NonCDF': np.nan}
    else:
        X1 = sm.add_constant(df_panel[exog_vars1]) # Constant will be absorbed by FEs if included, linearmodels handles this.
        try:
            model_lm1 = PanelOLS(df_panel[y_var], X1, entity_effects=True, time_effects=True, drop_absorbed=True)
            results_lm1 = model_lm1.fit(cov_type='clustered', cluster_entity=True, cluster_time=True) # 2-way cluster
            results['spec1'] = results_lm1.summary.tables[1].to_string() # Store summary table as string
            results['spec1_effects'] = {
                'CDF_effect': results_lm1.params['post_NAFTA_treatCDF'] if 'post_NAFTA_treatCDF' in results_lm1.params else np.nan,
                'NonCDF_effect': results_lm1.params['post_NAFTA_treatNonCDF'] if 'post_NAFTA_treatNonCDF' in results_lm1.params else np.nan
            }
            if pd.notna(results['spec1_effects']['CDF_effect']) and pd.notna(results['spec1_effects']['NonCDF_effect']):
                 results['spec1_effects']['Diff_CDF_NonCDF'] = results['spec1_effects']['CDF_effect'] - results['spec1_effects']['NonCDF_effect']
            else: results['spec1_effects']['Diff_CDF_NonCDF'] = np.nan
            print("\nOLS DDD (linearmodels - Spec 1) Results Summary Table:")
            print(results_lm1.summary.tables[1])
        except Exception as e:
            print(f"Error in linearmodels PanelOLS (Spec 1): {e}")
            results['spec1_effects'] = {'CDF_effect': np.nan, 'NonCDF_effect': np.nan, 'Diff_CDF_NonCDF': np.nan}
            results['spec1'] = f"Error: {e}"

    # Spec 2: Specific Treatment Types
    spec2_treat_vars_main = [f'post_NAFTA_treat{treat}' for treat, avail in treat_var_checks if avail]
    spec2_two_way_terms = []
    for treat, avail in treat_var_checks:
        if avail: spec2_two_way_terms.extend([f'post_treat{treat}', f'NAFTA_treat{treat}'])
    spec2_two_way_terms.append('post_NAFTA')
    exog_vars2 = spec2_treat_vars_main + list(set(spec2_two_way_terms))
    
    missing_exog2 = [v for v in exog_vars2 if v not in df_panel.columns]
    if not spec2_treat_vars_main or missing_exog2:
        print(f"No treatment vars or missing exog for PanelOLS Spec2. Missing: {missing_exog2}")
        results['spec2_effects'] = {f'{t}_effect': np.nan for t, avail in treat_var_checks if avail}
    else:
        X2 = sm.add_constant(df_panel[exog_vars2])
        try:
            model_lm2 = PanelOLS(df_panel[y_var], X2, entity_effects=True, time_effects=True, drop_absorbed=True)
            results_lm2 = model_lm2.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
            results['spec2'] = results_lm2.summary.tables[1].to_string()
            results['spec2_effects'] = {}
            for treat, available in treat_var_checks:
                if available:
                    term = f'post_NAFTA_treat{treat}'
                    results['spec2_effects'][f'{treat}_effect'] = results_lm2.params[term] if term in results_lm2.params else np.nan
            print("\nOLS DDD (linearmodels - Spec 2) Results Summary Table:")
            print(results_lm2.summary.tables[1])
        except Exception as e:
            print(f"Error in linearmodels PanelOLS (Spec 2): {e}")
            results['spec2_effects'] = {f'{t}_effect': np.nan for t, avail in treat_var_checks if avail}
            results['spec2'] = f"Error: {e}"
            
    return results


# Main function to run all analyses
def run_all_analyses():
    all_results_summary = {'regressions_HDFE': {}} # Store HDFE results here
    try:
        for file_type in ['rawCodes', 'consistentCodes']:
            try: prepare_ddd_data(file_type)
            except Exception as e: print(f"Error preparing {file_type} data: {e}")

        try: create_table_a3()
        except Exception as e: print(f"Error creating Table A3: {e}")

        try: create_table_2(file_suffix='CC')
        except Exception as e:
            print(f"Error creating Table 2: {e}")
            import traceback; traceback.print_exc()

        regression_configs_hdfe = [
            ('Mex', 'cv', 'CC'), ('Mex', 'uv', 'CC'), ('Mex', 'q', 'CC'),
            ('Can', 'cv', 'CC'), ('Can', 'uv', 'CC'), ('Can', 'q', 'CC'),
        ]
        for config in regression_configs_hdfe:
            try:
                key = f"OLS_DDD_HDFE_{config[0]}_{config[1]}_{config[2]}"
                all_results_summary['regressions_HDFE'][key] = run_ddd_regressions(*config)
                print(f"Successfully ran OLS DDD (HDFE) regression for {key}")
            except Exception as e:
                print(f"Error running OLS DDD (HDFE) for {config}: {e}")
        
        print("\nSummary of All OLS DDD (HDFE) Regression Results:")
        for key, result_dict in all_results_summary['regressions_HDFE'].items():
            print(f"\n{key}:")
            if isinstance(result_dict, dict) and 'error' not in result_dict:
                if 'spec1_effects' in result_dict: print("  Spec 1:", result_dict['spec1_effects'])
                if 'spec2_effects' in result_dict: print("  Spec 2:", result_dict['spec2_effects'])
                # if 'spec1' in result_dict : print(result_dict['spec1']) # Full table string
            elif isinstance(result_dict, dict) and 'error' in result_dict:
                print(f"  Error: {result_dict['error']}")
            else: print("  No valid results or unexpected format.")


        if all_results_summary['regressions_HDFE']:
            import pickle
            results_path = os.path.join(dir_estimates_base, 'all_HDFE_regression_results.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(all_results_summary['regressions_HDFE'], f)
            print(f"\nAll HDFE regression results saved to {results_path}")

    except Exception as e:
        print(f"Critical error in run_all_analyses: {e}")
        import traceback; traceback.print_exc()
    return all_results_summary

# Main execution block
if __name__ == "__main__":
    # Check for initial raw data files and create placeholders if missing, for structural testing.
    # These would typically be provided.
    dummy_raw_files = [
        os.path.join(dir_raw_data_base, "data_rawCodes.dta"),
        os.path.join(dir_raw_data_base, "data_consistentCodes.dta"),
        os.path.join(dir_raw_data_base, "schedulesUS.dta")
    ]
    for dummy_file_path in dummy_raw_files:
        if not os.path.exists(dummy_file_path):
            print(f"Placeholder: Raw input file {dummy_file_path} not found. Script might use internal dummies.")
            # For a true test, minimal dummy files should be created here if prepare_ddd_data doesn't handle their absence entirely.
            # prepare_ddd_data and create_table_a3 have internal dummy creation if primary file is missing.
            # So, we can just print a warning here.

    run_all_analyses()
