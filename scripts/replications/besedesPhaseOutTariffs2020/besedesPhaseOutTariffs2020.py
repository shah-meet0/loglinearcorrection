import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
import patsy
from scipy import sparse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# For high-dimensional fixed effects
try:
    import linearmodels
    from linearmodels.panel import PanelOLS
    linearmodels_available = True
except ImportError:
    print("Warning: linearmodels package not available. Install with 'pip install linearmodels' for better performance.")
    linearmodels_available = False

# Define directory structure (equivalent to Stata global variables)
dir_base = "./"
dir_data = os.path.join(dir_base, "raw-data/besedesPhaseOutTariffs2020")
dir_estimates = os.path.join(dir_base, "output/replications/besedesPhaseOutTariffs2020")
dir_figures_tables = os.path.join(dir_estimates, "figuresTables")
dir_appendix = os.path.join(dir_data, "appendix")

# Create directories if they don't exist
for directory in [dir_data, dir_estimates, dir_figures_tables, dir_appendix]:
    os.makedirs(directory, exist_ok=True)

# Function to prepare data for DDD analyses
def prepare_ddd_data(file_type):
    """
    Prepare data for difference-in-difference-in-differences analyses
    Equivalent to the 'PREPARE FOR DDD ANALYSES' section in do_all.do
    
    Args:
        file_type: String, either 'rawCodes' or 'consistentCodes'
    """
    print(f"Preparing DDD data with {file_type}...")
    
    # Load the main dataset
    df = pd.read_stata(os.path.join(dir_data, f"data_{file_type}.dta"))
    
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
    
    # Create variables for customs values and quantities
    for x in ['Mex', 'MexC', 'MexF', 'Can', 'CanC', 'CanF']:
        df[f'{x}_cv'] = df['r_cv']
        df[f'{x}_q'] = df['q']
    
    # Apply masks for robustness samples
    # Mexico samples
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
    
    # Canada samples
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
    
    # Calculate unit values
    for x in ['Mex', 'MexC', 'MexF', 'Can', 'CanC', 'CanF']:
        # Handle division by zero properly - replace infinities with NaN
        df[f'{x}_uv'] = np.divide(df[f'{x}_cv'], df[f'{x}_q'], out=np.full_like(df[f'{x}_cv'], np.nan), where=df[f'{x}_q']!=0)
        # Additional safety check to remove any remaining infinities
        df[f'{x}_uv'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Save the processed dataset - use pickle to avoid Stata format limitations
    df.to_pickle(os.path.join(dir_data, f"data2_{file_type}.pkl"))
    
    # If you still need Stata format, save a clean version without problematic values
    df_stata = df.copy()
    # Replace any remaining infinities with NaN for all columns
    for col in df_stata.columns:
        if df_stata[col].dtype in [np.float32, np.float64]:
            df_stata[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    try:
        df_stata.to_stata(os.path.join(dir_data, f"data2_{file_type}.dta"), write_index=False)
        print(f"Successfully saved Stata file for {file_type}")
    except ValueError as e:
        print(f"Warning: Could not save to Stata format due to: {e}")
        print("Using pickle format instead")
    
    # Create separate datasets for Mexico and Canada
    # Mexico dataset
    mex_df = df.copy()
    # Drop observations with mixed or missing categories
    mex_df = mex_df[~mex_df['catEffMex'].isin(['Mixed phasein (incl D)', 'Mixed phasein (no D)', 
                                           'missing code (see notes)', 'see notes'])]
    mex_df = mex_df[~mex_df['catEffMex'].isna()]
    
    # Drop Canadian variables
    columns_to_drop = [col for col in mex_df.columns if 'Can' in col]
    mex_df = mex_df.drop(columns=columns_to_drop)
    
    # Create treatment variables
    mex_df['treatCDF'] = (mex_df['catEffMex'] == 'D').astype(int)
    mex_df['treatNonCDF'] = (mex_df['catEffMex'] != 'D').astype(int)
    mex_df['treatImmed'] = (mex_df['catEffMex'] == 'A').astype(int)
    mex_df['treatGSP'] = (mex_df['catEffMex'] == 'GSP').astype(int)
    mex_df['treatPhase5'] = ((mex_df['catEffMex'] == 'B') | (mex_df['catEffMex'] == 'B6')).astype(int)
    mex_df['treatPhase10'] = ((mex_df['catEffMex'] == 'C') | (mex_df['catEffMex'] == 'C+') | 
                           (mex_df['catEffMex'] == 'C10')).astype(int)
    
    mex_df['NAFTA'] = (mex_df['country'] == 'Mexico').astype(int)
    
    # Create country group variable
    mex_df['cntry'] = pd.factorize(mex_df['country'])[0] + 1
    
    # Save Mexico dataset - use pickle format primarily
    mex_df.to_pickle(os.path.join(dir_data, f"data_Mex_{file_type}.pkl"))
    
    # Try to save as Stata but catch errors
    try:
        # Replace any infinities with NaN for all columns
        mex_stata = mex_df.copy()
        for col in mex_stata.columns:
            if mex_stata[col].dtype in [np.float32, np.float64]:
                mex_stata[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        mex_stata.to_stata(os.path.join(dir_data, f"data_Mex_{file_type}.dta"), write_index=False)
    except ValueError as e:
        print(f"Warning: Could not save Mexico data to Stata format due to: {e}")
        print("Using pickle format for Mexico data")
    
    # Canada dataset
    can_df = df.copy()
    # Drop observations with mixed or missing categories
    can_df = can_df[~can_df['catEffCan'].isin(['Mixed phasein (incl D)', 'Mixed phasein (no D)', 
                                           'missing code (see notes)', 'see notes'])]
    can_df = can_df[~can_df['catEffCan'].isna()]
    
    # Drop Mexican variables
    columns_to_drop = [col for col in can_df.columns if 'Mex' in col]
    can_df = can_df.drop(columns=columns_to_drop)
    
    # Create treatment variables
    can_df['treatCDF'] = (can_df['catEffCan'] == 'D').astype(int)
    can_df['treatNonCDF'] = (can_df['catEffCan'] != 'D').astype(int)
    can_df['treatImmed'] = (can_df['catEffCan'] == 'A').astype(int)
    can_df['treatPhase5'] = ((can_df['catEffCan'] == 'B') | (can_df['catEffCan'] == 'B6')).astype(int)
    can_df['treatPhase10'] = ((can_df['catEffCan'] == 'C') | (can_df['catEffCan'] == 'C+') | 
                           (can_df['catEffCan'] == 'C10')).astype(int)
    
    can_df['NAFTA'] = (can_df['country'] == 'Canada').astype(int)
    
    # Create country group variable
    can_df['cntry'] = pd.factorize(can_df['country'])[0] + 1
    
    # Save Canada dataset - use pickle format primarily
    can_df.to_pickle(os.path.join(dir_data, f"data_Can_{file_type}.pkl"))
    
    # Try to save as Stata but catch errors
    try:
        # Replace any infinities with NaN for all columns
        can_stata = can_df.copy()
        for col in can_stata.columns:
            if can_stata[col].dtype in [np.float32, np.float64]:
                can_stata[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        can_stata.to_stata(os.path.join(dir_data, f"data_Can_{file_type}.dta"), write_index=False)
    except ValueError as e:
        print(f"Warning: Could not save Canada data to Stata format due to: {e}")
        print("Using pickle format for Canada data")
    
    # Rename files to match Stata convention
    if file_type == 'rawCodes':
        # For pickle files
        os.rename(os.path.join(dir_data, f"data2_{file_type}.pkl"), 
                  os.path.join(dir_data, "dataRC.pkl"))
        os.rename(os.path.join(dir_data, f"data_Mex_{file_type}.pkl"), 
                  os.path.join(dir_data, "dataRC_Mex.pkl"))
        os.rename(os.path.join(dir_data, f"data_Can_{file_type}.pkl"), 
                  os.path.join(dir_data, "dataRC_Can.pkl"))
        # For Stata files if they exist
        stata_path = os.path.join(dir_data, f"data2_{file_type}.dta")
        if os.path.exists(stata_path):
            os.rename(stata_path, os.path.join(dir_data, "dataRC.dta"))
        
        stata_path = os.path.join(dir_data, f"data_Mex_{file_type}.dta")
        if os.path.exists(stata_path):
            os.rename(stata_path, os.path.join(dir_data, "dataRC_Mex.dta"))
        
        stata_path = os.path.join(dir_data, f"data_Can_{file_type}.dta")
        if os.path.exists(stata_path):
            os.rename(stata_path, os.path.join(dir_data, "dataRC_Can.dta"))
            
    elif file_type == 'consistentCodes':
        # For pickle files
        os.rename(os.path.join(dir_data, f"data2_{file_type}.pkl"), 
                  os.path.join(dir_data, "dataCC.pkl"))
        os.rename(os.path.join(dir_data, f"data_Mex_{file_type}.pkl"), 
                  os.path.join(dir_data, "dataCC_Mex.pkl"))
        os.rename(os.path.join(dir_data, f"data_Can_{file_type}.pkl"), 
                  os.path.join(dir_data, "dataCC_Can.pkl"))
        # For Stata files if they exist
        stata_path = os.path.join(dir_data, f"data2_{file_type}.dta")
        if os.path.exists(stata_path):
            os.rename(stata_path, os.path.join(dir_data, "dataCC.dta"))
        
        stata_path = os.path.join(dir_data, f"data_Mex_{file_type}.dta")
        if os.path.exists(stata_path):
            os.rename(stata_path, os.path.join(dir_data, "dataCC_Mex.dta"))
        
        stata_path = os.path.join(dir_data, f"data_Can_{file_type}.dta")
        if os.path.exists(stata_path):
            os.rename(stata_path, os.path.join(dir_data, "dataCC_Can.dta"))

# Function to create Table 2
def create_table_2(file_suffix='CC'):
    """
    Creates Table 2: Time-invariant DDD estimates of NAFTA
    
    Args:
        file_suffix: String, file suffix to use (e.g., 'CC' for consistent codes)
    """
    print(f"Creating Table 2 with {file_suffix} data...")
    
    # Initialize the Excel writer
    writer = pd.ExcelWriter(os.path.join(dir_figures_tables, f"table_2_{file_suffix}.xlsx"), engine='openpyxl')
    
    # Create headers for the table
    headers = pd.DataFrame({
        'A': ["Table 2: Time-invariant DDD estimates of NAFTA", 
              "", 
              "Panel A: NAFTA vs ROW approach",
              "",
              "A1. Phase-out products",
              "Mexico", 
              "Pre-NAFTA", "Post-NAFTA", "Growth", 
              "NAFTA partner", "", "", "",
              "ROW", "", "", "",
              "DD", "", 
              "",
              "A2. CDF products",
              "Mexico", 
              "Pre-NAFTA", "Post-NAFTA", "Growth", 
              "NAFTA partner", "", "", "",
              "ROW", "", "", "",
              "DD", "", 
              "DDD", "",
              "(continued on next page)",
              "",
              "Table 3 (continued)",
              "",
              "Panel B: Phase-out vs CDF-products approach",
              "",
              "B1. NAFTA partner",
              "Mexico", 
              "Pre-NAFTA", "Post-NAFTA", "Growth", 
              "Phase-out products", "", "", "",
              "CDF products", "", "", "",
              "DD", "", 
              "",
              "B2. ROW",
              "Mexico", 
              "Pre-NAFTA", "Post-NAFTA", "Growth", 
              "Phase-out products", "", "", "",
              "CDF products", "", "", "",
              "DD", "", 
              "DDD", ""
             ],
        'E': ["", 
              "", 
              "",
              "",
              "",
              "Canada", 
              "Pre-NAFTA", "Post-NAFTA", "Growth", 
              "NAFTA partner", "", "", "",
              "ROW", "", "", "",
              "DD", "", 
              "",
              "",
              "Canada", 
              "Pre-NAFTA", "Post-NAFTA", "Growth", 
              "NAFTA partner", "", "", "",
              "ROW", "", "", "",
              "DD", "", 
              "DDD", "",
              "",
              "",
              "",
              "",
              "",
              "",
              "",
              "Canada", 
              "Pre-NAFTA", "Post-NAFTA", "Growth", 
              "Phase-out products", "", "", "",
              "CDF products", "", "", "",
              "DD", "", 
              "",
              "",
              "Canada", 
              "Pre-NAFTA", "Post-NAFTA", "Growth", 
              "Phase-out products", "", "", "",
              "CDF products", "", "", "",
              "DD", "", 
              "DDD", ""
             ]
    })
    
    # Write headers to Excel
    headers.to_excel(writer, sheet_name=f'Table2Input{file_suffix}', index=False)
    
    # Results container for all data
    results = {}
    
    # Function for t-test
    def run_ttest(data, value_col, filter_cond, group_col):
        subset = data[filter_cond].copy()
        group1 = subset[subset[group_col] == 0][value_col]
        group2 = subset[subset[group_col] == 1][value_col]
        
        if len(group1) == 0 or len(group2) == 0:
            return {
                'mean1': np.nan, 'mean2': np.nan, 'diff': np.nan,
                'se1': np.nan, 'se2': np.nan, 'se_diff': np.nan,
                'n1': 0, 'n2': 0
            }
        
        # Run t-test
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        
        # Calculate standard errors
        se1 = group1.std() / np.sqrt(len(group1)) if len(group1) > 0 else np.nan
        se2 = group2.std() / np.sqrt(len(group2)) if len(group2) > 0 else np.nan
        
        # Calculate pooled standard error for the difference
        se_diff = np.sqrt(se1**2 + se2**2) if not np.isnan(se1) and not np.isnan(se2) else np.nan
        
        return {
            'mean1': group1.mean(), 'mean2': group2.mean(), 
            'diff': group2.mean() - group1.mean(),
            'se1': se1, 'se2': se2, 'se_diff': se_diff,
            'n1': len(group1), 'n2': len(group2)
        }
    
    # Function for running DD regression
    def run_dd_regression(data, y, treat_var, time_var, filter_cond=None):
        if filter_cond is not None:
            subset = data[filter_cond].copy()
        else:
            subset = data.copy()
        
        # Create interaction term
        subset['treat_time'] = subset[treat_var] * subset[time_var]
        
        # Prepare regression variables
        X = subset[[treat_var, time_var, 'treat_time']]
        X = sm.add_constant(X)
        y_vals = subset[y]
        
        # Run regression
        model = sm.OLS(y_vals, X)
        results = model.fit(cov_type='HC1')  # Robust standard errors
        
        return {
            'coef': results.params['treat_time'],
            'se': results.bse['treat_time']
        }
    
    # Function for running DDD regression
    def run_ddd_regression(data, y, treat_var, time_var, diff_var):
        # Create interaction terms
        data['treat_time'] = data[treat_var] * data[time_var]
        data['treat_diff'] = data[treat_var] * data[diff_var]
        data['time_diff'] = data[time_var] * data[diff_var]
        data['treat_time_diff'] = data[treat_var] * data[time_var] * data[diff_var]
        
        # Prepare regression variables
        X = data[[treat_var, time_var, diff_var, 'treat_time', 'treat_diff', 'time_diff', 'treat_time_diff']]
        X = sm.add_constant(X)
        y_vals = data[y]
        
        # Run regression
        model = sm.OLS(y_vals, X)
        results = model.fit(cov_type='HC1')  # Robust standard errors
        
        return {
            'coef': results.params['treat_time_diff'],
            'se': results.bse['treat_time_diff']
        }
    
    # Process Mexico data
    try:
        # Read the data
        mexico_file = os.path.join(dir_data, f"data{file_suffix}_Mex.pkl")
        if os.path.exists(mexico_file):
            mex_df = pd.read_pickle(mexico_file)
        else:
            mex_df = pd.read_stata(os.path.join(dir_data, f"data{file_suffix}_Mex.dta"))
        
        # Keep relevant columns and prepare variables
        mex_cols = ['year', 'country', 'cntry', 'hs8', f'Mex_cv', 'catEffMex']
        mex_df = mex_df[mex_df.columns.intersection(mex_cols)].copy()
        
        mex_df['NAFTA'] = (mex_df['country'] == 'Mexico').astype(int)
        mex_df['ROW'] = (~mex_df['NAFTA'].astype(bool)).astype(int)
        mex_df['phase'] = (mex_df['catEffMex'] != 'D').astype(int)
        mex_df['CDF'] = (mex_df['catEffMex'] == 'D').astype(int)
        mex_df['pre'] = (mex_df['year'] < 1993).astype(int)
        mex_df['post'] = (mex_df['year'] >= 1993).astype(int)
        mex_df['lnMex'] = np.log(mex_df['Mex_cv'])
        
        # Run t-tests for Panel A: NAFTA vs ROW approach
        # A1. Phase-out products
        # For NAFTA partner
        mex_a1_nafta = run_ttest(mex_df, 'lnMex', (mex_df['phase'] == 1) & (mex_df['NAFTA'] == 1), 'post')
        
        # For ROW
        mex_a1_row = run_ttest(mex_df, 'lnMex', (mex_df['phase'] == 1) & (mex_df['ROW'] == 1), 'post')
        
        # Run DD regression for A1
        mex_a1_dd = run_dd_regression(mex_df, 'lnMex', 'NAFTA', 'post', (mex_df['phase'] == 1))
        
        # A2. CDF products
        # For NAFTA partner
        mex_a2_nafta = run_ttest(mex_df, 'lnMex', (mex_df['CDF'] == 1) & (mex_df['NAFTA'] == 1), 'post')
        
        # For ROW
        mex_a2_row = run_ttest(mex_df, 'lnMex', (mex_df['CDF'] == 1) & (mex_df['ROW'] == 1), 'post')
        
        # Run DD regression for A2
        mex_a2_dd = run_dd_regression(mex_df, 'lnMex', 'NAFTA', 'post', (mex_df['CDF'] == 1))
        
        # Run DDD regression for Panel A
        mex_a_ddd = run_ddd_regression(mex_df, 'lnMex', 'NAFTA', 'post', 'phase')
        
        # Panel B: Phase-out vs CDF-products approach
        # B1. NAFTA partner
        # For phase-out products
        mex_b1_phase = run_ttest(mex_df, 'lnMex', (mex_df['phase'] == 1) & (mex_df['NAFTA'] == 1), 'post')
        
        # For CDF products
        mex_b1_cdf = run_ttest(mex_df, 'lnMex', (mex_df['CDF'] == 1) & (mex_df['NAFTA'] == 1), 'post')
        
        # Run DD regression for B1
        mex_b1_dd = run_dd_regression(mex_df, 'lnMex', 'phase', 'post', (mex_df['NAFTA'] == 1))
        
        # B2. ROW
        # For phase-out products
        mex_b2_phase = run_ttest(mex_df, 'lnMex', (mex_df['phase'] == 1) & (mex_df['ROW'] == 1), 'post')
        
        # For CDF products
        mex_b2_cdf = run_ttest(mex_df, 'lnMex', (mex_df['CDF'] == 1) & (mex_df['ROW'] == 1), 'post')
        
        # Run DD regression for B2
        mex_b2_dd = run_dd_regression(mex_df, 'lnMex', 'phase', 'post', (mex_df['ROW'] == 1))
        
        # Run DDD regression for Panel B
        mex_b_ddd = run_ddd_regression(mex_df, 'lnMex', 'NAFTA', 'post', 'phase')
        
        # Store Mexico results
        results['mex'] = {
            'a1_nafta': mex_a1_nafta, 'a1_row': mex_a1_row, 'a1_dd': mex_a1_dd,
            'a2_nafta': mex_a2_nafta, 'a2_row': mex_a2_row, 'a2_dd': mex_a2_dd, 'a_ddd': mex_a_ddd,
            'b1_phase': mex_b1_phase, 'b1_cdf': mex_b1_cdf, 'b1_dd': mex_b1_dd,
            'b2_phase': mex_b2_phase, 'b2_cdf': mex_b2_cdf, 'b2_dd': mex_b2_dd, 'b_ddd': mex_b_ddd
        }
        
    except Exception as e:
        print(f"Error processing Mexico data: {e}")
        results['mex'] = {}
    
    # Process Canada data
    try:
        # Read the data
        canada_file = os.path.join(dir_data, f"data{file_suffix}_Can.pkl")
        if os.path.exists(canada_file):
            can_df = pd.read_pickle(canada_file)
        else:
            can_df = pd.read_stata(os.path.join(dir_data, f"data{file_suffix}_Can.dta"))
        
        # Keep relevant columns and prepare variables
        can_cols = ['year', 'country', 'cntry', 'hs8', f'Can_cv', 'catEffCan']
        can_df = can_df[can_df.columns.intersection(can_cols)].copy()
        
        can_df['NAFTA'] = (can_df['country'] == 'Canada').astype(int)
        can_df['ROW'] = (~can_df['NAFTA'].astype(bool)).astype(int)
        can_df['phase'] = (can_df['catEffCan'] != 'D').astype(int)
        can_df['CDF'] = (can_df['catEffCan'] == 'D').astype(int)
        can_df['pre'] = (can_df['year'] < 1993).astype(int)
        can_df['post'] = (can_df['year'] >= 1993).astype(int)
        can_df['lnCan'] = np.log(can_df['Can_cv'])
        
        # Run t-tests for Panel A: NAFTA vs ROW approach
        # A1. Phase-out products
        # For NAFTA partner
        can_a1_nafta = run_ttest(can_df, 'lnCan', (can_df['phase'] == 1) & (can_df['NAFTA'] == 1), 'post')
        
        # For ROW
        can_a1_row = run_ttest(can_df, 'lnCan', (can_df['phase'] == 1) & (can_df['ROW'] == 1), 'post')
        
        # Run DD regression for A1
        can_a1_dd = run_dd_regression(can_df, 'lnCan', 'NAFTA', 'post', (can_df['phase'] == 1))
        
        # A2. CDF products
        # For NAFTA partner
        can_a2_nafta = run_ttest(can_df, 'lnCan', (can_df['CDF'] == 1) & (can_df['NAFTA'] == 1), 'post')
        
        # For ROW
        can_a2_row = run_ttest(can_df, 'lnCan', (can_df['CDF'] == 1) & (can_df['ROW'] == 1), 'post')
        
        # Run DD regression for A2
        can_a2_dd = run_dd_regression(can_df, 'lnCan', 'NAFTA', 'post', (can_df['CDF'] == 1))
        
        # Run DDD regression for Panel A
        can_a_ddd = run_ddd_regression(can_df, 'lnCan', 'NAFTA', 'post', 'phase')
        
        # Panel B: Phase-out vs CDF-products approach
        # B1. NAFTA partner
        # For phase-out products
        can_b1_phase = run_ttest(can_df, 'lnCan', (can_df['phase'] == 1) & (can_df['NAFTA'] == 1), 'post')
        
        # For CDF products
        can_b1_cdf = run_ttest(can_df, 'lnCan', (can_df['CDF'] == 1) & (can_df['NAFTA'] == 1), 'post')
        
        # Run DD regression for B1
        can_b1_dd = run_dd_regression(can_df, 'lnCan', 'phase', 'post', (can_df['NAFTA'] == 1))
        
        # B2. ROW
        # For phase-out products
        can_b2_phase = run_ttest(can_df, 'lnCan', (can_df['phase'] == 1) & (can_df['ROW'] == 1), 'post')
        
        # For CDF products
        can_b2_cdf = run_ttest(can_df, 'lnCan', (can_df['CDF'] == 1) & (can_df['ROW'] == 1), 'post')
        
        # Run DD regression for B2
        can_b2_dd = run_dd_regression(can_df, 'lnCan', 'phase', 'post', (can_df['ROW'] == 1))
        
        # Run DDD regression for Panel B
        can_b_ddd = run_ddd_regression(can_df, 'lnCan', 'NAFTA', 'post', 'phase')
        
        # Store Canada results
        results['can'] = {
            'a1_nafta': can_a1_nafta, 'a1_row': can_a1_row, 'a1_dd': can_a1_dd,
            'a2_nafta': can_a2_nafta, 'a2_row': can_a2_row, 'a2_dd': can_a2_dd, 'a_ddd': can_a_ddd,
            'b1_phase': can_b1_phase, 'b1_cdf': can_b1_cdf, 'b1_dd': can_b1_dd,
            'b2_phase': can_b2_phase, 'b2_cdf': can_b2_cdf, 'b2_dd': can_b2_dd, 'b_ddd': can_b_ddd
        }
        
    except Exception as e:
        print(f"Error processing Canada data: {e}")
        results['can'] = {}
    
    # Create results table
    if 'mex' in results and 'can' in results:
        # Function to format values
        def format_val(val, format_str="{:.3f}"):
            if pd.isna(val):
                return ""
            return format_str.format(val)
        
        # Create results DataFrame
        results_df = pd.DataFrame(index=range(59), columns=list('ABCDEFGHIJ'))
        
        # Fill in Mexico results for Panel A, A1
        if 'a1_nafta' in results['mex']:
            # NAFTA partner
            results_df.loc[7, 'B'] = format_val(results['mex']['a1_nafta']['mean1'])
            results_df.loc[7, 'C'] = format_val(results['mex']['a1_nafta']['mean2'])
            results_df.loc[7, 'D'] = format_val(results['mex']['a1_nafta']['diff'])
            
            results_df.loc[8, 'B'] = f"({format_val(results['mex']['a1_nafta']['se1'])})"
            results_df.loc[8, 'C'] = f"({format_val(results['mex']['a1_nafta']['se2'])})"
            results_df.loc[8, 'D'] = f"({format_val(results['mex']['a1_nafta']['se_diff'])})"
            
            results_df.loc[9, 'B'] = f"[{results['mex']['a1_nafta']['n1']}]"
            results_df.loc[9, 'C'] = f"[{results['mex']['a1_nafta']['n2']}]"
        
        if 'a1_row' in results['mex']:
            # ROW
            results_df.loc[10, 'B'] = format_val(results['mex']['a1_row']['mean1'])
            results_df.loc[10, 'C'] = format_val(results['mex']['a1_row']['mean2'])
            results_df.loc[10, 'D'] = format_val(results['mex']['a1_row']['diff'])
            
            results_df.loc[11, 'B'] = f"({format_val(results['mex']['a1_row']['se1'])})"
            results_df.loc[11, 'C'] = f"({format_val(results['mex']['a1_row']['se2'])})"
            results_df.loc[11, 'D'] = f"({format_val(results['mex']['a1_row']['se_diff'])})"
            
            results_df.loc[12, 'B'] = f"[{results['mex']['a1_row']['n1']}]"
            results_df.loc[12, 'C'] = f"[{results['mex']['a1_row']['n2']}]"
        
        if 'a1_dd' in results['mex']:
            # DD
            results_df.loc[13, 'D'] = format_val(results['mex']['a1_dd']['coef'])
            results_df.loc[14, 'D'] = f"({format_val(results['mex']['a1_dd']['se'])})"
        
        # Fill in Mexico results for Panel A, A2
        if 'a2_nafta' in results['mex']:
            # NAFTA partner
            results_df.loc[18, 'B'] = format_val(results['mex']['a2_nafta']['mean1'])
            results_df.loc[18, 'C'] = format_val(results['mex']['a2_nafta']['mean2'])
            results_df.loc[18, 'D'] = format_val(results['mex']['a2_nafta']['diff'])
            
            results_df.loc[19, 'B'] = f"({format_val(results['mex']['a2_nafta']['se1'])})"
            results_df.loc[19, 'C'] = f"({format_val(results['mex']['a2_nafta']['se2'])})"
            results_df.loc[19, 'D'] = f"({format_val(results['mex']['a2_nafta']['se_diff'])})"
            
            results_df.loc[20, 'B'] = f"[{results['mex']['a2_nafta']['n1']}]"
            results_df.loc[20, 'C'] = f"[{results['mex']['a2_nafta']['n2']}]"
        
        if 'a2_row' in results['mex']:
            # ROW
            results_df.loc[21, 'B'] = format_val(results['mex']['a2_row']['mean1'])
            results_df.loc[21, 'C'] = format_val(results['mex']['a2_row']['mean2'])
            results_df.loc[21, 'D'] = format_val(results['mex']['a2_row']['diff'])
            
            results_df.loc[22, 'B'] = f"({format_val(results['mex']['a2_row']['se1'])})"
            results_df.loc[22, 'C'] = f"({format_val(results['mex']['a2_row']['se2'])})"
            results_df.loc[22, 'D'] = f"({format_val(results['mex']['a2_row']['se_diff'])})"
            
            results_df.loc[23, 'B'] = f"[{results['mex']['a2_row']['n1']}]"
            results_df.loc[23, 'C'] = f"[{results['mex']['a2_row']['n2']}]"
        
        if 'a2_dd' in results['mex']:
            # DD
            results_df.loc[24, 'D'] = format_val(results['mex']['a2_dd']['coef'])
            results_df.loc[25, 'D'] = f"({format_val(results['mex']['a2_dd']['se'])})"
        
        if 'a_ddd' in results['mex']:
            # DDD
            results_df.loc[26, 'D'] = format_val(results['mex']['a_ddd']['coef'])
            results_df.loc[27, 'D'] = f"({format_val(results['mex']['a_ddd']['se'])})"
        
        # Fill in Mexico results for Panel B, B1
        if 'b1_phase' in results['mex']:
            # Phase-out products
            results_df.loc[37, 'B'] = format_val(results['mex']['b1_phase']['mean1'])
            results_df.loc[37, 'C'] = format_val(results['mex']['b1_phase']['mean2'])
            results_df.loc[37, 'D'] = format_val(results['mex']['b1_phase']['diff'])
            
            results_df.loc[38, 'B'] = f"({format_val(results['mex']['b1_phase']['se1'])})"
            results_df.loc[38, 'C'] = f"({format_val(results['mex']['b1_phase']['se2'])})"
            results_df.loc[38, 'D'] = f"({format_val(results['mex']['b1_phase']['se_diff'])})"
            
            results_df.loc[39, 'B'] = f"[{results['mex']['b1_phase']['n1']}]"
            results_df.loc[39, 'C'] = f"[{results['mex']['b1_phase']['n2']}]"
        
        if 'b1_cdf' in results['mex']:
            # CDF products
            results_df.loc[40, 'B'] = format_val(results['mex']['b1_cdf']['mean1'])
            results_df.loc[40, 'C'] = format_val(results['mex']['b1_cdf']['mean2'])
            results_df.loc[40, 'D'] = format_val(results['mex']['b1_cdf']['diff'])
            
            results_df.loc[41, 'B'] = f"({format_val(results['mex']['b1_cdf']['se1'])})"
            results_df.loc[41, 'C'] = f"({format_val(results['mex']['b1_cdf']['se2'])})"
            results_df.loc[41, 'D'] = f"({format_val(results['mex']['b1_cdf']['se_diff'])})"
            
            results_df.loc[42, 'B'] = f"[{results['mex']['b1_cdf']['n1']}]"
            results_df.loc[42, 'C'] = f"[{results['mex']['b1_cdf']['n2']}]"
        
        if 'b1_dd' in results['mex']:
            # DD
            results_df.loc[43, 'D'] = format_val(results['mex']['b1_dd']['coef'])
            results_df.loc[44, 'D'] = f"({format_val(results['mex']['b1_dd']['se'])})"
        
        # Fill in Mexico results for Panel B, B2
        if 'b2_phase' in results['mex']:
            # Phase-out products
            results_df.loc[48, 'B'] = format_val(results['mex']['b2_phase']['mean1'])
            results_df.loc[48, 'C'] = format_val(results['mex']['b2_phase']['mean2'])
            results_df.loc[48, 'D'] = format_val(results['mex']['b2_phase']['diff'])
            
            results_df.loc[49, 'B'] = f"({format_val(results['mex']['b2_phase']['se1'])})"
            results_df.loc[49, 'C'] = f"({format_val(results['mex']['b2_phase']['se2'])})"
            results_df.loc[49, 'D'] = f"({format_val(results['mex']['b2_phase']['se_diff'])})"
            
            results_df.loc[50, 'B'] = f"[{results['mex']['b2_phase']['n1']}]"
            results_df.loc[50, 'C'] = f"[{results['mex']['b2_phase']['n2']}]"
        
        if 'b2_cdf' in results['mex']:
            # CDF products
            results_df.loc[51, 'B'] = format_val(results['mex']['b2_cdf']['mean1'])
            results_df.loc[51, 'C'] = format_val(results['mex']['b2_cdf']['mean2'])
            results_df.loc[51, 'D'] = format_val(results['mex']['b2_cdf']['diff'])
            
            results_df.loc[52, 'B'] = f"({format_val(results['mex']['b2_cdf']['se1'])})"
            results_df.loc[52, 'C'] = f"({format_val(results['mex']['b2_cdf']['se2'])})"
            results_df.loc[52, 'D'] = f"({format_val(results['mex']['b2_cdf']['se_diff'])})"
            
            results_df.loc[53, 'B'] = f"[{results['mex']['b2_cdf']['n1']}]"
            results_df.loc[53, 'C'] = f"[{results['mex']['b2_cdf']['n2']}]"
        
        if 'b2_dd' in results['mex']:
            # DD
            results_df.loc[54, 'D'] = format_val(results['mex']['b2_dd']['coef'])
            results_df.loc[55, 'D'] = f"({format_val(results['mex']['b2_dd']['se'])})"
        
        if 'b_ddd' in results['mex']:
            # DDD
            results_df.loc[56, 'D'] = format_val(results['mex']['b_ddd']['coef'])
            results_df.loc[57, 'D'] = f"({format_val(results['mex']['b_ddd']['se'])})"
        
        # Fill in Canada results for Panel A, A1
        if 'a1_nafta' in results['can']:
            # NAFTA partner
            results_df.loc[7, 'F'] = format_val(results['can']['a1_nafta']['mean1'])
            results_df.loc[7, 'G'] = format_val(results['can']['a1_nafta']['mean2'])
            results_df.loc[7, 'H'] = format_val(results['can']['a1_nafta']['diff'])
            
            results_df.loc[8, 'F'] = f"({format_val(results['can']['a1_nafta']['se1'])})"
            results_df.loc[8, 'G'] = f"({format_val(results['can']['a1_nafta']['se2'])})"
            results_df.loc[8, 'H'] = f"({format_val(results['can']['a1_nafta']['se_diff'])})"
            
            results_df.loc[9, 'F'] = f"[{results['can']['a1_nafta']['n1']}]"
            results_df.loc[9, 'G'] = f"[{results['can']['a1_nafta']['n2']}]"
        
        if 'a1_row' in results['can']:
            # ROW
            results_df.loc[10, 'F'] = format_val(results['can']['a1_row']['mean1'])
            results_df.loc[10, 'G'] = format_val(results['can']['a1_row']['mean2'])
            results_df.loc[10, 'H'] = format_val(results['can']['a1_row']['diff'])
            
            results_df.loc[11, 'F'] = f"({format_val(results['can']['a1_row']['se1'])})"
            results_df.loc[11, 'G'] = f"({format_val(results['can']['a1_row']['se2'])})"
            results_df.loc[11, 'H'] = f"({format_val(results['can']['a1_row']['se_diff'])})"
            
            results_df.loc[12, 'F'] = f"[{results['can']['a1_row']['n1']}]"
            results_df.loc[12, 'G'] = f"[{results['can']['a1_row']['n2']}]"
        
        if 'a1_dd' in results['can']:
            # DD
            results_df.loc[13, 'H'] = format_val(results['can']['a1_dd']['coef'])
            results_df.loc[14, 'H'] = f"({format_val(results['can']['a1_dd']['se'])})"
        
        # Fill in Canada results for Panel A, A2
        if 'a2_nafta' in results['can']:
            # NAFTA partner
            results_df.loc[18, 'F'] = format_val(results['can']['a2_nafta']['mean1'])
            results_df.loc[18, 'G'] = format_val(results['can']['a2_nafta']['mean2'])
            results_df.loc[18, 'H'] = format_val(results['can']['a2_nafta']['diff'])
            
            results_df.loc[19, 'F'] = f"({format_val(results['can']['a2_nafta']['se1'])})"
            results_df.loc[19, 'G'] = f"({format_val(results['can']['a2_nafta']['se2'])})"
            results_df.loc[19, 'H'] = f"({format_val(results['can']['a2_nafta']['se_diff'])})"
            
            results_df.loc[20, 'F'] = f"[{results['can']['a2_nafta']['n1']}]"
            results_df.loc[20, 'G'] = f"[{results['can']['a2_nafta']['n2']}]"
        
        if 'a2_row' in results['can']:
            # ROW
            results_df.loc[21, 'F'] = format_val(results['can']['a2_row']['mean1'])
            results_df.loc[21, 'G'] = format_val(results['can']['a2_row']['mean2'])
            results_df.loc[21, 'H'] = format_val(results['can']['a2_row']['diff'])
            
            results_df.loc[22, 'F'] = f"({format_val(results['can']['a2_row']['se1'])})"
            results_df.loc[22, 'G'] = f"({format_val(results['can']['a2_row']['se2'])})"
            results_df.loc[22, 'H'] = f"({format_val(results['can']['a2_row']['se_diff'])})"
            
            results_df.loc[23, 'F'] = f"[{results['can']['a2_row']['n1']}]"
            results_df.loc[23, 'G'] = f"[{results['can']['a2_row']['n2']}]"
        
        if 'a2_dd' in results['can']:
            # DD
            results_df.loc[24, 'H'] = format_val(results['can']['a2_dd']['coef'])
            results_df.loc[25, 'H'] = f"({format_val(results['can']['a2_dd']['se'])})"
        
        if 'a_ddd' in results['can']:
            # DDD
            results_df.loc[26, 'H'] = format_val(results['can']['a_ddd']['coef'])
            results_df.loc[27, 'H'] = f"({format_val(results['can']['a_ddd']['se'])})"
        
        # Fill in Canada results for Panel B, B1
        if 'b1_phase' in results['can']:
            # Phase-out products
            results_df.loc[37, 'F'] = format_val(results['can']['b1_phase']['mean1'])
            results_df.loc[37, 'G'] = format_val(results['can']['b1_phase']['mean2'])
            results_df.loc[37, 'H'] = format_val(results['can']['b1_phase']['diff'])
            
            results_df.loc[38, 'F'] = f"({format_val(results['can']['b1_phase']['se1'])})"
            results_df.loc[38, 'G'] = f"({format_val(results['can']['b1_phase']['se2'])})"
            results_df.loc[38, 'H'] = f"({format_val(results['can']['b1_phase']['se_diff'])})"
            
            results_df.loc[39, 'F'] = f"[{results['can']['b1_phase']['n1']}]"
            results_df.loc[39, 'G'] = f"[{results['can']['b1_phase']['n2']}]"
        
        if 'b1_cdf' in results['can']:
            # CDF products
            results_df.loc[40, 'F'] = format_val(results['can']['b1_cdf']['mean1'])
            results_df.loc[40, 'G'] = format_val(results['can']['b1_cdf']['mean2'])
            results_df.loc[40, 'H'] = format_val(results['can']['b1_cdf']['diff'])
            
            results_df.loc[41, 'F'] = f"({format_val(results['can']['b1_cdf']['se1'])})"
            results_df.loc[41, 'G'] = f"({format_val(results['can']['b1_cdf']['se2'])})"
            results_df.loc[41, 'H'] = f"({format_val(results['can']['b1_cdf']['se_diff'])})"
            
            results_df.loc[42, 'F'] = f"[{results['can']['b1_cdf']['n1']}]"
            results_df.loc[42, 'G'] = f"[{results['can']['b1_cdf']['n2']}]"
        
        if 'b1_dd' in results['can']:
            # DD
            results_df.loc[43, 'H'] = format_val(results['can']['b1_dd']['coef'])
            results_df.loc[44, 'H'] = f"({format_val(results['can']['b1_dd']['se'])})"
        
        # Fill in Canada results for Panel B, B2
        if 'b2_phase' in results['can']:
            # Phase-out products
            results_df.loc[48, 'F'] = format_val(results['can']['b2_phase']['mean1'])
            results_df.loc[48, 'G'] = format_val(results['can']['b2_phase']['mean2'])
            results_df.loc[48, 'H'] = format_val(results['can']['b2_phase']['diff'])
            
            results_df.loc[49, 'F'] = f"({format_val(results['can']['b2_phase']['se1'])})"
            results_df.loc[49, 'G'] = f"({format_val(results['can']['b2_phase']['se2'])})"
            results_df.loc[49, 'H'] = f"({format_val(results['can']['b2_phase']['se_diff'])})"
            
            results_df.loc[50, 'F'] = f"[{results['can']['b2_phase']['n1']}]"
            results_df.loc[50, 'G'] = f"[{results['can']['b2_phase']['n2']}]"
        
        if 'b2_cdf' in results['can']:
            # CDF products
            results_df.loc[51, 'F'] = format_val(results['can']['b2_cdf']['mean1'])
            results_df.loc[51, 'G'] = format_val(results['can']['b2_cdf']['mean2'])
            results_df.loc[51, 'H'] = format_val(results['can']['b2_cdf']['diff'])
            
            results_df.loc[52, 'F'] = f"({format_val(results['can']['b2_cdf']['se1'])})"
            results_df.loc[52, 'G'] = f"({format_val(results['can']['b2_cdf']['se2'])})"
            results_df.loc[52, 'H'] = f"({format_val(results['can']['b2_cdf']['se_diff'])})"
            
            results_df.loc[53, 'F'] = f"[{results['can']['b2_cdf']['n1']}]"
            results_df.loc[53, 'G'] = f"[{results['can']['b2_cdf']['n2']}]"
        
        if 'b2_dd' in results['can']:
            # DD
            results_df.loc[54, 'H'] = format_val(results['can']['b2_dd']['coef'])
            results_df.loc[55, 'H'] = f"({format_val(results['can']['b2_dd']['se'])})"
        
        if 'b_ddd' in results['can']:
            # DDD
            results_df.loc[56, 'H'] = format_val(results['can']['b_ddd']['coef'])
            results_df.loc[57, 'H'] = f"({format_val(results['can']['b_ddd']['se'])})"
        
        # Write results to Excel
        results_df.to_excel(writer, sheet_name=f'Table2Results{file_suffix}', index=False)
    
    writer.close()
    print(f"Table 2 created and saved to {os.path.join(dir_figures_tables, f'table_2_{file_suffix}.xlsx')}")
    return True

# Function to create Table A3
def create_table_a3():
    """Generate data for Table A3: Tariff cuts by staging category"""
    print("Creating Table A3...")
    
    # Load raw codes data
    df = pd.read_stata(os.path.join(dir_data, "dataRC.dta"))
    schedules = pd.read_stata(os.path.join(dir_data, "schedulesUS.dta"))
    
    # Merge datasets
    df = pd.merge(df, schedules, on='hs8', how='left')
    
    # Keep only one observation per HS8 code
    df = df.drop_duplicates(subset=['hs8'])
    
    # Process categories
    for z in ['catC', 'catEffC', 'catEffM']:
        df.loc[df[z].isin(["missing code (see notes)", "see notes"]), z] = "Missing"
        df.loc[df[z].isin(["Mixed phasein (no D)", "Mixed phasein (incl D)"]), z] = "Mixed"
    
    df.loc[df['catC'] == "3 EQUAL ANNUAL STAGES", 'catC'] = "Missing"
    
    # Add a counter column
    df['a'] = 1
    
    # Calculate means and counts for CUSFTA
    cusfta_agg = df.groupby('catC').agg({
        'baseCUSFTA': 'mean',
        'a': 'sum'
    }).reset_index()
    cusfta_agg.columns = ['staging', 'CUSFTAmean', 'CUSFTAobs']
    
    # Calculate means and counts for Canada NAFTA
    can_nafta_agg = df.groupby('catEffC').agg({
        'baseCanNAFTA': 'mean',
        'a': 'sum'
    }).reset_index()
    can_nafta_agg.columns = ['staging', 'CanNAFTAmean', 'CanNAFTAobs']
    
    # Calculate means and counts for Mexico NAFTA
    mex_nafta_agg = df.groupby('catEffM').agg({
        'baseMexNAFTA': 'mean',
        'a': 'sum'
    }).reset_index()
    mex_nafta_agg.columns = ['staging', 'MexNAFTAmean', 'MexNAFTAobs']
    
    # Combine all aggregates
    combined = pd.merge(cusfta_agg, can_nafta_agg, on='staging', how='outer')
    combined = pd.merge(combined, mex_nafta_agg, on='staging', how='outer')
    
    # Add years column for annual calculation
    combined['years'] = np.nan
    combined.loc[combined['staging'] == 'A', 'years'] = 1
    combined.loc[combined['staging'] == 'B', 'years'] = 5
    combined.loc[combined['staging'] == 'B6', 'years'] = 6
    combined.loc[combined['staging'] == 'C', 'years'] = 10
    combined.loc[combined['staging'] == 'C10', 'years'] = 10
    combined.loc[combined['staging'] == 'C+', 'years'] = 15
    
    # Set means to NaN for certain categories
    for col in ['CUSFTAmean', 'CanNAFTAmean', 'MexNAFTAmean']:
        combined.loc[combined['staging'].isin(['D', 'Mixed', 'Missing']), col] = np.nan
    
    # Set MexNAFTAmean to NaN for GSP
    combined.loc[combined['staging'] == 'GSP', 'MexNAFTAmean'] = np.nan
    
    # Calculate annual tariff cuts
    combined['CUSFTAannual'] = combined['CUSFTAmean'] / combined['years']
    combined['CanNAFTAannual'] = combined['CanNAFTAmean'] / combined['years']
    combined['MexNAFTAannual'] = combined['MexNAFTAmean'] / combined['years']
    
    # Drop empty rows and sort
    combined = combined[combined['staging'] != '']
    combined = combined.sort_values(['years', 'staging'])
    combined = combined.drop(columns=['years'])
    
    # Order columns
    combined = combined[['staging', 'CUSFTAobs', 'CUSFTAmean', 'CUSFTAannual', 
                         'CanNAFTAobs', 'CanNAFTAmean', 'CanNAFTAannual',
                         'MexNAFTAobs', 'MexNAFTAmean', 'MexNAFTAannual']]
    
    # Save to Excel
    combined.to_excel(os.path.join(dir_figures_tables, "table_A3.xlsx"), 
                     sheet_name="TableA3RC_data", index=False)

# Function to run DDD regressions
def run_ddd_regressions(country, dependent_var, file_suffix):
    """
    Run difference-in-difference-in-differences regressions using a memory-efficient approach
    
    Args:
        country: String, either 'Mex' or 'Can'
        dependent_var: String, variable to use as dependent variable (e.g., 'cv', 'uv', 'q')
        file_suffix: String, file suffix to use (e.g., 'CC' for consistent codes)
    """
    print(f"Running DDD regressions for {country}, dependent var: {dependent_var}, file: {file_suffix}")
    
    # Load the appropriate dataset - prefer pickle due to Stata compatibility issues
    try:
        df = pd.read_pickle(os.path.join(dir_data, f"data{file_suffix}_{country}.pkl"))
    except FileNotFoundError:
        try:
            # Fall back to Stata if pickle doesn't exist
            df = pd.read_stata(os.path.join(dir_data, f"data{file_suffix}_{country}.dta"))
        except Exception as e:
            print(f"Error loading data: {e}")
            # Try to see what files exist
            import glob
            available_files = glob.glob(os.path.join(dir_data, f"*{country}*"))
            print(f"Available files matching '{country}': {available_files}")
            return {"error": str(e)}
    
    # Print data info
    print(f"Data shape: {df.shape}")
    print(f"Available columns: {df.columns.tolist()}")
    
    # Create post-NAFTA dummy (1994 and after)
    df['post'] = (df['year'] >= 1994).astype(int)
    
    # Create interaction terms - ensure all are integers
    try:
        df['NAFTA'] = df['NAFTA'].astype(int)
        df['treatCDF'] = df['treatCDF'].astype(int)
        df['treatNonCDF'] = df['treatNonCDF'].astype(int)
        
        df['post_NAFTA'] = (df['post'] * df['NAFTA']).astype(int)
        df['post_treatCDF'] = (df['post'] * df['treatCDF']).astype(int)
        df['post_treatNonCDF'] = (df['post'] * df['treatNonCDF']).astype(int)
        df['NAFTA_treatCDF'] = (df['NAFTA'] * df['treatCDF']).astype(int)
        df['NAFTA_treatNonCDF'] = (df['NAFTA'] * df['treatNonCDF']).astype(int)
        df['post_NAFTA_treatCDF'] = (df['post'] * df['NAFTA'] * df['treatCDF']).astype(int)
        df['post_NAFTA_treatNonCDF'] = (df['post'] * df['NAFTA'] * df['treatNonCDF']).astype(int)
    except KeyError as e:
        print(f"Error creating interaction terms: {e}")
        print("Available columns:", df.columns.tolist())
        return {"error": f"Missing required columns: {e}"}
    
    # More specific treatment interactions for different treatment types
    treat_var_checks = []
    for treat in ['Immed', 'GSP', 'Phase5', 'Phase10']:
        if treat == 'GSP' and country == 'Can':
            continue  # Skip GSP for Canada since it doesn't apply
        
        try:
            treat_col = f'treat{treat}'
            if treat_col in df.columns:
                df[treat_col] = df[treat_col].astype(int)
                df[f'post_treat{treat}'] = (df['post'] * df[treat_col]).astype(int)
                df[f'NAFTA_treat{treat}'] = (df['NAFTA'] * df[treat_col]).astype(int)
                df[f'post_NAFTA_treat{treat}'] = (df['post'] * df['NAFTA'] * df[treat_col]).astype(int)
                treat_var_checks.append((treat, True))
            else:
                print(f"Column treat{treat} not found in dataset")
                treat_var_checks.append((treat, False))
        except Exception as e:
            print(f"Error creating treatment variable {treat}: {e}")
            treat_var_checks.append((treat, False))
    
    # Define dependent variable
    y_var = f"{country}_{dependent_var}"
    if y_var not in df.columns:
        print(f"Error: Dependent variable {y_var} not found in dataset")
        print("Available columns:", df.columns.tolist())
        return {"error": f"Dependent variable {y_var} not found"}
    
    # Clean data - remove missing values for dependent variable and remove infinities
    df_clean = df.dropna(subset=[y_var])
    # Also clean any infinities in the dependent variable
    df_clean = df_clean[~np.isinf(df_clean[y_var])]
    
    # Make sure dependent variable is float
    df_clean[y_var] = df_clean[y_var].astype(float)
    
    print(f"Clean data shape: {df_clean.shape}")
    
    # Check if we have enough data
    if len(df_clean) < 100:
        print("Warning: Very small sample size after cleaning")
        if len(df_clean) < 10:
            print("Error: Sample too small for regression")
            return {"error": "Sample too small for regression"}
    
    # Results container
    results = {}
    
    # Create entity and time fixed effects
    df_clean['entity_fe'] = df_clean['hs8'].astype(str) + "_" + df_clean['cntry'].astype(str)
    df_clean['time_fe'] = df_clean['year']
    
    # Approach: Use demeaning for high-dimensional fixed effects
    print("Using demeaning approach for high-dimensional fixed effects")
    
    # Variables to demean for Specification 1
    demean_vars1 = [y_var, 'post_NAFTA', 'post_treatCDF', 'post_treatNonCDF', 
                   'NAFTA_treatCDF', 'NAFTA_treatNonCDF', 
                   'post_NAFTA_treatCDF', 'post_NAFTA_treatNonCDF']
    
    # Create copy to avoid modifying original
    df_demean1 = df_clean[demean_vars1 + ['entity_fe', 'time_fe']].copy()
    
    try:
        # Step 1: Demean within entities (product-country fixed effects)
        for var in demean_vars1:
            entity_means = df_demean1.groupby('entity_fe')[var].transform('mean')
            df_demean1[f'{var}_entity_demean'] = df_demean1[var] - entity_means
        
        # Step 2: Demean within time periods (year fixed effects)
        entity_demeaned_vars = [f'{var}_entity_demean' for var in demean_vars1]
        for var in entity_demeaned_vars:
            time_means = df_demean1.groupby('time_fe')[var].transform('mean')
            df_demean1[f'{var}_time_demean'] = df_demean1[var] - time_means
        
        # Final demeaned variables
        y_demean1 = df_demean1[f'{y_var}_entity_demean_time_demean']
        X_demean_vars1 = [f'{var}_entity_demean_time_demean' for var in demean_vars1[1:]]  # Skip dependent var
        X_demean1 = df_demean1[X_demean_vars1]
        
        # Run regression on demeaned variables
        X_demean1 = sm.add_constant(X_demean1)
        model_demean1 = sm.OLS(y_demean1, X_demean1)
        results_demean1 = model_demean1.fit(cov_type='HC1')
        results['spec1'] = results_demean1
        
        # Extract key coefficients for Specification 1
        triple_cdf_col = None
        triple_noncdf_col = None
        
        for col in X_demean_vars1:
            if 'post_NAFTA_treatCDF' in col:
                triple_cdf_col = col
            elif 'post_NAFTA_treatNonCDF' in col:
                triple_noncdf_col = col
        
        # Extract Specification 1 effects
        results['spec1_effects'] = {}
        
        if triple_cdf_col and triple_cdf_col in results_demean1.params.index:
            results['spec1_effects']['CDF_effect'] = results_demean1.params[triple_cdf_col]
        else:
            results['spec1_effects']['CDF_effect'] = np.nan
            
        if triple_noncdf_col and triple_noncdf_col in results_demean1.params.index:
            results['spec1_effects']['NonCDF_effect'] = results_demean1.params[triple_noncdf_col]
        else:
            results['spec1_effects']['NonCDF_effect'] = np.nan
        
        # Calculate difference for Specification 1
        if not np.isnan(results['spec1_effects']['CDF_effect']) and not np.isnan(results['spec1_effects']['NonCDF_effect']):
            results['spec1_effects']['Diff_CDF_NonCDF'] = (
                results['spec1_effects']['CDF_effect'] - 
                results['spec1_effects']['NonCDF_effect']
            )
        else:
            results['spec1_effects']['Diff_CDF_NonCDF'] = np.nan
            
        # Print summary of Specification 1 results
        print("\nSpecification 1 (CDF vs non-CDF) Results:")
        print(results['spec1'].summary().tables[1])  # Print just the coefficient table
        
        print("\nKey Effects for Specification 1:")
        for effect, value in results['spec1_effects'].items():
            print(f"  {effect}: {value:.4f}")
            
        # Now run Specification 2 with specific treatment types
        treat_vars = []
        for treat, available in treat_var_checks:
            if available:
                treat_vars.append(treat)
        
        if treat_vars:
            # Create variables to demean for Specification 2
            demean_vars2 = [y_var]
            for treat in treat_vars:
                demean_vars2.extend([
                    f'post_treat{treat}', 
                    f'NAFTA_treat{treat}', 
                    f'post_NAFTA_treat{treat}'
                ])
            
            # Create copy for Specification 2
            df_demean2 = df_clean[demean_vars2 + ['entity_fe', 'time_fe']].copy()
            
            # Step 1: Demean within entities
            for var in demean_vars2:
                entity_means = df_demean2.groupby('entity_fe')[var].transform('mean')
                df_demean2[f'{var}_entity_demean'] = df_demean2[var] - entity_means
            
            # Step 2: Demean within time periods
            entity_demeaned_vars2 = [f'{var}_entity_demean' for var in demean_vars2]
            for var in entity_demeaned_vars2:
                time_means = df_demean2.groupby('time_fe')[var].transform('mean')
                df_demean2[f'{var}_time_demean'] = df_demean2[var] - time_means
            
            # Final demeaned variables for Specification 2
            y_demean2 = df_demean2[f'{y_var}_entity_demean_time_demean']
            X_demean_vars2 = [f'{var}_entity_demean_time_demean' for var in demean_vars2[1:]]  # Skip dependent var
            X_demean2 = df_demean2[X_demean_vars2]
            
            # Run regression on demeaned variables for Specification 2
            X_demean2 = sm.add_constant(X_demean2)
            model_demean2 = sm.OLS(y_demean2, X_demean2)
            results_demean2 = model_demean2.fit(cov_type='HC1')
            results['spec2'] = results_demean2
            
            # Extract Specification 2 effects
            results['spec2_effects'] = {}
            
            for treat in treat_vars:
                triple_term = None
                for col in X_demean_vars2:
                    if f'post_NAFTA_treat{treat}' in col:
                        triple_term = col
                        break
                
                if triple_term and triple_term in results_demean2.params.index:
                    results['spec2_effects'][f'{treat}_effect'] = results_demean2.params[triple_term]
                else:
                    results['spec2_effects'][f'{treat}_effect'] = np.nan
            
            # Print summary of Specification 2 results
            print("\nSpecification 2 (Treatment Types) Results:")
            print(results['spec2'].summary().tables[1])
            
            print("\nKey Effects for Specification 2:")
            for effect, value in results['spec2_effects'].items():
                print(f"  {effect}: {value:.4f}")
        else:
            print("No treatment variables available for Specification 2")
            
    except Exception as e:
        print(f"Error in demeaning approach: {e}")
        import traceback
        traceback.print_exc()
        
        # Fall back to basic regression without fixed effects
        print("\nFalling back to basic regression without fixed effects")
        
        try:
            # Basic regression for Specification 1 (only triple interaction terms)
            triple_vars1 = ['post_NAFTA_treatCDF', 'post_NAFTA_treatNonCDF']
            
            X1 = df_clean[triple_vars1].values
            X1 = sm.add_constant(X1)
            y1 = df_clean[y_var].values
            
            model_basic1 = sm.OLS(y1, X1)
            results_basic1 = model_basic1.fit(cov_type='HC1')
            results['spec1'] = results_basic1
            
            # Extract key coefficients for Specification 1
            results['spec1_effects'] = {
                'CDF_effect': results_basic1.params[1],  # post_NAFTA_treatCDF
                'NonCDF_effect': results_basic1.params[2],  # post_NAFTA_treatNonCDF
                'Diff_CDF_NonCDF': results_basic1.params[1] - results_basic1.params[2]
            }
            
            print("\nBasic Regression Results (Specification 1):")
            print(results['spec1'].summary().tables[1])
            
            print("\nKey Effects for Specification 1:")
            for effect, value in results['spec1_effects'].items():
                print(f"  {effect}: {value:.4f}")
            
            # Try Specification 2 if treatment variables are available
            treat_vars = []
            for treat, available in treat_var_checks:
                if available:
                    treat_vars.append(treat)
            
            if treat_vars:
                # Create triple interaction terms for each treatment type
                triple_vars2 = []
                for treat in treat_vars:
                    term = f'post_NAFTA_treat{treat}'
                    if term in df_clean.columns:
                        triple_vars2.append(term)
                
                if triple_vars2:
                    # Basic regression for Specification 2
                    X2 = df_clean[triple_vars2].values
                    X2 = sm.add_constant(X2)
                    y2 = df_clean[y_var].values
                    
                    model_basic2 = sm.OLS(y2, X2)
                    results_basic2 = model_basic2.fit(cov_type='HC1')
                    results['spec2'] = results_basic2
                    
                    # Extract effects for Specification 2
                    results['spec2_effects'] = {}
                    
                    # Initialize with NaN for all treatment types
                    for treat in treat_vars:
                        results['spec2_effects'][f'{treat}_effect'] = np.nan
                        
                    # Fill in values from regression
                    for i, col in enumerate(triple_vars2):
                        for treat in treat_vars:
                            if f'post_NAFTA_treat{treat}' in col:
                                results['spec2_effects'][f'{treat}_effect'] = results_basic2.params[i+1]  # +1 for constant
                                break
                    
                    print("\nBasic Regression Results (Specification 2):")
                    print(results['spec2'].summary().tables[1])
                    
                    print("\nKey Effects for Specification 2:")
                    for effect, value in results['spec2_effects'].items():
                        print(f"  {effect}: {value:.4f}")
                else:
                    print("No triple interaction terms available for Specification 2")
            else:
                print("No treatment variables available for Specification 2")
        
        except Exception as e:
            print(f"Error in basic regression approach: {e}")
            results['error'] = str(e)
    
    # Summary of results
    print("\nSummary of Results:")
    if 'spec1_effects' in results:
        print("Specification 1 (CDF vs non-CDF):")
        for effect, value in results['spec1_effects'].items():
            print(f"  {effect}: {value:.4f}")
    
    if 'spec2_effects' in results:
        print("Specification 2 (Treatment Types):")
        for effect, value in results['spec2_effects'].items():
            print(f"  {effect}: {value:.4f}")
    
    return results

# Main function to run all analyses
def run_all_analyses():
    """Run all analyses in the correct order"""
    # Create directories if they don't exist
    for directory in [dir_data, dir_estimates, dir_figures_tables, dir_appendix]:
        os.makedirs(directory, exist_ok=True)
    
    try:    
        # Prepare data for DDD analyses
        for file_type in ['rawCodes', 'consistentCodes']:
            try:
                prepare_ddd_data(file_type)
            except Exception as e:
                print(f"Error preparing {file_type} data: {e}")
                print("Continuing with next steps...")
        
        # Create Table A2 and A3 (these must be run first according to the Stata script)
        try:
            create_table_a2()
        except Exception as e:
            print(f"Error creating Table A2: {e}")
            print("Continuing with next steps...")
            
        try:
            create_table_a3()
        except Exception as e:
            print(f"Error creating Table A3: {e}")
            print("Continuing with next steps...")
        
        # Create Table 2
        try:
            create_table_2()
        except Exception as e:
            print(f"Error creating Table 2: {e}")
            print("Continuing with next steps...")
        
        # Run regressions
        # Format: country, dependent variable, file suffix
        regression_configs = [
            ('Mex', 'cv', 'CC'),
            ('Mex', 'uv', 'CC'),
            ('Mex', 'q', 'CC'),
            ('Can', 'cv', 'CC'),
            ('Can', 'uv', 'CC'),
            ('Can', 'q', 'CC'),
        ]
        
        regression_results = {}
        for config in regression_configs:
            try:
                key = f"{config[0]}_{config[1]}_{config[2]}"
                regression_results[key] = run_ddd_regressions(*config)
                print(f"Successfully ran regression for {key}")
            except Exception as e:
                print(f"Error running regression for {config}: {e}")
                print("Continuing with next configurations...")
        
        # Print summary of all regression results
        print("\nSummary of All Regression Results:")
        for key, result in regression_results.items():
            print(f"\n{key}:")
            if 'spec1_effects' in result:
                print("  Specification 1 (CDF vs non-CDF):")
                for effect, value in result['spec1_effects'].items():
                    print(f"    {effect}: {value:.4f}")
            
            if 'spec2_effects' in result:
                print("  Specification 2 (Treatment Types):")
                for effect, value in result['spec2_effects'].items():
                    print(f"    {effect}: {value:.4f}")
        
        # Save regression results to file
        if regression_results:
            import pickle
            with open(os.path.join(dir_estimates, 'regression_results.pkl'), 'wb') as f:
                pickle.dump(regression_results, f)
            
            print(f"\nRegression results saved to {os.path.join(dir_estimates, 'regression_results.pkl')}")
        else:
            print("\nNo regression results to save")
        
        return regression_results
    
    except Exception as e:
        print(f"Error in run_all_analyses: {e}")
        return {}

# If this script is run directly, execute all analyses
if __name__ == "__main__":
    run_all_analyses()

