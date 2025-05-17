import numpy as np
import pandas as pd
import pickle
import os
import sys
from rf_incidence import run_rf_incidence
from struct_incidence import run_struct_incidence
from exportLatexTableSZ2 import exportLatexTableSZ2

# table2.py
# This program calculates the best fitting parameters (based on a method of
# moment estimation) of model specified in estimation_SZ

def table2(data_dict=None):
    """
    Generate Table 2 showing estimation results and incidence calculations
    
    Args:
        data_dict: Dictionary containing data or None to load from file
    """
    
    # Setup
    sys.path.append(os.path.join(os.getcwd(), 'toolbox'))
    sys.path.append(os.path.join(os.getcwd(), 'programs'))
    
    # Load data if not provided
    if data_dict is None:
        try:
            with open(os.path.join(os.getcwd(), 'estimates', 'RFmoments.pkl'), 'rb') as f:
                data_dict = pickle.load(f)
        except FileNotFoundError:
            print("Warning: RFmoments.pkl not found. Using dummy data for demonstration.")
            data_dict = {
                'beta_Mod1': np.array([0.5, 0.3, 0.2, 0.1]),
                'cov_Mod1': np.eye(4) * 0.01,
                'beta_Mod3': np.random.randn(12),
                'cov_Mod3': np.eye(12) * 0.01
            }
    
    # Extract data
    beta_Mod1 = data_dict['beta_Mod1']
    cov_Mod1 = data_dict['cov_Mod1']
    beta_Mod3 = data_dict['beta_Mod3']
    cov_Mod3 = data_dict['cov_Mod3']
    
    # Estimate reduced form incidence
    # Reproduce MMM-S estimates
    rf_results = run_rf_incidence(beta_Mod1, cov_Mod1)
    
    # Estimate structural model and incidence
    # New structural estimates
    struct_results = run_struct_incidence(beta_Mod3, cov_Mod3)
    
    # Clean up for table
    C2_P1 = np.array([struct_results['cal_4mom'][0], 
                      struct_results['cal_4mom'][1], 
                      struct_results['cal_4mom'][3]]).reshape(-1, 1)
    
    C2_P2 = np.vstack([
        struct_results['res_col1'][:6, :],
        struct_results['eLS_col1'].reshape(-1, 1),
        struct_results['eLD_col1'].reshape(-1, 1)
    ])
    
    C2_P3 = np.vstack([
        struct_results['wdot_col1'].reshape(-1, 1),
        struct_results['rdot_col1'].reshape(-1, 1),
        struct_results['wkdot_col1'].reshape(-1, 1),
        struct_results['pidot_col1'].reshape(-1, 1)
    ])
    
    C2_P4 = np.vstack([
        struct_results['rshare_col1'].reshape(-1, 1),
        struct_results['wkshare_col1'].reshape(-1, 1),
        struct_results['pishare_col1'].reshape(-1, 1)
    ])
    
    C2_P5 = np.vstack([
        struct_results['rshareWGT_col1'].reshape(-1, 1),
        struct_results['wkshareWGT_col1'].reshape(-1, 1),
        struct_results['pishareWGT_col1'].reshape(-1, 1)
    ])
    
    C2_conv = struct_results['conv_view_col1']
    C2_convWGT = struct_results['conv_viewWGT_col1']
    
    # Extract RF results
    C3_P1 = rf_results['C3_P1']
    C3_P3 = rf_results['C3_P3']
    C3_P4 = rf_results['C3_P4']
    C3_P5 = rf_results['C3_P5']
    C3_conv = rf_results['C3_conv']
    
    # Hardcode from SZ 2016
    C1_P1 = np.array([0.15, 0.3, -2.5]).reshape(-1, 1)
    C1_P2 = np.array([0.277, 0.138,
                      0.829, 0.282,
                      0.513, 1.417,
                      0.780, 0.386,
                      -1.766, 0.269]).reshape(-1, 1)
    C1_P3 = np.array([0.944, 0.408,
                      1.111, 1.119,
                      0.611, 0.293,
                      0.990, 0.092]).reshape(-1, 1)
    C1_P4 = np.array([0.410, 0.263,
                      0.225, 0.134,
                      0.365, 0.168]).reshape(-1, 1)
    C1_P5 = np.full((6, 1), np.nan)
    C1_conv = 0
    
    # Format and export table
    P1 = np.hstack([C1_P1, C2_P1, C3_P1])
    P2 = np.hstack([C1_P2, C2_P2, np.full((10, 1), np.nan)])
    
    # Handle P3 with proper dimensions
    C3_P3_reshaped = C3_P3[4:].reshape(-1, 1)  # Skip first 4 elements
    
    # Replace with this code to ensure all arrays have 8 rows
    P3 = np.hstack([C1_P3, C2_P3, 
                    np.vstack([
                        np.full((2, 1), np.nan),  # First 2 rows of NaN
                        C3_P3_reshaped,           # Your reshaped data
                        np.full((8 - 2 - len(C3_P3_reshaped), 1), np.nan)  # Fill remaining rows
                    ])])
    # Handle P4 with proper dimensions
    C3_P4_reshaped = C3_P4.reshape(-1, 1)
    P4 = np.hstack([C1_P4, C2_P4, C3_P4_reshaped])
    
    # Handle P5 (weighted shares - RF doesn't have these, so use NaN)
    P5 = np.hstack([C1_P5, C2_P5, np.full((6, 1), np.nan)])
    
    C1_conv_scalar = float(C1_conv) if hasattr(C1_conv, 'item') else float(C1_conv)
    C2_conv_scalar = float(C2_conv.item()) if hasattr(C2_conv, 'item') else float(C2_conv)
    C3_conv_scalar = float(C3_conv.item()) if hasattr(C3_conv, 'item') else float(C3_conv)

    # Use scalars to create the array
    CONV = np.array([[C1_conv_scalar, C2_conv_scalar, C3_conv_scalar]])

    
    # Table labels
    column_names = [
        ['SZ Table 5 col.1', 'Structural', 'Calibrating'],
        ['', 'Estimation', 'Product Demand']
    ]
    
    columns = [
        ['Intensive Margin', '', 'Calibrating', 'Average', 'Weighted Avg.',
         'Calibrating', 'Average', 'Weighted Avg.'],
        ['Labor Demand', 'TFP', 'Product Demand', 'of (1),(2),(3)',
         'of (1),(2),(3)', 'Product Demand', 'of (1),(2),(6)', 'of (1),(2),(6)']
    ]
    
    panel_names = [
        '\\textit{Panel A. Calibrated Parameters}',
        '\\textit{Panel B. Estimated Parameters}',
        '\\textit{Panel C. Incidence}',
        '\\textit{Panel D. Share of Incidence}',
        '\\textit{Panel E. Weighted Share of Incidence}',
        ''
    ]
    
    # Row names
    rows_A = [
        '\\hspace{1em}Output elasticity $\\gamma$',
        '\\hspace{1em}Housing share $\\alpha$',
        '\\hspace{1em}Elasticity of product demand $\\varepsilon^{PD}$'
    ]
    
    rows_B = [
        '\\hspace{1em}Idiosyncratic location', '\\hspace{2em}productivity dispersion $\\sigma^F$',
        '\\hspace{1em}Idiosyncratic location', '\\hspace{2em}preference dispersion $\\sigma^W$',
        '\\hspace{1em}Elasticity of housing', '\\hspace{2em}supply $\\eta$',
        '\\hspace{1em}Elasticity of labor supply $\\varepsilon^{LS}$',
        '',
        '\\hspace{1em}Elasticity of labor demand $\\varepsilon^{LD}$',
        ''
    ]
    
    rows_C = [
        '\\hspace{1em}Wages $\\tilde{w}$',
        '',
        '\\hspace{1em}Landowners $\\tilde{r}$',
        '',
        '\\hspace{1em}Workers $\\tilde{w} - \\alpha\\tilde{r}$',
        '',
        '\\hspace{1em}Firm owners $\\tilde{\\pi}$', ''
    ]
    
    rows_D = [
        '\\hspace{1em}Landowners $\\tilde{r}$',
        '',
        '\\hspace{1em}Workers $\\tilde{w} - \\alpha\\tilde{r}$',
        '',
        '\\hspace{1em}Firm owners $\\tilde{\\pi}$',
        ''
    ]
    
    rows_E = [
        '\\hspace{1em}Landowners $\\tilde{r}$',
        '',
        '\\hspace{1em}Workers $\\tilde{w} - \\alpha\\tilde{r}$',
        '',
        '\\hspace{1em}Firm owners $\\tilde{\\pi}$',
        ''
    ]
    
    rows_F = ['Test of standard view ($p$-value)']
    
    # Put everything together
    panel_ind = np.array([len(rows_A), len(rows_B), len(rows_C), 
                          len(rows_D), len(rows_E), len(rows_F)])
    
    data_entry = np.vstack([P1, P2, P3, P4, P5, CONV])
    rows = rows_A + rows_B + rows_C + rows_D + rows_E + rows_F
    
    # Decimal places (3 for most, 2 for parameters)
    dp_matrix = np.full(data_entry.shape, 3, dtype=int)
    dp_matrix[:3, :] = 2  # 2 decimal places for calibrated parameters
    
    # Export table
    os.makedirs(os.path.join(os.getcwd(), 'tables'), exist_ok=True)
    exportLatexTableSZ2(os.path.join(os.getcwd(), 'tables', 'table2'), 
                        data_entry, rows,
                        [0, 1, 1, 1, 1, 0], dp_matrix, column_names,
                        panel_names, panel_ind, [], [], [], 1)
    
    print("Table 2 generated successfully!")
    return data_entry

if __name__ == "__main__":
    table2()

