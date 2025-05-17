import numpy as np
import pandas as pd
import pickle
from numdifftools import Jacobian
import os
from rf_estimation_SZ import rf_estimation_SZ
from estimation_SZ import estimation_SZ
from exportLatexTableSZ2 import exportLatexTableSZ2

# table1.py
# Plot firm incidence across values of ePD for different approaches

def table1(data_dict=None):
    """
    Generate Table 1 showing firm incidence across different approaches
    
    Args:
        data_dict: Dictionary containing data or None to load from file
    """
    
    # Setup - load data if not provided
    if data_dict is None:
        try:
            with open(os.path.join(os.getcwd(), 'estimates', 'RFmoments.pkl'), 'rb') as f:
                data_dict = pickle.load(f)
        except FileNotFoundError:
            print("Warning: RFmoments.pkl not found. Using dummy data for demonstration.")
            data_dict = {
                'beta_Mod1': np.array([0.5, 0.3, 0.2, 0.1]),
                'cov_Mod1': np.eye(4) * 0.01,
                'beta_Mod2': np.random.randn(8),
                'cov_Mod2': np.eye(8) * 0.01,
                'beta_Mod3': np.random.randn(12),
                'cov_Mod3': np.eye(12) * 0.01
            }
    
    # Extract data
    beta_Mod1 = data_dict['beta_Mod1']
    cov_Mod1 = data_dict['cov_Mod1']
    beta_Mod2 = data_dict['beta_Mod2']
    cov_Mod2 = data_dict['cov_Mod2']
    beta_Mod3 = data_dict['beta_Mod3']
    cov_Mod3 = data_dict['cov_Mod3']
    
    # Reduced form incidences
    cal = [0.15, 0.30, 0.135, -2.5]
    
    # Baseline model
    beta = beta_Mod1
    VC = cov_Mod1
    type_list = [1]
    
    # Matrices to collect outputs
    OS = []
    OVC = []
    
    # Run Models and Collect Outputs
    for i in range(len(type_list)):
        _, _, SH, SH_VC, _, _, _, _ = rf_estimation_SZ(beta, VC, cal, type_list[i])
        OS.append(SH[0])
        OVC.append(np.sqrt(SH_VC[0, 0]))
    
    # Model with Bartik controls
    beta = beta_Mod2[:4]  # First 4 elements
    VC = cov_Mod2[:4, :4]
    type_val = 1
    
    # Run Models and Collect Outputs
    _, _, SH, SH_VC, _, _, _, _ = rf_estimation_SZ(beta, VC, cal, type_val)
    OS.append(SH[0])
    OVC.append(np.sqrt(SH_VC[0, 0]))
    
    # Structural versions
    # Reshape for 4 moments
    beta = beta_Mod3
    VC = cov_Mod3
    
    # Calibrated parameters: (gamma, alpha, delta, eta^PD, phi/rho)
    cal_4mom = [0.15, 0.30, 0.135, -2.5, 1]
    
    BOUNDS1 = np.array([[0, 0, 0, -5, -5, -5],
                        [10, 10, 5, 5, 0, 5]])
    
    # Estimate 4 Moment Model
    res_col1, _, vc_col1, _ = estimation_SZ(beta, VC, cal_4mom, BOUNDS1, 4)
    
    # Calculate firm share
    OS2 = frm_incS_rho(res_col1[::2, 0], cal_4mom, 1)
    
    jac_1 = Jacobian(lambda f: frm_incS_rho(f, cal_4mom, 1))(res_col1[::2, 0])
    OVC2 = np.sqrt(jac_1 @ vc_col1 @ jac_1.T)
    
    # Rearrange for table with hard-coded values
    # hard coding columns (1) and (5) from SZ 2016
    sz2016 = np.array([[0.365, 0.420], [0.168, 0.120]])
    # hard coding columns (6) and (7) from SZ 2023
    sz2023 = np.array([[0.619, 0.523], [0.108, 0.337]])
    
    OS_final = [sz2016[0, 0], OS2] + OS + [sz2016[0, 1]] + list(sz2023[0, :])
    OVC_final = [sz2016[1, 0], OVC2] + OVC + [sz2016[1, 1]] + list(sz2023[1, :])

    # Export table
    OVC_final[1] = float(OVC_final[1].item())  # Extract the scalar value from the array
    data = np.array([np.array(OS_final) * 100, np.array(OVC_final) * 100])
    # Row names
    rows_A = ['Share', 'S.E.']
    
    # Export table
    os.makedirs(os.path.join(os.getcwd(), 'tables'), exist_ok=True)
    exportLatexTableSZ2(os.path.join(os.getcwd(), 'tables', 'table1'), 
                        data, rows_A, [1], 1, [], [], [], [], [], [], 0)
    
    print("Table 1 generated successfully!")
    return data

# Include the necessary functions from struct_incidence.py
def tot_inc_rho(res, cal, rho):
    return r_inc_rho(res, cal, rho) + wrk_inc_rho(res, cal, rho) + frm_inc_rho(res, cal, rho)

def w_pred_rho(res, cal, rho):
    return ((cal[2] * rho) / res[0] - 1 / (res[0] * (cal[3] + 1)) - 1) / (eLS(res, cal) - eLD(res, cal))

def r_inc_rho(res, cal, rho):
    return w_pred_rho(res, cal, rho) * (1 + eLS(res, cal)) / (1 + res[2])

def r_incS_rho(res, cal, rho):
    return r_inc_rho(res, cal, rho) / tot_inc_rho(res, cal, rho)

def wrk_inc_rho(res, cal, rho):
    return w_pred_rho(res, cal, rho) - cal[1] * r_inc_rho(res, cal, rho)

def wrk_incS_rho(res, cal, rho):
    return wrk_inc_rho(res, cal, rho) / tot_inc_rho(res, cal, rho)

def frm_inc_rho(res, cal, rho):
    return 1 - cal[2] * rho * (cal[3] + 1) + cal[0] * (cal[3] + 1) * w_pred_rho(res, cal, rho)

def frm_incS_rho(res, cal, rho):
    return frm_inc_rho(res, cal, rho) / tot_inc_rho(res, cal, rho)

def eLS(res, cal):
    """epsilon LS"""
    return (1 + res[2] - cal[1]) / (res[1] * (1 + res[2]) + cal[1])

def eLD(res, cal):
    """epsilon LD"""
    return -(cal[0] / res[0]) - 1

if __name__ == "__main__":
    table1()
