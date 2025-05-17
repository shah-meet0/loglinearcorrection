import numpy as np
import pickle
import os
from rf_estimation_SZ import rf_estimation_SZ

# rf_incidence.py
# Estimates incidence and weighted incidence for alternative models of
# profit incidence, as well as taking simple averages and weighted averages

def run_rf_incidence(beta_Mod1=None, cov_Mod1=None, data_dict=None):
    """
    Run reduced form incidence estimation
    
    Args:
        beta_Mod1: coefficient estimates for base model (optional if data_dict provided)
        cov_Mod1: covariance matrix for base model (optional if data_dict provided)
        data_dict: dictionary containing all data (optional)
    
    Returns:
        Dictionary containing results for table formatting
    """
    # Load data if not provided directly
    if beta_Mod1 is None or cov_Mod1 is None:
        if data_dict is None:
            try:
                with open(os.path.join(os.getcwd(), 'estimates', 'RFmoments.pkl'), 'rb') as f:
                    data_dict = pickle.load(f)
            except FileNotFoundError:
                print("Warning: RFmoments.pkl not found. Using dummy data for demonstration.")
                data_dict = {
                    'beta_Mod1': np.array([0.5, 0.3, 0.2, 0.1]),
                    'cov_Mod1': np.eye(4) * 0.01
                }
        
        beta_Mod1 = data_dict['beta_Mod1']
        cov_Mod1 = data_dict['cov_Mod1']
    
    # (1) Data Base Model
    # Order: N, W, R, E, l, z
    beta = beta_Mod1
    VC = cov_Mod1
    
    # For case of epd = -2.5
    # (2) Set calibrated moments and types
    cal = [0.15, 0.30, 0.135, -2.5]
    type_val = 1
    
    # (3) Run Model and Collect Output
    INC, INC_VC, SH, SH_VC, WSH, WSH_VC, pVAL, _ = rf_estimation_SZ(beta, VC, cal, type_val)
    
    INC_SE = np.sqrt(np.diag(INC_VC))
    SH_SE = np.sqrt(np.diag(SH_VC))
    WSH_SE = np.sqrt(np.diag(WSH_VC))
    
    # Format output for table
    C3_P1 = np.array([cal[0], cal[1], cal[3]]).reshape(-1, 1)
    
    # Create output arrays with interleaved estimates and standard errors
    C3_P3 = np.full(2 * len(INC), np.nan)
    C3_P4 = np.full(2 * len(SH), np.nan)
    C3_P5 = np.full(2 * len(WSH), np.nan)
    
    # Fill arrays with estimates and standard errors in alternating positions
    # Reverse order for INC, SH, WSH: [3,2,1] corresponds to [2,1,0] in Python
    C3_P3[::2] = INC[[2, 1, 0]]
    C3_P3[1::2] = INC_SE[[2, 1, 0]]
    C3_P4[::2] = SH[[2, 1, 0]]
    C3_P4[1::2] = SH_SE[[2, 1, 0]]
    C3_P5[::2] = WSH[[2, 1, 0]]
    C3_P5[1::2] = WSH_SE[[2, 1, 0]]
    
    C3_conv = pVAL[0]  # First p-value from conventional view test
    
    # Return results for use in table2.py
    return {
        'C3_P1': C3_P1,
        'C3_P3': C3_P3,
        'C3_P4': C3_P4,
        'C3_P5': C3_P5,
        'C3_conv': C3_conv
    }

# If run as a script, execute with dummy data
if __name__ == "__main__":
    results = run_rf_incidence()
    print("RF Incidence results:", results)
