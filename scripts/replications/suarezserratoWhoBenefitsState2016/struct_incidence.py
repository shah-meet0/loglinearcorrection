import numpy as np
import pickle
import os
from scipy.linalg import inv
from scipy.stats import chi2
from numdifftools import Jacobian
from estimation_SZ import estimation_SZ

# struct_incidence.py
# Estimate 4 moment model, phi=1: need to call within script that loads workspace

def run_struct_incidence(beta_Mod3=None, cov_Mod3=None, data_dict=None):
    """
    Run structural incidence estimation
    
    Args:
        beta_Mod3: coefficient estimates for full model (optional if data_dict provided)
        cov_Mod3: covariance matrix for full model (optional if data_dict provided)
        data_dict: dictionary containing all data (optional)
    
    Returns:
        Dictionary containing results for table formatting
    """
    # Load data if not provided directly
    if beta_Mod3 is None or cov_Mod3 is None:
        if data_dict is None:
            try:
                with open(os.path.join(os.getcwd(), 'estimates', 'RFmoments.pkl'), 'rb') as f:
                    data_dict = pickle.load(f)
            except FileNotFoundError:
                print("Warning: RFmoments.pkl not found. Using dummy data for demonstration.")
                data_dict = {
                    'beta_Mod3': np.random.randn(12),
                    'cov_Mod3': np.eye(12) * 0.01
                }
        
        beta_Mod3 = data_dict['beta_Mod3']
        cov_Mod3 = data_dict['cov_Mod3']
    
    # Reshape for 4 moments
    beta = beta_Mod3
    VC = cov_Mod3
    
    # Calibrated parameters: (gamma, alpha, delta, eta^PD, phi/rho)
    cal_4mom = [0.15, 0.30, 0.135, -2.5, 1]
    
    BOUNDS1 = np.array([[0, 0, 0, -5, -5, -5],
                        [10, 10, 5, 5, 0, 5]])
    
    # Estimate 4 Moment Model
    res_col1, predM_col1, vc_col1, _ = estimation_SZ(beta, VC, cal_4mom, BOUNDS1, 4)
    
    # Helper function to calculate results with standard errors
    def calculate_with_se(func, res, vc, *args):
    
        value = func(res[::2, 0], *args)
        jac = Jacobian(lambda x: func(x, *args))(res[::2, 0])
        se_matrix = jac @ vc @ jac.T
        
        # Extract scalar values
        if hasattr(value, 'item'):
            value = value.item()  # Convert single-element array to scalar
        
        if hasattr(se_matrix, 'item'):
            se = float(np.sqrt(se_matrix).item())  # Convert to scalar
        else:
            se = float(np.sqrt(se_matrix))
            
        return np.array([value, se])
    # Labor Demand
    eLD_col1 = calculate_with_se(eLD, res_col1, vc_col1, cal_4mom)
    
    # Labor Supply
    eLS_col1 = calculate_with_se(eLS, res_col1, vc_col1, cal_4mom)
    
    # Incidence Calculations
    # Wage incidence
    wdot_col1 = calculate_with_se(w_pred_rho, res_col1, vc_col1, cal_4mom, 1)
    
    # landowner incidence and shares
    rdot_col1 = calculate_with_se(r_inc_rho, res_col1, vc_col1, cal_4mom, 1)
    rshare_col1 = calculate_with_se(r_incS_rho, res_col1, vc_col1, cal_4mom, 1)
    rshareWGT_col1 = calculate_with_se(r_incS_wgt_rho, res_col1, vc_col1, cal_4mom, 1)
    
    # worker incidence and shares
    wkdot_col1 = calculate_with_se(wrk_inc_rho, res_col1, vc_col1, cal_4mom, 1)
    wkshare_col1 = calculate_with_se(wrk_incS_rho, res_col1, vc_col1, cal_4mom, 1)
    wkshareWGT_col1 = calculate_with_se(wrk_incS_wgt_rho, res_col1, vc_col1, cal_4mom, 1)
    
    # firm incidence and shares
    pidot_col1 = calculate_with_se(frm_inc_rho, res_col1, vc_col1, cal_4mom, 1)
    pishare_col1 = calculate_with_se(frm_incS_rho, res_col1, vc_col1, cal_4mom, 1)
    pishareWGT_col1 = calculate_with_se(frm_incS_wgt_rho, res_col1, vc_col1, cal_4mom, 1)
    
    # Test of standard view
    L = np.eye(2)
    c = np.array([[1], [0]])
    
    # Need variance of [wkshare, pishare]
    b_hat = conv_view_rho(res_col1[::2, 0], cal_4mom, 1)
    jac_1 = Jacobian(lambda f: conv_view_rho(f, cal_4mom, 1))(res_col1[::2, 0])
    vc_hat = jac_1 @ vc_col1 @ jac_1.T
    
    qF = (L @ b_hat.reshape(-1, 1) - c).T @ inv(L @ vc_hat @ L.T) @ (L @ b_hat.reshape(-1, 1) - c)
    conv_view_col1 = 1 - chi2.cdf(qF, len(c))
    
    # Weighted version
    b_hat = conv_viewWGT_rho(res_col1[::2, 0], cal_4mom, 1)
    jac_1 = Jacobian(lambda f: conv_viewWGT_rho(f, cal_4mom, 1))(res_col1[::2, 0])
    vc_hat = jac_1 @ vc_col1 @ jac_1.T
    
    qF = (L @ b_hat.reshape(-1, 1) - c).T @ inv(L @ vc_hat @ L.T) @ (L @ b_hat.reshape(-1, 1) - c)
    conv_viewWGT_col1 = 1 - chi2.cdf(qF, len(c))
    
    # Return results for use in table2.py
    return {
        'cal_4mom': cal_4mom,
        'res_col1': res_col1,
        'eLS_col1': eLS_col1,
        'eLD_col1': eLD_col1,
        'wdot_col1': wdot_col1,
        'rdot_col1': rdot_col1,
        'wkdot_col1': wkdot_col1,
        'pidot_col1': pidot_col1,
        'rshare_col1': rshare_col1,
        'wkshare_col1': wkshare_col1,
        'pishare_col1': pishare_col1,
        'rshareWGT_col1': rshareWGT_col1,
        'wkshareWGT_col1': wkshareWGT_col1,
        'pishareWGT_col1': pishareWGT_col1,
        'conv_view_col1': conv_view_col1,
        'conv_viewWGT_col1': conv_viewWGT_col1
    }

## FUNCTIONS

# Conventional view test
def conv_view(res, cal):
    wshare = wrk_incS(res, cal)
    fshare = frm_incS(res, cal)
    return np.array([wshare, fshare])

def conv_viewWGT(res, cal):
    wshare = wrk_incS_wgt(res, cal)
    fshare = frm_incS_wgt(res, cal)
    return np.array([wshare, fshare])

def conv_view_rho(res, cal, rho):
    wshare = wrk_incS_rho(res, cal, rho)
    fshare = frm_incS_rho(res, cal, rho)
    return np.array([wshare, fshare])

def conv_viewWGT_rho(res, cal, rho):
    wshare = wrk_incS_wgt_rho(res, cal, rho)
    fshare = frm_incS_wgt_rho(res, cal, rho)
    return np.array([wshare, fshare])

# income share weights
def inc_share_f(cal):
    """firm share"""
    return (1 - cal[0] * (cal[3] + 1)) / (-1 * (cal[3] + 1) * (1 - cal[0]))

def w_weight(cal):
    """w weight"""
    return 1 / (1 + cal[1] + inc_share_f(cal))

def l_weight(cal):
    """l weight"""
    return cal[1] / (1 + cal[1] + inc_share_f(cal))

def f_weight(cal):
    """f weight"""
    return inc_share_f(cal) / (1 + cal[1] + inc_share_f(cal))

# total incidence
def tot_inc(res, cal):
    return r_inc(res, cal) + wrk_inc(res, cal) + frm_inc(res, cal)

def tot_inc_rho(res, cal, rho):
    return r_inc_rho(res, cal, rho) + wrk_inc_rho(res, cal, rho) + frm_inc_rho(res, cal, rho)

# Income weighted
def tot_inc_wgt(res, cal):
    return l_weight(cal) * r_inc(res, cal) + w_weight(cal) * wrk_inc(res, cal) + f_weight(cal) * frm_inc(res, cal)

def tot_inc_wgt_rho(res, cal, rho):
    return l_weight(cal) * r_inc_rho(res, cal, rho) + w_weight(cal) * wrk_inc_rho(res, cal, rho) + f_weight(cal) * frm_inc_rho(res, cal, rho)

# wage prediction
def w_pred(res, cal):
    return ((cal[2] * res[3]) / res[0] - 1 / (res[0] * (cal[3] + 1)) - 1) / (eLS(res, cal) - eLD(res, cal))

def w_pred_rho(res, cal, rho):
    return ((cal[2] * rho) / res[0] - 1 / (res[0] * (cal[3] + 1)) - 1) / (eLS(res, cal) - eLD(res, cal))

# landowner incidence
def r_inc(res, cal):
    return w_pred(res, cal) * (1 + eLS(res, cal)) / (1 + res[2])

def r_inc_rho(res, cal, rho):
    return w_pred_rho(res, cal, rho) * (1 + eLS(res, cal)) / (1 + res[2])

# incidence share
def r_incS(res, cal):
    return r_inc(res, cal) / tot_inc(res, cal)

def r_incS_rho(res, cal, rho):
    return r_inc_rho(res, cal, rho) / tot_inc_rho(res, cal, rho)

# incidence share (income weighted)
def r_incS_wgt(res, cal):
    return l_weight(cal) * (r_inc(res, cal) / tot_inc_wgt(res, cal))

def r_incS_wgt_rho(res, cal, rho):
    return l_weight(cal) * (r_inc_rho(res, cal, rho) / tot_inc_wgt_rho(res, cal, rho))

# worker incidence
def wrk_inc(res, cal):
    return w_pred(res, cal) - cal[1] * r_inc(res, cal)

def wrk_inc_rho(res, cal, rho):
    return w_pred_rho(res, cal, rho) - cal[1] * r_inc_rho(res, cal, rho)

# incidence share
def wrk_incS(res, cal):
    return wrk_inc(res, cal) / tot_inc(res, cal)

def wrk_incS_rho(res, cal, rho):
    return wrk_inc_rho(res, cal, rho) / tot_inc_rho(res, cal, rho)

# incidence share (income weighted)
def wrk_incS_wgt(res, cal):
    return w_weight(cal) * (wrk_inc(res, cal) / tot_inc_wgt(res, cal))

def wrk_incS_wgt_rho(res, cal, rho):
    return w_weight(cal) * (wrk_inc_rho(res, cal, rho) / tot_inc_wgt_rho(res, cal, rho))

# firm incidence
def frm_inc(res, cal):
    return 1 - cal[2] * res[3] * (cal[3] + 1) + cal[0] * (cal[3] + 1) * w_pred(res, cal)

def frm_inc_rho(res, cal, rho):
    return 1 - cal[2] * rho * (cal[3] + 1) + cal[0] * (cal[3] + 1) * w_pred_rho(res, cal, rho)

# incidence share
def frm_incS(res, cal):
    return frm_inc(res, cal) / tot_inc(res, cal)

def frm_incS_rho(res, cal, rho):
    return frm_inc_rho(res, cal, rho) / tot_inc_rho(res, cal, rho)

# incidence share (income weighted)
def frm_incS_wgt(res, cal):
    return f_weight(cal) * (frm_inc(res, cal) / tot_inc_wgt(res, cal))

def frm_incS_wgt_rho(res, cal, rho):
    return f_weight(cal) * (frm_inc_rho(res, cal, rho) / tot_inc_wgt_rho(res, cal, rho))

# model parameters
def eLS(res, cal):
    """epsilon LS"""
    return (1 + res[2] - cal[1]) / (res[1] * (1 + res[2]) + cal[1])

def eLD(res, cal):
    """epsilon LD"""
    return -(cal[0] / res[0]) - 1

# If run as a script, execute with dummy data
if __name__ == "__main__":
    results = run_struct_incidence()
    print("Structural incidence results calculated successfully")
