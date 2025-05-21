import numpy as np
from scipy.stats import chi2
from numdifftools import Jacobian
from scipy.linalg import inv

def rf_estimation_SZ(beta, VC, cal, type_val):
    """
    Computes reduced form incidence of estimates
    
    Args:
        beta: coefficient estimates
        VC: variance-covariance matrix
        cal: calibrated parameters [gamma, alpha, delta, epd]
        type_val: type of specification (currently only type=1 supported)
    
    Returns:
        INC: incidence estimates
        INC_VC: variance-covariance matrix of incidence
        SH: incidence shares
        SH_VC: variance-covariance matrix of shares
        WSH: weighted incidence shares
        WSH_VC: variance-covariance matrix of weighted shares
        pCV: p-values for conventional view tests
        SI: income shares matrix
    """
    # Setup
    gamma = cal[0]
    alpha = cal[1]
    delta = cal[2]
    epd = cal[3]
    
    # Set income shares
    profits = (1 - delta * (epd + 1)) / (-(epd + 1) * (1 - delta))
    total = (1 + alpha + profits)
    ishares1 = np.diag([profits/total, 1/total, alpha/total])
    SI = ishares1
    
    # Compute Incidence for different types
    if type_val == 1:
        # Calibration
        # Matrix of linear combinations
        c = np.array([1 - (epd + 1) * delta, 0, 0])
        A = np.array([
            [0, 0, 0],
            [(epd + 1) * gamma, 1, 0],
            [0, -alpha, 1],
            [0, 0, 0]
        ])
        
        # Incidence
        INC = beta @ A + c
        INC_VC = A.T @ VC @ A  # Apply continuous mapping theorem
    else:
        raise ValueError(f"Type {type_val} not implemented")
    
    # Helper functions
    def inc_shares(INC_vec):
        """Calculate incidence shares"""
        total = np.sum(INC_vec)
        shares = INC_vec / total
        return shares
    
    def winc_shares(INC_vec, ishares1):
        """Calculate income weighted incidence shares"""
        WINC = INC_vec @ ishares1
        total = np.sum(WINC)
        wshares = WINC / total
        return wshares
    
    # Shares of incidence
    SH = inc_shares(INC)
    # Delta method for variance
    jac_1 = Jacobian(inc_shares)(INC)
    SH_VC = jac_1 @ INC_VC @ jac_1.T
    
    # Test of conventional view
    L = np.eye(2)
    c_test = np.array([[0], [1]])
    qF1 = ((L @ SH[:2].reshape(-1, 1) - c_test).T @ 
           inv(L @ SH_VC[:2, :2] @ L.T) @ 
           (L @ SH[:2].reshape(-1, 1) - c_test))
    pTest1 = 1 - chi2.cdf(qF1, len(c_test))
    
    # Weighted Shares
    WSH = winc_shares(INC, ishares1)
    # Delta method
    jac_1 = Jacobian(lambda x: winc_shares(x, ishares1))(INC)
    WSH_VC = jac_1 @ INC_VC @ jac_1.T
    
    # Test of conventional view (weighted)
    qF2 = ((L @ WSH[:2].reshape(-1, 1) - c_test).T @ 
           inv(L @ WSH_VC[:2, :2] @ L.T) @ 
           (L @ WSH[:2].reshape(-1, 1) - c_test))
    pTest2 = 1 - chi2.cdf(qF2, len(c_test))
    
    pCV = np.array([pTest1, pTest2]).flatten()
    
    return INC, INC_VC, SH, SH_VC, WSH, WSH_VC, pCV, SI
