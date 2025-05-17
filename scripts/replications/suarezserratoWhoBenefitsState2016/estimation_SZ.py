import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.linalg import pinv, inv
from numdifftools import Jacobian
import warnings

def estimation_SZ(beta, VC, cal, bounds, n_moments):
    """
    Main structural estimation function using GMM
    
    Args:
        beta: empirical moments vector
        VC: variance-covariance matrix
        cal: calibrated parameters
        bounds: parameter bounds (2 x n_params)
        n_moments: number of moments (4 or 6)
    
    Returns:
        RES: results matrix (estimates and standard errors)
        PredM: predicted moments
        VC_out: covariance matrix
        SSE_final: final sum of squared errors
    """
    # Setup
    gamma = cal[0]
    alpha = cal[1] 
    delta = cal[2]
    ePD = cal[3]
    rho = cal[4] if len(cal) == 5 else None
    
    # Select appropriate estimates
    EmpM = beta.flatten() if beta.ndim > 1 else beta
    EmpVC = VC
    
    n_beta = len(EmpM)
    
    # Set the weighting matrix
    W = inv(EmpVC)
    
    # Set number of random starting points
    if n_beta == 18:
        noSearchInits = 200
    elif n_moments == 4:
        noSearchInits = 1000
    else:
        noSearchInits = 20
    
    # Bounds
    noOfParams = bounds.shape[1]
    lb = bounds[0, :]
    ub = bounds[1, :]
    
    # Generate random starting points
    searchInits = lb + (ub - lb) * np.random.rand(noSearchInits, noOfParams)
    
    ParamHats = np.full((noSearchInits, noOfParams), np.nan)
    Sse = np.full((noSearchInits, 1), np.nan)
    
    # Define nested functions
    def fpredict4(p):
        """Prediction function for structural model"""
        if n_moments == 4:
            A = np.array([
                [1, -1/p[1], alpha/p[1], 0],
                [1, -(gamma*(ePD+1)-1), 0, -(1+p[0]*(1+ePD))],
                [-1/(1 + p[2]), -1/(1 + p[2]), 1, 0],
                [0, gamma/p[0], 0, 1]
            ])
            
            if (n_beta//4) == 1:
                B = np.array([
                    [0],
                    [-(ePD+1)*delta],
                    [0],
                    [delta*rho/p[0] - (1/(p[0]*(ePD+1)))]
                ])
            elif (n_beta//4) == 3:
                B = np.array([
                    [0, 0, 1/p[1]],
                    [-(ePD+1)*delta, -(ePD+1)*p[3], 0],
                    [0, (-p[2]/(1+p[2]))*p[4], (1-p[5])/(1+p[2])],
                    [delta*rho/p[0] - (1/(p[0]*(ePD+1))), p[3]/p[0], 0]
                ])
            else:
                raise ValueError('Invalid input')
                
        elif n_moments == 6:
            A = np.array([
                [1, -1/p[1], alpha/p[1], 0, 0, 0],
                [1, 0, 0, -1, -1, -1],
                [-1/(1 + p[2]), -1/(1 + p[2]), 1, 0, 0, 0],
                [0, gamma/p[0], 0, 1, 0, 0],
                [0, -(gamma*(ePD+1)-1), 0, 0, 1, 0],
                [0, 0, 0, -p[0]*(ePD + 1), 0, 1]
            ])
            
            if (n_beta//6) == 1:
                B = np.array([
                    [0],
                    [0],
                    [0],
                    [delta*rho/p[0] - (1/(p[0]*(ePD+1)))],
                    [-(ePD+1)*delta*rho],
                    [0]
                ])
            elif (n_beta//6) == 3:
                B = np.array([
                    [0, 0, 1/p[1]],
                    [0, 0, 0],
                    [0, (-p[2]/(1+p[2]))*p[4], (1-p[5])/(1+p[2])],
                    [delta*rho/p[0] - (1/(p[0]*(ePD+1))), p[3]/p[0], 0],
                    [-(ePD+1)*delta*rho, -(ePD+1)*p[3], 0],
                    [0, 0, 0]
                ])
            else:
                raise ValueError('Invalid input')
        else:
            raise ValueError('Invalid input')
        
        # Solve and reshape for GMM
        try:
            b_targ = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            b_targ = np.linalg.lstsq(A, B, rcond=None)[0]
        
        f = b_targ.flatten()[:n_beta]
        return f
    
    def gmm4(p):
        """GMM objective function"""
        try:
            error = EmpM - fpredict4(p)
            error2 = error.T @ W @ error
            return error2
        except:
            return 1e10  # Return large value if evaluation fails
    
    # Point estimates - run optimization from multiple starting points
    for i in range(noSearchInits):
        try:
            # Use scipy's minimize with bounds
            bounds_scipy = [(lb[j], ub[j]) for j in range(len(lb))]
            
            result = minimize(gmm4, searchInits[i, :], 
                            method='L-BFGS-B', 
                            bounds=bounds_scipy,
                            options={'maxiter': 10000})
            
            if result.success:
                ParamHats[i, :] = result.x
                Sse[i, 0] = result.fun
            else:
                # Try different method if first fails
                result = minimize(gmm4, searchInits[i, :], 
                                method='SLSQP', 
                                bounds=bounds_scipy,
                                options={'maxiter': 10000})
                if result.success:
                    ParamHats[i, :] = result.x
                    Sse[i, 0] = result.fun
        except:
            continue
    
    # Find best estimate
    valid_indices = ~np.isnan(Sse[:, 0])
    if not np.any(valid_indices):
        raise RuntimeError("No successful optimizations")
    
    best_ind = np.argmin(Sse[valid_indices, 0])
    best_sse_idx = np.where(valid_indices)[0][best_ind]
    bestestimate = ParamHats[best_sse_idx, :]
    
    # Jacobian estimate
    jac_func = Jacobian(fpredict4)
    DFDY_CSD = jac_func(bestestimate)
    
    # Recover standard errors and covariance matrix
    DFDY_CSD = -DFDY_CSD
    A1 = DFDY_CSD.T @ W @ DFDY_CSD
    B1 = DFDY_CSD.T @ W @ EmpVC @ W @ DFDY_CSD
    
    # Use pseudo-inverse for numerical stability
    VC_out = pinv(A1) @ B1 @ pinv(A1)
    ses = np.sqrt(np.diag(VC_out))
    
    # Predicted moments at best estimate
    PredM = fpredict4(bestestimate)
    
    # Save the results
    RES = np.full((noOfParams * 2, 1), np.nan)
    RES[::2, 0] = bestestimate
    RES[1::2, 0] = ses
    
    SSE_final = np.min(Sse[valid_indices])
    
    return RES, PredM, VC_out, SSE_final
