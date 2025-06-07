#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
from tensorflow.python.framework.tensor_conversion_registry import convert
matplotlib.use('Agg')  # Non-interactive backend for cloud environments
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import gc
import os
import json
import signal
import sys
import time
from contextlib import contextmanager
warnings.filterwarnings('ignore')

def convert_arrays_for_json(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_arrays_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_arrays_for_json(item) for item in obj]
    else:
        return obj

# ============================================================================
# MEMORY LEAK PREVENTION SETUP - MUST BE BEFORE TensorFlow IMPORT
# ============================================================================

# Set environment variables to prevent memory leaks
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
except ImportError:
    PSUTIL_AVAILABLE = False
    def get_memory_usage():
        return 0
    print("psutil not available, memory monitoring disabled")

# Import TensorFlow with memory configuration
import tensorflow as tf

def setup_tensorflow_memory_management():
    """Configure TensorFlow for proper memory management and leak prevention."""
    print("🔧 Setting up TensorFlow memory management...")
    
    # Configure GPU memory growth to prevent memory fragmentation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️  GPU memory setup warning: {e}")
    
    # Configure CPU parallelism to reduce memory overhead
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    
    # Disable XLA JIT compilation which can cause memory leaks
    tf.config.optimizer.set_jit(False)
    
    print("✅ TensorFlow memory management configured")

# Set up TensorFlow immediately
setup_tensorflow_memory_management()

# Import other libraries after TensorFlow setup
from loglinearcorrection import DRE
import statsmodels.api as sm

# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================

@contextmanager
def tensorflow_memory_cleanup():
    """Context manager for proper TensorFlow cleanup after each model training."""
    initial_memory = get_memory_usage()
    
    try:
        yield
    finally:
        # Comprehensive cleanup sequence
        tf.keras.backend.clear_session()
        
        # Reset default graph (for TF 1.x compatibility)
        try:
            tf.compat.v1.reset_default_graph()
        except:
            pass
        
        # Force Python garbage collection
        gc.collect()
        
        # Brief pause to allow cleanup
        time.sleep(0.1)
        
        # Check cleanup effectiveness
        final_memory = get_memory_usage()
        memory_diff = final_memory - initial_memory
        
        if memory_diff > 100:  # More than 100MB increase
            print(f"⚠️  Memory increased by {memory_diff:.1f}MB - executing emergency cleanup")
            emergency_memory_cleanup()

def emergency_memory_cleanup():
    """Emergency memory cleanup when normal cleanup isn't sufficient."""
    for i in range(3):
        tf.keras.backend.clear_session()
        gc.collect()
        time.sleep(0.1)
    
    # Try to force TensorFlow to release GPU memory
    if tf.config.list_physical_devices('GPU'):
        try:
            with tf.device('/GPU:0'):
                temp = tf.constant([1.0])
                del temp
        except:
            pass

def memory_monitor(func):
    """Decorator to monitor memory usage of functions."""
    def wrapper(*args, **kwargs):
        initial = get_memory_usage()
        result = func(*args, **kwargs)
        final = get_memory_usage()
        diff = final - initial
        
        if diff > 20:  # More than 20MB increase
            print(f"📊 {func.__name__}: Memory +{diff:.1f}MB ({initial:.1f} → {final:.1f})")
        
        return result
    return wrapper

# ============================================================================
# GLOBAL VARIABLES AND SIGNAL HANDLING
# ============================================================================

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False
LAST_CHECKPOINT_TIME = time.time()
CLOUD_BACKUP_ENABLED = False
BUCKET_NAME = None

def setup_cloud_backup():
    """Setup cloud backup if running on GCP."""
    global CLOUD_BACKUP_ENABLED, BUCKET_NAME
    
    try:
        import subprocess
        result = subprocess.run(['gsutil', 'ls'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            try:
                project_id = subprocess.run(['gcloud', 'config', 'get-value', 'project'], 
                                          capture_output=True, text=True, timeout=5).stdout.strip()
                BUCKET_NAME = f"{project_id}-bias-variance"
                
                result = subprocess.run(['gsutil', 'ls', f'gs://{BUCKET_NAME}/'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    CLOUD_BACKUP_ENABLED = True
                    print(f"✓ Cloud backup enabled: gs://{BUCKET_NAME}/")
                    return True
            except:
                pass
    except:
        pass
    
    print("⚠ Cloud backup disabled (not on GCP or gsutil unavailable)")
    return False

def cloud_backup_file(local_path, cloud_path):
    """Backup file to cloud storage."""
    if not CLOUD_BACKUP_ENABLED or not BUCKET_NAME:
        return False
    
    try:
        import subprocess
        cmd = f"gsutil cp '{local_path}' 'gs://{BUCKET_NAME}/{cloud_path}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
        return result.returncode == 0
    except:
        return False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully with immediate checkpoint save."""
    global SHUTDOWN_REQUESTED
    print(f"\n🚨 PREEMPTION SIGNAL {signum} RECEIVED!")
    print("⏱️  Have ~30 seconds to save state...")
    SHUTDOWN_REQUESTED = True
    
    if hasattr(signal_handler, 'emergency_save'):
        try:
            print("💾 Executing emergency save...")
            signal_handler.emergency_save()
            print("✅ Emergency save completed")
        except Exception as e:
            print(f"❌ Emergency save failed: {e}")

# Register signal handlers for GCP preemption
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ============================================================================
# DATA GENERATING PROCESSES
# ============================================================================

class DataGeneratingProcess:
    """Base class for data generating processes with known semi-elasticities."""
    
    def __init__(self, beta0=1.0, beta1=0.5, seed=None):
        self.beta0 = beta0
        self.beta1 = beta1
        if seed is not None:
            np.random.seed(seed)
    
    def generate_x(self, n):
        """Generate covariate data."""
        return np.random.uniform(0.5, 4.5, n)
    
    def generate_data(self, n):
        """Generate (y, X) data."""
        raise NotImplementedError
    
    def true_semi_elasticity(self, x):
        """Calculate true semi-elasticity at point x."""
        raise NotImplementedError

class ExponentialModel(DataGeneratingProcess):
    """Exponential model: y = exp(β₀ + β₁x + u) with E[exp(u)|x] = 1"""
    
    def generate_data(self, n):
        x = self.generate_x(n)
        X = np.column_stack([np.ones(n), x])
        
        sigma2 = 0.5
        u = np.random.normal(-sigma2/2, np.sqrt(sigma2), n)
        
        y = np.exp(self.beta0 + self.beta1 * x + u)
        return y, X[:, 1:]
    
    def true_semi_elasticity(self, x):
        return self.beta1

class LogLinearHeteroskedastic(DataGeneratingProcess):
    """Log-linear model with heteroskedasticity: log(y) = β₀ + β₁x + v"""
    
    def generate_data(self, n):
        x = self.generate_x(n)
        X = np.column_stack([np.ones(n), x])
        
        v = np.random.normal(0, 0.3 * x, n)
        
        log_y = self.beta0 + self.beta1 * x + v
        y = np.exp(log_y)
        return y, X[:, 1:]
    
    def true_semi_elasticity(self, x):
        return self.beta1 + 0.09 * x

class SkewedLogLinear(DataGeneratingProcess):
    """Log-linear model with skewed errors."""
    
    def generate_data(self, n):
        x = self.generate_x(n)
        X = np.column_stack([np.ones(n), x])
        
        df = 3
        v_raw = (np.random.chisquare(df, n) - df) / np.sqrt(2*df)
        v = 0.3 * v_raw
        
        log_y = self.beta0 + self.beta1 * x + v
        y = np.exp(log_y)
        return y, X[:, 1:]
    
    def true_semi_elasticity(self, x):
        return self.beta1 


class HeteroSkewedLogLinear(DataGeneratingProcess):
    """Log linear model with skewed errors dependent on x."""
    def generate_data(self, n):
        x = self.generate_x(n)
        X = np.column_stack([np.ones(n), x])
        
        # Degrees of freedom as a function of x
        df = np.maximum(1, 3 + 0.5 * x)  # Ensure df >= 1
        
        # Generate chi-square errors with varying df
        v_raw = np.zeros(n)
        for i in range(n):
            chi_sq = np.random.chisquare(df[i])
            v_raw[i] = (chi_sq - df[i]) / np.sqrt(2 * df[i])
        
        v = 1.0 * v_raw  # Updated scale to 1.0
        
        log_y = self.beta0 + self.beta1 * x + v
        y = np.exp(log_y)
        return y, X[:, 1:]
    
    def true_semi_elasticity(self, x):
        """
        Analytical semi-elasticity when df varies with x.
        
        For v = 1.0 * (χ²(df) - df)/√(2df), we have:
        E[exp(v)|x] = exp(-1.0√(df/2)) * (1 - 1.0√(2/df))^(-df/2)
        
        Semi-elasticity = β₁ + d/dx[ln(E[exp(v)|x])]
        """
        df = np.maximum(1, 3 + 0.5 * x)  # df(x) = 3 + 0.5x, d(df)/dx = 0.5
        
        # Analytical derivative of ln(E[exp(v)|x]) with respect to df
        sqrt_2_over_df = np.sqrt(2 / df)
        term1 = 1 - 1.0 * sqrt_2_over_df  # Updated scale to 1.0
        
        # Ensure term1 > 0 for log to be defined
        term1 = np.maximum(term1, 1e-10)
        
        # g'(df) where g(df) = ln(E[exp(v)|x])
        derivative_wrt_df = (
            -0.5 / np.sqrt(2 * df) -  # From d/d(df)[-1.0√(df/2)] = -0.5/√(2df)
            0.5 * np.log(term1) -      # From d/d(df)[-(df/2)*ln(1-1.0√(2/df))]
            (0.25 * sqrt_2_over_df) / term1  # From chain rule: 1.0 * 0.25 = 0.25
        )
        
        # Chain rule: d/dx = (d/d(df)) * (d(df)/dx) = g'(df) * 0.5
        derivative_wrt_x = 0.5 * derivative_wrt_df
        
        return self.beta1 + derivative_wrt_x


class MisspecifiedExponential(DataGeneratingProcess):
    """Exponential model with E[exp(u)|x] ≠ 1 (misspecified)."""
    
    def generate_data(self, n):
        x = self.generate_x(n)
        X = np.column_stack([np.ones(n), x])
        
        u = -0.5 + 0.2 * x + np.random.normal(0, 0.5, n)
        
        y = np.exp(self.beta0 + self.beta1 * x + u)
        return y, X[:, 1:]
    
    def true_semi_elasticity(self, x):
        return self.beta1 + 0.2

# ============================================================================
# MEMORY-EFFICIENT DRE ESTIMATION
# ============================================================================
def run_dre_with_memory_management(df, method='ols', bootstrap_reps=30):
    """Run DRE estimation with AGGRESSIVE memory leak prevention."""
    
    import tensorflow as tf
    import gc
    import time
    
    # Store initial memory state
    initial_memory = get_memory_usage()
    print(f"🧠 Starting DRE estimation (mem: {initial_memory:.1f}MB)")
    
    results = None
    
    try:
        # ULTRA memory-efficient neural network parameters
        nn_params = {
            'epochs': 8,          # Reduced from 12
            'batch_size': 16,     # Reduced from 24  
            'verbose': 0,
            'validation_split': 0.0,  # Disabled to save memory
            'shuffle': False,
            'use_multiprocessing': False,
            'workers': 1,
            'num_layers': 2,      # Reduced from 3
            'num_units': 32,      # Reduced from 64
        }
        
        # Clear any existing TensorFlow state
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Create DRE model
        dre = DRE(df['y'], df[['x']], interest='x', 
                 estimator_type='nn', 
                 nn_params=nn_params)
        
        # Fit model with MINIMAL bootstrap reps for memory
        results = dre.fit(method=method, 
                         compute_asymptotic_variance=False,
                         bootstrap=True, 
                         bootstrap_reps=min(bootstrap_reps, 10))  # Cap at 10 reps
        
        # Extract and clean results before cleanup
        clean_results = {'estimator_values': np.nan, 'bootstrap_variance': np.nan}
        
        if hasattr(results, 'estimator_values') and len(results.estimator_values) > 0:
            estimator_values = results.estimator_values[0]
            
            # Convert to numpy and copy to avoid TensorFlow references
            if hasattr(estimator_values, 'numpy'):
                estimator_values = estimator_values.numpy().copy()
            elif hasattr(estimator_values, 'copy'):
                estimator_values = estimator_values.copy()
            
            clean_results['estimator_values'] = estimator_values
            
            # Extract bootstrap variance
            if hasattr(results, 'bootstrap_se_dict') and len(results.bootstrap_se_dict) > 0:
                se = results.bootstrap_se_dict[0][2]
                bootstrap_variance = se**2 if se < 1.0 else se
                clean_results['bootstrap_variance'] = bootstrap_variance
        
        return clean_results
            
    except Exception as e:
        print(f"❌ DRE estimation failed: {e}")
        return {'estimator_values': np.nan, 'bootstrap_variance': np.nan}
        
    finally:
        # NUCLEAR memory cleanup sequence
        try:
            # Delete all local objects
            if 'dre' in locals():
                del dre
            if 'results' in locals():
                del results
            
            # Clear TensorFlow backend completely
            tf.keras.backend.clear_session()
            
            # Reset default graph (TF 1.x compatibility)
            try:
                tf.compat.v1.reset_default_graph()
            except:
                pass
            
            # Force Python garbage collection multiple times
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)
            
            # Try to force GPU memory release
            if tf.config.list_physical_devices('GPU'):
                try:
                    # Create and immediately delete a small tensor to trigger cleanup
                    with tf.device('/GPU:0'):
                        temp = tf.constant([1.0])
                        del temp
                except:
                    pass
            
            # Final memory check
            final_memory = get_memory_usage()
            memory_diff = final_memory - initial_memory
            
            if memory_diff > 50:  # More than 50MB increase
                print(f"⚠️ DRE memory increased by {memory_diff:.1f}MB")
                
                # Emergency cleanup if needed
                for _ in range(5):
                    tf.keras.backend.clear_session()
                    gc.collect()
                    time.sleep(0.05)
            else:
                print(f"✅ DRE memory managed (+{memory_diff:.1f}MB)")
                
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup error: {cleanup_error}")

# ============================================================================
# CORRECTED SIMULATION WITH MEMORY LEAK FIXES
# ============================================================================

@memory_monitor
def run_simulation_FIXED(dgp, n, bootstrap_reps=30):
    """
    Run simulation with:
    1. Fixed bootstrap variance logic
    2. Memory leak prevention 
    3. Proper bias calculation accounting for spatial variation
    """
    global SHUTDOWN_REQUESTED
    
    if SHUTDOWN_REQUESTED:
        return {method: {'bias': np.nan, 'variance': np.nan} 
                for method in ['OLS', 'PPML', 'DRSEE-OLS', 'DRSEE-PPML']}
    
    initial_memory = get_memory_usage()
    
    # Generate data
    y, X = dgp.generate_data(n)
    df = pd.DataFrame({'y': y, 'x': X.flatten()})
    
    # Calculate true semi-elasticities at each observation point
    true_se_values = np.array([dgp.true_semi_elasticity(xi) for xi in X.flatten()])
    true_se_mean = np.mean(true_se_values)  # For DGPs with constant true values
    base_true_values = true_se_values.copy()  # Store for all methods
    
    results = {}
    
    # ========================================================================
    # 1. LOG-LINEAR OLS
    # ========================================================================
    try:
        ols = sm.OLS(np.log(df['y']), sm.add_constant(df['x'])).fit()
        original_coef = ols.params[1]
        
        # Bootstrap for sampling variance of coefficient
        bs_coefficients = []
        for i in range(bootstrap_reps):
            if SHUTDOWN_REQUESTED:
                break
            idx = np.random.choice(n, n, replace=True)
            bs_ols = sm.OLS(np.log(df['y'].iloc[idx]), 
                           sm.add_constant(df['x'].iloc[idx])).fit()
            bs_coefficients.append(bs_ols.params[1])
            
            # Memory cleanup every 10 iterations
            if i % 10 == 0:
                gc.collect()
        
        if bs_coefficients:
            # Corrected bias calculation
            bootstrap_mean = np.mean(bs_coefficients)
            bias = bootstrap_mean - true_se_mean  # Difference from true parameter
            variance = np.var(bs_coefficients, ddof=1)  # Sample variance
            
            results['OLS'] = {
                'bias': bias,
                'variance': variance,
                'bootstrap_mean': bootstrap_mean,
                'true_value': true_se_mean,
                'n_bootstrap': len(bs_coefficients),
                'estimator_values_array': np.full(n, bootstrap_mean),  # OLS gives constant estimate
                'true_values_array': base_true_values
            }
        else:
            results['OLS'] = {'bias': np.nan, 'variance': np.nan}
        
        del bs_coefficients
        gc.collect()
        
    except Exception as e:
        print(f"OLS failed: {e}")
        results['OLS'] = {'bias': np.nan, 'variance': np.nan}
    
    if SHUTDOWN_REQUESTED:
        return results
    
    # ========================================================================
    # 2. PPML
    # ========================================================================
    try:
        ppml = sm.GLM(df['y'], sm.add_constant(df['x']), 
                      family=sm.families.Poisson()).fit()
        
        bs_coefficients = []
        for i in range(bootstrap_reps):
            if SHUTDOWN_REQUESTED:
                break
            idx = np.random.choice(n, n, replace=True)
            bs_ppml = sm.GLM(df['y'].iloc[idx], 
                            sm.add_constant(df['x'].iloc[idx]),
                            family=sm.families.Poisson()).fit()
            bs_coefficients.append(bs_ppml.params[1])
            
            if i % 10 == 0:
                gc.collect()
        
        if bs_coefficients:
            bootstrap_mean = np.mean(bs_coefficients)
            bias = bootstrap_mean - true_se_mean
            variance = np.var(bs_coefficients, ddof=1)
            
            results['PPML'] = {
                'bias': bias,
                'variance': variance,
                'bootstrap_mean': bootstrap_mean,
                'true_value': true_se_mean,
                'n_bootstrap': len(bs_coefficients),
                'estimator_values_array': np.full(n, bootstrap_mean),  # PPML gives constant estimate
                'true_values_array': base_true_values,
                }
        else:
            results['PPML'] = {'bias': np.nan, 'variance': np.nan}
            
        del bs_coefficients
        gc.collect()
        
    except Exception as e:
        print(f"PPML failed: {e}")
        results['PPML'] = {'bias': np.nan, 'variance': np.nan}
    
    if SHUTDOWN_REQUESTED:
        return results
    
    # ========================================================================
    # 3. DRSEE-OLS (Memory leak prone - use special handling)
    # ========================================================================
    try:
        print(f"🧠 DRSEE-OLS n={n} (mem: {get_memory_usage():.1f}MB)")
        
        dre_results = run_dre_with_memory_management(df, method='ols', 
                                                    bootstrap_reps=bootstrap_reps)
        
        if not (isinstance(dre_results['estimator_values'], float) and 
                np.isnan(dre_results['estimator_values'])):
            
            estimates = np.array(dre_results['estimator_values'])
            
            # Handle different result formats
            if estimates.ndim == 1 and len(estimates) == n:
                # Semi-elasticity estimates at each observation point
                bias_by_obs = estimates - true_se_values
                overall_bias = np.mean(bias_by_obs)
                variance = dre_results['bootstrap_variance']
                
            elif estimates.ndim == 1:
                # Bootstrap estimates (single value per bootstrap)
                overall_bias = np.mean(estimates) - true_se_mean
                variance = np.var(estimates, ddof=1)
                
            elif estimates.ndim == 2:
                # Bootstrap matrix [bootstrap_rep, observation] or vice versa
                if estimates.shape[0] == bootstrap_reps:
                    # [bootstrap_rep, observation]
                    bias_by_obs = np.mean(estimates, axis=0) - true_se_values
                    overall_bias = np.mean(bias_by_obs)
                    variance = np.mean(np.var(estimates, axis=0, ddof=1))
                else:
                    # [observation, bootstrap_rep]
                    bias_by_obs = np.mean(estimates, axis=1) - true_se_values
                    overall_bias = np.mean(bias_by_obs)
                    variance = np.mean(np.var(estimates, axis=1, ddof=1))
            else:
                # Scalar estimate
                overall_bias = float(estimates) - true_se_mean
                variance = dre_results['bootstrap_variance']
            
            results['DRSEE-OLS'] = {
                'bias': overall_bias,
                'variance': variance,
                'true_value': true_se_mean,
                'estimator_values_array': estimates.copy() if estimates.ndim == 1 and len(estimates) == n else np.full(n, np.mean(estimates)),
                'true_values_array': base_true_values
            }
        else:
            results['DRSEE-OLS'] = {'bias': np.nan, 'variance': np.nan}
        
        print(f"✅ DRSEE-OLS done (mem: {get_memory_usage():.1f}MB)")
        
    except Exception as e:
        print(f"DRSEE-OLS failed for n={n}: {e}")
        results['DRSEE-OLS'] = {'bias': np.nan, 'variance': np.nan}
    
    if SHUTDOWN_REQUESTED:
        return results
    
    # ========================================================================
    # 4. DRSEE-PPML
    # ========================================================================
    try:
        print(f"🧠 DRSEE-PPML n={n} (mem: {get_memory_usage():.1f}MB)")
        
        dre_results = run_dre_with_memory_management(df, method='ppml', 
                                                    bootstrap_reps=bootstrap_reps)
        
        if not (isinstance(dre_results['estimator_values'], float) and 
                np.isnan(dre_results['estimator_values'])):
            
            estimates = np.array(dre_results['estimator_values'])
            
            # Same logic as DRSEE-OLS
            if estimates.ndim == 1 and len(estimates) == n:
                bias_by_obs = estimates - true_se_values
                overall_bias = np.mean(bias_by_obs)
                variance = dre_results['bootstrap_variance']
            elif estimates.ndim == 1:
                overall_bias = np.mean(estimates) - true_se_mean
                variance = np.var(estimates, ddof=1)
            elif estimates.ndim == 2:
                if estimates.shape[0] == bootstrap_reps:
                    bias_by_obs = np.mean(estimates, axis=0) - true_se_values
                    overall_bias = np.mean(bias_by_obs)
                    variance = np.mean(np.var(estimates, axis=0, ddof=1))
                else:
                    bias_by_obs = np.mean(estimates, axis=1) - true_se_values
                    overall_bias = np.mean(bias_by_obs)
                    variance = np.mean(np.var(estimates, axis=1, ddof=1))
            else:
                overall_bias = float(estimates) - true_se_mean
                variance = dre_results['bootstrap_variance']
            
            results['DRSEE-PPML'] = {
                'bias': overall_bias,
                'variance': variance,
                'true_value': true_se_mean,
                'estimator_values_array': estimates.copy() if estimates.ndim == 1 and len(estimates) == n else np.full(n, np.mean(estimates)),
                'true_values_array': base_true_values
            }
        else:
            results['DRSEE-PPML'] = {'bias': np.nan, 'variance': np.nan}
        
        print(f"✅ DRSEE-PPML done (mem: {get_memory_usage():.1f}MB)")
        
    except Exception as e:
        print(f"DRSEE-PPML failed for n={n}: {e}")
        results['DRSEE-PPML'] = {'bias': np.nan, 'variance': np.nan}
    
    # ========================================================================
    # FINAL CLEANUP AND MEMORY CHECK
    # ========================================================================
    del df, y, X, true_se_values
    gc.collect()
    
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    if memory_increase > 50:  # More than 50MB increase per simulation
        print(f"⚠️  Memory leak: +{memory_increase:.1f}MB in simulation")
        emergency_memory_cleanup()
    
    return results

# ============================================================================
# CHECKPOINT AND PERSISTENCE FUNCTIONS
# ============================================================================

def save_frequent_checkpoint(state, checkpoint_id="current"):
    """Save checkpoint with cloud backup."""
    global LAST_CHECKPOINT_TIME
    
    checkpoint_file = f'./output/checkpoints/checkpoint_{checkpoint_id}.json'
    
    try:
        state['checkpoint_time'] = time.time()
        state['checkpoint_timestamp'] = str(pd.Timestamp.now())
        
        # Save locally
        with open(checkpoint_file, 'w') as f:
            json.dump(convert_arrays_for_json(state), f, indent=2)
        
        # Backup to cloud if available
        if CLOUD_BACKUP_ENABLED:
            cloud_path = f"checkpoints/checkpoint_{checkpoint_id}.json"
            cloud_backup_file(checkpoint_file, cloud_path)
        
        LAST_CHECKPOINT_TIME = time.time()
        print(f"💾 Checkpoint saved: {checkpoint_id}")
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint save failed: {e}")
        return False

def load_latest_checkpoint():
    """Load the most recent checkpoint with improved format handling."""
    # Try cloud first if available
    if CLOUD_BACKUP_ENABLED:
        try:
            import subprocess
            subprocess.run(['gsutil', '-m', 'cp', f'gs://{BUCKET_NAME}/checkpoints/*', 
                          './output/checkpoints/'], capture_output=True, timeout=30)
        except:
            pass
    
    # Find most recent checkpoint
    checkpoint_files = []
    if os.path.exists('./output/checkpoints/'):
        for f in os.listdir('./output/checkpoints/'):
            if f.startswith('checkpoint_') and f.endswith('.json'):
                full_path = f'./output/checkpoints/{f}'
                if os.path.getsize(full_path) > 0:  # Only consider non-empty files
                    checkpoint_files.append(f)
    
    if not checkpoint_files:
        print("📂 No valid checkpoints found - starting fresh")
        return None
    
    # Load the most recent
    latest_file = max(checkpoint_files, key=lambda x: os.path.getmtime(f'./output/checkpoints/{x}'))
    
    try:
        with open(f'./output/checkpoints/{latest_file}', 'r') as f:
            checkpoint = json.load(f)
        
        print(f"📂 Loaded checkpoint: {latest_file}")
        
        # Validate checkpoint structure
        if not isinstance(checkpoint, dict):
            print("❌ Invalid checkpoint format - starting fresh")
            return None
            
        # Print checkpoint info for debugging
        print(f"📊 Checkpoint contains: {list(checkpoint.keys())}")
        if 'dgp_name' in checkpoint:
            print(f"   DGP: {checkpoint['dgp_name']}")
        if 'current_size' in checkpoint:
            print(f"   Current size: {checkpoint['current_size']}")
        if 'current_rep' in checkpoint:
            print(f"   Current rep: {checkpoint['current_rep']}")
        if 'results' in checkpoint:
            print(f"   Results for sizes: {list(checkpoint['results'].keys())}")
            
        return checkpoint
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint {latest_file}: {e}")
        print("🔄 Starting fresh simulation")
        return None

# Replace your run_single_dgp_with_checkpoint function with this FIXED version:

def run_single_dgp_with_checkpoint(dgp_name, dgp_class, sample_sizes, n_reps_by_size, checkpoint_state):
    """Run simulation for a single DGP with FIXED checkpoint handling."""
    global SHUTDOWN_REQUESTED
    global LAST_CHECKPOINT_TIME
    print(f"\n🔬 Processing {dgp_name}...")
    
    # FIXED: Better checkpoint state initialization
    if (checkpoint_state and 
        isinstance(checkpoint_state, dict) and 
        checkpoint_state.get('dgp_name') == dgp_name and
        'results' in checkpoint_state):
        
        print(f"📂 Resuming {dgp_name} from checkpoint...")
        results = checkpoint_state['results']
        current_size = checkpoint_state.get('current_size', sample_sizes[0])
        current_rep = checkpoint_state.get('current_rep', 0)
        
        # Validate the current_size is actually in our sample_sizes
        if current_size not in sample_sizes:
            print(f"⚠️ Invalid current_size {current_size} in checkpoint, starting from beginning")
            current_size = sample_sizes[0]
            current_rep = 0
        
        print(f"   🎯 Resuming from size={current_size}, rep={current_rep}")
        
        # Debug: Show what we have in results
        for size, size_data in results.items():
            if isinstance(size_data, dict) and 'completed_reps' in size_data:
                print(f"   📊 Size {size}: {size_data['completed_reps']} reps completed")
        
    else:
        print(f"🆕 Starting {dgp_name} fresh...")
        results = {}
        current_size = sample_sizes[0]
        current_rep = 0
    
    # Emergency save function for signal handler
    def emergency_save():
        save_state = {
            'dgp_name': dgp_name,
            'current_size': current_size,
            'current_rep': current_rep,
            'results': results,
            'status': 'interrupted',
            'emergency_save': True,
            'timestamp': str(pd.Timestamp.now())
        }
        
        emergency_file = f'./output/checkpoints/checkpoint_emergency_{int(time.time())}.json'
        try:
            with open(emergency_file, 'w') as f:
                json.dump(save_state, f, indent=2)
            print(f"🚨 Emergency checkpoint saved: {emergency_file}")
            
            if CLOUD_BACKUP_ENABLED:
                cloud_backup_file(emergency_file, f'checkpoints/emergency_{int(time.time())}.json')
                
        except Exception as e:
            print(f"❌ Emergency save failed: {e}")
    
    signal_handler.emergency_save = emergency_save
    
    # Find starting point
    start_size_idx = 0
    for i, n in enumerate(sample_sizes):
        if n >= current_size:
            start_size_idx = i
            break
    
    print(f"📍 Starting from size index {start_size_idx} (size={sample_sizes[start_size_idx]})")
    
    for size_idx in range(start_size_idx, len(sample_sizes)):
        n = sample_sizes[size_idx]
        current_size = n
        
        if SHUTDOWN_REQUESTED:
            break
        
        n_reps = n_reps_by_size.get(n, 5)
        
        # FIXED: Initialize size results with proper structure
        if str(n) not in results:  # Use string keys for JSON compatibility
            results[str(n)] = {
                'completed_reps': 0,
                'biases': {method: [] for method in ['OLS', 'PPML', 'DRSEE-OLS', 'DRSEE-PPML']},
                'variances': {method: [] for method in ['OLS', 'PPML', 'DRSEE-OLS', 'DRSEE-PPML']}
            }
        
        # FIXED: Determine starting rep more carefully
        size_key = str(n)
        if n == current_size and size_key in results:
            # If we're resuming this exact size, check what we have
            completed_reps = results[size_key].get('completed_reps', 0)
            if completed_reps >= n_reps:
                print(f"✅ Size {n} already completed ({completed_reps}/{n_reps})")
                current_rep = 0  # Reset for next size
                continue
            else:
                # Resume from completed reps, not from checkpoint current_rep
                start_rep = completed_reps
                print(f"📊 Size {n}: Resuming from completed rep {start_rep}")
        else:
            # Different size, start from completed reps
            start_rep = results[size_key].get('completed_reps', 0)
        
        print(f"📊 Size {n}: Starting rep {start_rep}/{n_reps} (mem: {get_memory_usage():.1f}MB)")
        
        for rep in range(start_rep, n_reps):
            current_rep = rep
            
            if SHUTDOWN_REQUESTED:
                print(f"🛑 Shutdown during {dgp_name}, n={n}, rep={rep}")
                break
            
            print(f"   🔄 Rep {rep + 1}/{n_reps}...", end=' ')
            
            # Generate data and run simulation
            try:
                # Use more unique seeds to avoid repetition
                seed = hash(f"{dgp_name}_{n}_{rep}") % (2**31)
                dgp = dgp_class(beta0=1.0, beta1=0.5, seed=seed)
                rep_results = run_simulation_FIXED(dgp, n, bootstrap_reps=25)
                
                # Validate and store results
                valid_methods = 0
                for method in ['OLS', 'PPML', 'DRSEE-OLS', 'DRSEE-PPML']:
                    if (method in rep_results and 
                        not np.isnan(rep_results[method]['bias']) and 
                        not np.isnan(rep_results[method]['variance'])):
                        
                        results[size_key]['biases'][method].append(rep_results[method]['bias'])
                        results[size_key]['variances'][method].append(rep_results[method]['variance'])
                        if 'estimator_arrays' not in results[size_key]:
                            results[size_key]['estimator_arrays'] = {method: [] for method in ['OLS', 'PPML', 'DRSEE-OLS', 'DRSEE-PPML']}
                            results[size_key]['true_arrays'] = {method: [] for method in ['OLS', 'PPML', 'DRSEE-OLS', 'DRSEE-PPML']}
                        results[size_key]['estimator_arrays'][method].append(rep_results[method]['estimator_values_array'])
                        results[size_key]['true_arrays'][method].append(rep_results[method]['true_values_array'])
                        valid_methods += 1
                
                if valid_methods > 0:
                    results[size_key]['completed_reps'] = rep + 1
                    print(f"✅ ({valid_methods}/4 methods)")
                else:
                    print(f"⚠️ Failed (no valid results)")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
            
            # FIXED: Save checkpoint more frequently and reliably
            should_checkpoint = (
                (rep + 1) % 2 == 0 or  # Every 2 reps
                rep == n_reps - 1 or   # Last rep
                time.time() - LAST_CHECKPOINT_TIME > 180  # Every 3 minutes
            )
            
            if should_checkpoint:
                checkpoint_state = {
                    'dgp_name': dgp_name,
                    'current_size': current_size,
                    'current_rep': rep + 1,  # Next rep to do
                    'results': results,
                    'status': 'running',
                    'checkpoint_time': time.time(),
                    'checkpoint_timestamp': str(pd.Timestamp.now())
                }
                
                checkpoint_file = f'./output/checkpoints/checkpoint_{dgp_name.replace(" ", "_")}_current.json'
                try:
                    with open(checkpoint_file, 'w') as f:
                        json.dump(convert_arrays_for_json(checkpoint_state), f, indent=2)
                    
                    # Also save a timestamped backup
                    backup_file = f'./output/checkpoints/checkpoint_{dgp_name.replace(" ", "_")}_{int(time.time())}.json'
                    with open(backup_file, 'w') as f:
                        json.dump(convert_arrays_for_json(checkpoint_state), f, indent=2)
                    
                    if CLOUD_BACKUP_ENABLED:
                        cloud_backup_file(checkpoint_file, f'checkpoints/{os.path.basename(checkpoint_file)}')
                    
                    LAST_CHECKPOINT_TIME = time.time()
                    print(f"   💾 Checkpoint saved (rep {rep + 1})")
                    
                except Exception as e:
                    print(f"   ⚠️ Checkpoint save failed: {e}")
        
        # Save intermediate results after each size
        if results[size_key]['completed_reps'] > 0:
            size_summary = {
                method: {
                    'bias': np.mean(results[size_key]['biases'][method]) if results[size_key]['biases'][method] else np.nan,
                    'variance': np.mean(results[size_key]['variances'][method]) if results[size_key]['variances'][method] else np.nan,
                    'n_reps': len(results[size_key]['biases'][method])
                }
                for method in ['OLS', 'PPML', 'DRSEE-OLS', 'DRSEE-PPML']
            }
            
            size_file = f'./output/intermediate/{dgp_name.replace(" ", "_")}_{n}.json'
            try:
                with open(size_file, 'w') as f:
                    json.dump(size_summary, f, indent=2)
                
                if CLOUD_BACKUP_ENABLED:
                    cloud_backup_file(size_file, f'intermediate/{os.path.basename(size_file)}')
                
                print(f"💾 Size {n} results saved ({results[size_key]['completed_reps']} reps)")
            except Exception as e:
                print(f"⚠️ Failed to save size results: {e}")
        
        current_rep = 0  # Reset for next size
        
        if SHUTDOWN_REQUESTED:
            break
    
    # Clean up emergency save
    if hasattr(signal_handler, 'emergency_save'):
        delattr(signal_handler, 'emergency_save')
    
    # Convert string keys back to int keys for final results
    final_results = {}
    for size_str, size_data in results.items():
        try:
            size_int = int(size_str)
            final_results[size_int] = size_data
        except (ValueError, TypeError):
            final_results[size_str] = size_data
    
    return final_results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_bias_variance_plot(all_results, sample_sizes):
    """Create bias-variance plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    dgp_names = list(all_results.keys())
    methods = ['OLS', 'PPML', 'DRSEE-OLS', 'DRSEE-PPML']
    colors = {'OLS': 'blue', 'PPML': 'orange', 'DRSEE-OLS': 'green', 'DRSEE-PPML': 'red'}
    
    for idx, dgp_name in enumerate(dgp_names):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        for method in methods:
            biases = []
            variances = []
            valid_sizes = []
            
            for n in sample_sizes:
                if (n in all_results[dgp_name] and 
                    isinstance(all_results[dgp_name][n], dict) and
                    method in all_results[dgp_name][n]):
                    
                    bias = all_results[dgp_name][n][method]['bias']
                    variance = all_results[dgp_name][n][method]['variance']
                    
                    if not np.isnan(bias) and not np.isnan(variance) and variance > 0:
                        biases.append(abs(bias))
                        variances.append(variance)
                        valid_sizes.append(n)
            
            if len(valid_sizes) > 0:
                ax.plot(variances, biases, '-', color=colors[method], alpha=0.5)
                
                sizes = np.array(valid_sizes)
                if len(sizes) > 1:
                    sizes_normalized = 20 + 200 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
                else:
                    sizes_normalized = [50] * len(sizes)
                
                ax.scatter(variances, biases, s=sizes_normalized, 
                          color=colors[method], alpha=0.7, label=method)
        
        ax.set_xlabel('Variance', fontsize=12)
        ax.set_ylabel('Absolute Bias', fontsize=12)
        ax.set_title(dgp_name, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Use log scale if needed
        if len(ax.get_xlim()) > 0 and ax.get_xlim()[1] / (ax.get_xlim()[0] + 1e-10) > 100:
            ax.set_xscale('log')
        if len(ax.get_ylim()) > 0 and ax.get_ylim()[1] / (ax.get_ylim()[0] + 1e-10) > 100:
            ax.set_yscale('log')
    
    plt.suptitle('Bias-Variance Trade-off (Fixed Version)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./output/bias_variance_FIXED.png', dpi=150, bbox_inches='tight')
    
    if CLOUD_BACKUP_ENABLED:
        cloud_backup_file('./output/bias_variance_FIXED.png', 'final/bias_variance_FIXED.png')
    
    plt.close()
    print("📊 Bias-variance plot saved")

def create_summary_table(all_results):
    """Create summary DataFrame."""
    rows = []
    
    for dgp_name in all_results:
        for n in all_results[dgp_name]:
            if isinstance(all_results[dgp_name][n], dict):
                for method in ['OLS', 'PPML', 'DRSEE-OLS', 'DRSEE-PPML']:
                    if method in all_results[dgp_name][n]:
                        bias = all_results[dgp_name][n][method]['bias']
                        variance = all_results[dgp_name][n][method]['variance']
                        
                        row = {
                            'DGP': dgp_name,
                            'Sample Size': n,
                            'Method': method,
                            'Absolute Bias': abs(bias) if not np.isnan(bias) else np.nan,
                            'Variance': variance,
                            'RMSE': np.sqrt(bias**2 + variance) if not (np.isnan(bias) or np.isnan(variance)) else np.nan
                        }
                        rows.append(row)
    
    return pd.DataFrame(rows)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with complete fixes."""
    print("🚀 FIXED BIAS-VARIANCE SIMULATION")
    print("="*60)
    print("✅ Memory leak prevention")
    print("✅ Corrected bootstrap logic") 
    print("✅ Preemption resistance")
    print("✅ GPU optimization")
    print("="*60)
    
    # Setup cloud backup
    setup_cloud_backup()
    
    # GPU Detection and status
    print("\n=== GPU/TensorFlow Status ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA built: {tf.test.is_built_with_cuda()}")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
    
    if gpus:
        print("✅ GPU acceleration enabled")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
        try:
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([1.0, 2.0])
                print("✅ GPU computation test successful")
        except Exception as e:
            print(f"⚠ GPU test failed: {e}")
    else:
        print("⚠ NO GPU - Using CPU only")
    
    print("="*50)
    
    # Memory info
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        print(f"💾 Memory: {mem.available/1024/1024/1024:.1f} GB available")
        print(f"💾 Current usage: {get_memory_usage():.1f} MB")
    
        # Load checkpoint with better error handling
    checkpoint = load_latest_checkpoint()
    
    # Define DGPs
    dgps = {
        'Hetero-Skewed Log-Linear': HeteroSkewedLogLinear,
        'Heteroskedastic Log-Linear': LogLinearHeteroskedastic,
        'Misspecified Exponential': MisspecifiedExponential,
    }
    
    sample_sizes = [30, 100, 300, 1000, 3000, 10000, 30000, 100000]
    n_reps_by_size = {30: 12, 100: 10, 300: 8, 1000: 8, 3000: 8, 10000: 8, 30000: 8, 100000: 8}
    
    # Store all results
    all_results = {}
    
    # FIXED checkpoint handling logic
    if checkpoint:
        print("🔄 RESUMING FROM CHECKPOINT")
        print("="*40)
        
        # Determine what's been completed and what to resume
        completed_dgps = []
        resume_dgp = None
        
        if 'completed_dgps' in checkpoint:
            # New format - list of completed DGPs
            completed_dgps = checkpoint['completed_dgps']
            current_dgp = checkpoint.get('current_dgp')
            print(f"📋 Completed DGPs: {completed_dgps}")
            if current_dgp and current_dgp not in completed_dgps:
                resume_dgp = current_dgp
                print(f"🎯 Will resume: {resume_dgp}")
        elif 'dgp_name' in checkpoint:
            # Old format - single DGP in progress
            resume_dgp = checkpoint['dgp_name']
            print(f"🎯 Resuming DGP: {resume_dgp}")
        
        # Load any existing results from checkpoint
        if 'results' in checkpoint and isinstance(checkpoint['results'], dict):
            print("📊 Loading existing results from checkpoint...")
            # The checkpoint['results'] should be in the format we expect
            if resume_dgp:
                all_results[resume_dgp] = {}
                # Convert checkpoint format to final format if needed
                for size, size_data in checkpoint['results'].items():
                    if isinstance(size_data, dict) and 'completed_reps' in size_data:
                        # This is the internal format - convert to final format
                        all_results[resume_dgp][int(size)] = {
                            method: {
                                'bias': np.mean(size_data['biases'][method]) if size_data['biases'][method] else np.nan,
                                'variance': np.mean(size_data['variances'][method]) if size_data['variances'][method] else np.nan
                            }
                            for method in ['OLS', 'PPML', 'DRSEE-OLS', 'DRSEE-PPML']
                        }
                        print(f"   📏 Size {size}: {size_data['completed_reps']} reps completed")
        
        print("="*40)
    else:
        print("🆕 STARTING FRESH SIMULATION")
        completed_dgps = []
        resume_dgp = None
    
    # Process one DGP at a time
    dgp_items = list(dgps.items())
    
    for dgp_name, dgp_class in dgp_items:
        if SHUTDOWN_REQUESTED:
            print("\nShutdown requested, exiting gracefully...")
            break
            
        # Skip if already completed
        if dgp_name in completed_dgps:
            print(f"⏭️ Skipping {dgp_name} (already completed)")
            continue
        
        print(f"\n🎯 Processing {dgp_name} (memory: {get_memory_usage():.1f}MB)")
        
        # Prepare checkpoint state for this DGP
        if resume_dgp == dgp_name and checkpoint:
            print(f"🔄 Using checkpoint data for {dgp_name}")
            dgp_checkpoint_state = checkpoint
        else:
            print(f"🆕 Starting {dgp_name} fresh")
            dgp_checkpoint_state = {
                'dgp_name': dgp_name,
                'current_size': sample_sizes[0],
                'current_rep': 0,
                'results': {}
            }
        
        # Run the DGP with checkpoint support
        dgp_results = run_single_dgp_with_checkpoint(
            dgp_name, dgp_class, sample_sizes, n_reps_by_size, dgp_checkpoint_state
        )
        
        # Convert checkpoint format to final format if needed
        final_results = {}
        for n in dgp_results:
            if isinstance(dgp_results[n], dict) and 'biases' in dgp_results[n]:
                # This is the internal checkpoint format
                final_results[n] = {
                    method: {
                        'bias': np.mean(dgp_results[n]['biases'][method]) if dgp_results[n]['biases'][method] else np.nan,
                        'variance': np.mean(dgp_results[n]['variances'][method]) if dgp_results[n]['variances'][method] else np.nan,
                        'estimator_arrays': dgp_results[n]['estimator_arrays'][method] if 'estimator_arrays' in dgp_results[n] else [],
                        'true_arrays': dgp_results[n]['true_arrays'][method] if 'true_arrays' in dgp_results[n] else []
                    }
                    for method in ['OLS', 'PPML', 'DRSEE-OLS', 'DRSEE-PPML']
                }
            else:
                # Already in final format
                final_results[n] = dgp_results[n]
        
        all_results[dgp_name] = final_results
        
        # Save after each DGP with better error handling
        try:
            dgp_filename = f'./output/results_{dgp_name.replace(" ", "_")}.json'
            with open(dgp_filename, 'w') as f:
                json.dump(convert_arrays_for_json(final_results), f, indent=2)
            print(f"💾 Saved results for {dgp_name}")
            
            if CLOUD_BACKUP_ENABLED:
                cloud_backup_file(dgp_filename, f'final/results_{dgp_name.replace(" ", "_")}.json')
                
        except Exception as e:
            print(f"⚠️ Failed to save results for {dgp_name}: {e}")
        
        # Update completed DGPs and save progress checkpoint
        completed_dgps.append(dgp_name)
        
        # Find next DGP for checkpoint
        next_dgp = None
        for next_name, _ in dgp_items:
            if next_name not in completed_dgps:
                next_dgp = next_name
                break
        
        # Save overall progress checkpoint
        progress_checkpoint = {
            'completed_dgps': completed_dgps,
            'current_dgp': next_dgp,
            'status': 'dgp_completed',
            'timestamp': str(pd.Timestamp.now())
        }
        
        try:
            with open('./output/checkpoints/checkpoint_progress.json', 'w') as f:
                json.dump(progress_checkpoint, f, indent=2)
            
            if CLOUD_BACKUP_ENABLED:
                cloud_backup_file('./output/checkpoints/checkpoint_progress.json', 'checkpoints/checkpoint_progress.json')
                
            print(f"📊 Progress checkpoint saved: {len(completed_dgps)}/{len(dgp_items)} DGPs completed")
            
        except Exception as e:
            print(f"⚠️ Failed to save progress checkpoint: {e}")
        
        print(f"✅ {dgp_name} completed (memory: {get_memory_usage():.1f}MB)")
        
        # Aggressive memory cleanup between DGPs
        emergency_memory_cleanup()

    
    # Create visualization and summary
    if all_results:
        print("\n📊 Creating visualizations...")
        create_bias_variance_plot(all_results, sample_sizes)
        
        df = create_summary_table(all_results)
        df.to_csv('./output/bias_variance_results_FIXED.csv', index=False)
        
        if CLOUD_BACKUP_ENABLED:
            cloud_backup_file('./output/bias_variance_results_FIXED.csv', 'final/results.csv')
    
    # Remove checkpoint file on successful completion
    if os.path.exists('./output/intermediate/checkpoint.json'):
        os.remove('./output/intermediate/checkpoint.json')
    
    # Final cleanup
    if hasattr(signal_handler, 'checkpoint_callback'):
        delattr(signal_handler, 'checkpoint_callback')
    
    print("\n✅ SIMULATION COMPLETED!")
    print("📁 Results saved to ./output/")
    if CLOUD_BACKUP_ENABLED:
        print(f"☁️ Results backed up to gs://{BUCKET_NAME}/")
    
    print(f"\n📊 Final memory usage: {get_memory_usage():.1f}MB")

# ============================================================================
# CREATE OUTPUT DIRECTORIES AND RUN
# ============================================================================

if __name__ == "__main__":
    # Create output directories
    os.makedirs('./output', exist_ok=True)
    os.makedirs('./output/intermediate', exist_ok=True)
    os.makedirs('./output/checkpoints', exist_ok=True)
    
    main()
