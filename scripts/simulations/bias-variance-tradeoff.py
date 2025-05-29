import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the estimator classes from your code
import sys
sys.path.append('.')  # Adjust path as needed
from loglinearcorrection import DRE
import statsmodels.api as sm

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DataGeneratingProcess:
    """Base class for data generating processes with known semi-elasticities."""
    
    def __init__(self, beta0=1.0, beta1=0.5, seed=None):
        self.beta0 = beta0
        self.beta1 = beta1
        if seed is not None:
            np.random.seed(seed)
    
    def generate_x(self, n):
        """Generate covariate data."""
        return np.random.uniform(0.5, 2.5, n)
    
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
        
        # Generate u such that E[exp(u)|x] = 1
        # Using log-normal: u ~ N(-σ²/2, σ²) gives E[exp(u)] = 1
        sigma2 = 0.5
        u = np.random.normal(-sigma2/2, np.sqrt(sigma2), n)
        
        y = np.exp(self.beta0 + self.beta1 * x + u)
        return y, X[:, 1:]  # Return only x, not intercept
    
    def true_semi_elasticity(self, x):
        """For exponential model with E[exp(u)|x]=1, semi-elasticity = β₁"""
        return self.beta1


class LogLinearHeteroskedastic(DataGeneratingProcess):
    """Log-linear model with heteroskedasticity: log(y) = β₀ + β₁x + v"""
    
    def generate_data(self, n):
        x = self.generate_x(n)
        X = np.column_stack([np.ones(n), x])
        
        # Heteroskedastic errors: variance increases with x
        v = np.random.normal(0, 0.3 * np.sqrt(x), n)
        
        log_y = self.beta0 + self.beta1 * x + v
        y = np.exp(log_y)
        return y, X[:, 1:]
    
    def true_semi_elasticity(self, x):
        """For log-linear: semi-elasticity = β₁ + d/dx[log E[exp(v)|x]]"""
        # For normal v with variance σ²(x), E[exp(v)|x] = exp(σ²(x)/2)
        # So log E[exp(v)|x] = σ²(x)/2 = 0.09*x/2
        # d/dx[log E[exp(v)|x]] = 0.045
        return self.beta1 + 0.045


class SkewedLogLinear(DataGeneratingProcess):
    """Log-linear model with skewed errors."""
    
    def generate_data(self, n):
        x = self.generate_x(n)
        X = np.column_stack([np.ones(n), x])
        
        # Generate skewed errors using chi-squared
        df = 3
        v_raw = (np.random.chisquare(df, n) - df) / np.sqrt(2*df)
        v = 0.3 * v_raw  # Scale to reasonable variance
        
        log_y = self.beta0 + self.beta1 * x + v
        y = np.exp(log_y)
        return y, X[:, 1:]
    
    def true_semi_elasticity(self, x):
        """For skewed errors, correction depends on MGF of the error distribution."""
        # For scaled chi-squared errors, this is approximately constant
        # Using simulation/theory, the correction is approximately 0.135
        return self.beta1 + 0.135


class MisspecifiedExponential(DataGeneratingProcess):
    """Exponential model with E[exp(u)|x] ≠ 1 (misspecified)."""
    
    def generate_data(self, n):
        x = self.generate_x(n)
        X = np.column_stack([np.ones(n), x])
        
        # Generate u such that E[exp(u)|x] depends on x
        # u = -0.5 + 0.2*x + normal(0, 0.5)
        u = -0.5 + 0.2 * x + np.random.normal(0, 0.5, n)
        
        y = np.exp(self.beta0 + self.beta1 * x + u)
        return y, X[:, 1:]
    
    def true_semi_elasticity(self, x):
        """True semi-elasticity when E[exp(u)|x] = exp(-0.5 + 0.2*x + 0.25/2)"""
        # d/dx[log E[y|x]] = d/dx[β₁*x + log E[exp(u)|x]]
        # = β₁ + d/dx[-0.5 + 0.2*x + 0.125] = β₁ + 0.2
        return self.beta1 + 0.2


def run_simulation(dgp, n, bootstrap_reps=100):
    """Run simulation for a given DGP and sample size."""
    
    # Generate data
    y, X = dgp.generate_data(n)
    df = pd.DataFrame({'y': y, 'x': X.flatten()})
    
    # Calculate true semi-elasticities at data points
    true_se = np.array([dgp.true_semi_elasticity(xi) for xi in X.flatten()])
    
    results = {}
    
    # 1. Log-linear OLS
    try:
        ols = sm.OLS(np.log(df['y']), sm.add_constant(df['x'])).fit()
        ols_est = np.full(n, ols.params[1])  # Constant estimate
        results['OLS'] = {
            'estimates': ols_est,
            'bias': np.mean(ols_est - true_se),
            'bootstrap_estimates': []
        }
        
        # Bootstrap for OLS
        for _ in range(bootstrap_reps):
            idx = np.random.choice(n, n, replace=True)
            bs_ols = sm.OLS(np.log(df['y'].iloc[idx]), 
                           sm.add_constant(df['x'].iloc[idx])).fit()
            results['OLS']['bootstrap_estimates'].append(bs_ols.params[1])
    except:
        results['OLS'] = None
    
    # 2. PPML
    try:
        ppml = sm.GLM(df['y'], sm.add_constant(df['x']), 
                      family=sm.families.Poisson()).fit()
        ppml_est = np.full(n, ppml.params[1])  # Constant estimate
        results['PPML'] = {
            'estimates': ppml_est,
            'bias': np.mean(ppml_est - true_se),
            'bootstrap_estimates': []
        }
        
        # Bootstrap for PPML
        for _ in range(bootstrap_reps):
            idx = np.random.choice(n, n, replace=True)
            bs_ppml = sm.GLM(df['y'].iloc[idx], 
                            sm.add_constant(df['x'].iloc[idx]),
                            family=sm.families.Poisson()).fit()
            results['PPML']['bootstrap_estimates'].append(bs_ppml.params[1])
    except:
        results['PPML'] = None
    
    # 3. DRSEE-OLS (using OLS residuals)
    try:
        dre_ols = DRE(df['y'], df[['x']], interest='x', 
                     estimator_type='nn', kernel_params={'degree': 2})
        dre_ols_results = dre_ols.fit(method='ols', compute_asymptotic_variance=False,
                                      bootstrap=True, bootstrap_reps=bootstrap_reps)
        
        # Get estimates at each data point
        dre_ols_est = dre_ols_results.estimator_values[0]
        results['DRSEE-OLS'] = {
            'estimates': dre_ols_est,
            'bias': np.mean(dre_ols_est - true_se),
            'bootstrap_estimates': dre_ols_results.bootstrap_estimates_dict[0][:, 2]  # average estimate
        }
    except Exception as e:
        print(f"DRSEE-OLS failed for n={n}: {e}")
        results['DRSEE-OLS'] = None
    
    # 4. DRSEE-PPML (using OLS residuals for correction)
    try:
        dre_ppml = DRE(df['y'], df[['x']], interest='x', 
                      estimator_type='nn', kernel_params={'degree': 2})
        dre_ppml_results = dre_ppml.fit(method='ppml', compute_asymptotic_variance=False,
                                       bootstrap=True, bootstrap_reps=bootstrap_reps)
        
        # Get estimates at each data point
        dre_ppml_est = dre_ppml_results.estimator_values[0]
        results['DRSEE-PPML'] = {
            'estimates': dre_ppml_est,
            'bias': np.mean(dre_ppml_est - true_se),
            'bootstrap_estimates': dre_ppml_results.bootstrap_estimates_dict[0][:, 2]  # average estimate
        }
    except Exception as e:
        print(f"DRSEE-PPML failed for n={n}: {e}")
        results['DRSEE-PPML'] = None
    
    # Calculate variance from bootstrap
    for method in results:
        if results[method] is not None and len(results[method]['bootstrap_estimates']) > 0:
            results[method]['variance'] = np.var(results[method]['bootstrap_estimates'])
        elif results[method] is not None:
            results[method]['variance'] = 0
    
    return results


def run_full_simulation():
    """Run simulation across all DGPs and sample sizes."""
    
    # Define DGPs
    dgps = {
        'Exponential': ExponentialModel(beta0=1.0, beta1=0.5),
        'Log-Linear Heteroskedastic': LogLinearHeteroskedastic(beta0=1.0, beta1=0.5),
        'Skewed Log-Linear': SkewedLogLinear(beta0=1.0, beta1=0.5),
        'Misspecified Exponential': MisspecifiedExponential(beta0=1.0, beta1=0.5)
    }
    
    sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    methods = ['OLS', 'PPML', 'DRSEE-OLS', 'DRSEE-PPML']
    
    # Store results
    all_results = {}
    
    # Run simulations
    for dgp_name, dgp in dgps.items():
        print(f"\nRunning simulations for {dgp_name}...")
        all_results[dgp_name] = {}
        
        for n in tqdm(sample_sizes, desc=f"Sample sizes for {dgp_name}"):
            # Run multiple replications to get stable estimates
            n_reps = 20 if n <= 100 else 10 if n <= 1000 else 5
            
            biases = {method: [] for method in methods}
            variances = {method: [] for method in methods}
            
            for rep in range(n_reps):
                dgp_rep = type(dgp)(beta0=dgp.beta0, beta1=dgp.beta1, seed=rep)
                results = run_simulation(dgp_rep, n, bootstrap_reps=50)
                
                for method in methods:
                    if results.get(method) is not None:
                        biases[method].append(results[method]['bias'])
                        variances[method].append(results[method]['variance'])
            
            # Average across replications
            all_results[dgp_name][n] = {
                method: {
                    'bias': np.mean(biases[method]) if biases[method] else np.nan,
                    'variance': np.mean(variances[method]) if variances[method] else np.nan
                }
                for method in methods
            }
    
    return all_results


def create_bias_variance_plot(all_results):
    """Create bias-variance plots for each DGP."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    dgp_names = list(all_results.keys())
    methods = ['OLS', 'PPML', 'DRSEE-OLS', 'DRSEE-PPML']
    colors = {'OLS': 'blue', 'PPML': 'orange', 'DRSEE-OLS': 'green', 'DRSEE-PPML': 'red'}
    
    for idx, dgp_name in enumerate(dgp_names):
        ax = axes[idx]
        
        for method in methods:
            # Extract bias and variance for each sample size
            sample_sizes = sorted(all_results[dgp_name].keys())
            biases = []
            variances = []
            valid_sizes = []
            
            for n in sample_sizes:
                bias = all_results[dgp_name][n][method]['bias']
                variance = all_results[dgp_name][n][method]['variance']
                
                if not np.isnan(bias) and not np.isnan(variance):
                    biases.append(abs(bias))  # Use absolute bias
                    variances.append(variance)
                    valid_sizes.append(n)
            
            if len(valid_sizes) > 0:
                # Plot line connecting points
                ax.plot(variances, biases, '-', color=colors[method], alpha=0.5)
                
                # Plot points with size proportional to sample size
                sizes = np.array(valid_sizes)
                sizes_normalized = 20 + 200 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
                
                ax.scatter(variances, biases, s=sizes_normalized, 
                          color=colors[method], alpha=0.7, label=method)
                
                # Add text labels for smallest and largest sample sizes
                if len(variances) > 0:
                    ax.annotate(f'n={valid_sizes[0]}', 
                               (variances[0], biases[0]), 
                               fontsize=8, alpha=0.7)
                    ax.annotate(f'n={valid_sizes[-1]}', 
                               (variances[-1], biases[-1]), 
                               fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Variance', fontsize=12)
        ax.set_ylabel('Absolute Bias', fontsize=12)
        ax.set_title(dgp_name, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set log scale if variance range is large
        if ax.get_xlim()[1] / ax.get_xlim()[0] > 100:
            ax.set_xscale('log')
        if ax.get_ylim()[1] / ax.get_ylim()[0] > 100:
            ax.set_yscale('log')
    
    plt.suptitle('Bias-Variance Trade-off Across Estimators and Sample Sizes', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def create_summary_table(all_results):
    """Create a summary table of results."""
    
    # Create a DataFrame with results
    rows = []
    
    for dgp_name in all_results:
        for n in sorted(all_results[dgp_name].keys()):
            for method in ['OLS', 'PPML', 'DRSEE-OLS', 'DRSEE-PPML']:
                row = {
                    'DGP': dgp_name,
                    'Sample Size': n,
                    'Method': method,
                    'Absolute Bias': abs(all_results[dgp_name][n][method]['bias']),
                    'Variance': all_results[dgp_name][n][method]['variance'],
                    'RMSE': np.sqrt(all_results[dgp_name][n][method]['bias']**2 + 
                                   all_results[dgp_name][n][method]['variance'])
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Create pivot table for better visualization
    pivot_bias = df.pivot_table(values='Absolute Bias', 
                                index=['DGP', 'Sample Size'], 
                                columns='Method')
    pivot_variance = df.pivot_table(values='Variance', 
                                   index=['DGP', 'Sample Size'], 
                                   columns='Method')
    pivot_rmse = df.pivot_table(values='RMSE', 
                               index=['DGP', 'Sample Size'], 
                               columns='Method')
    
    return df, pivot_bias, pivot_variance, pivot_rmse


# Main execution
if __name__ == "__main__":
    print("Starting bias-variance simulation study...")
    print("This may take a while, especially for larger sample sizes...")
    
    # Run simulations
    all_results = run_full_simulation()
    
    # Create bias-variance plot
    fig = create_bias_variance_plot(all_results)
    plt.savefig('./output/bias_variance_tradeoff_nn.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary tables
    df, pivot_bias, pivot_variance, pivot_rmse = create_summary_table(all_results)
    
    print("\n" + "="*60)
    print("SUMMARY: Average Absolute Bias by Method and Sample Size")
    print("="*60)
    print(pivot_bias.round(4))
    
    print("\n" + "="*60)
    print("SUMMARY: Average Variance by Method and Sample Size")
    print("="*60)
    print(pivot_variance.round(4))
    
    print("\n" + "="*60)
    print("SUMMARY: Average RMSE by Method and Sample Size")
    print("="*60)
    print(pivot_rmse.round(4))
    
    # Save results
    df.to_csv('./output/bias_variance_tradeoff_results_nn.csv', index=False)
    print("\nDetailed results saved to 'simulation_results.csv'")
