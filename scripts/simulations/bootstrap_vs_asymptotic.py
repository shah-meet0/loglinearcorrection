#!/usr/bin/env python
"""
Bootstrap vs Asymptotic Variance Comparison
Compares asymptotic variance estimates (kernel-based) with bootstrap variance estimates
for DRSEE-OLS and DRSEE-PPML across different sample sizes and models.

Outputs:
- ./output/bootstrap_vs_asymptotic.png - Variance ratio plot
- ./output/bootstrap_vs_asymptotic.csv - Detailed results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# Import the estimator classes
from loglinearcorrection import DRE
import statsmodels.api as sm

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


class VarianceComparisonSimulation:
    """Compare asymptotic and bootstrap variance estimates across different models."""
    
    def __init__(self, seed=42):
        """Initialize the simulation."""
        np.random.seed(seed)
        self.models = {
            'Exponential': self.generate_exponential_data,
            'Log-Linear Heteroskedastic': self.generate_heteroskedastic_data,
            'Heavy-Tailed': self.generate_heavy_tailed_data,
            'Mixture Model': self.generate_mixture_data
        }
    
    def generate_exponential_data(self, n):
        """Generate data from exponential model: y = exp(β₀ + β₁x + u)"""
        x = np.random.uniform(0.5, 2.5, n)
        sigma2 = 0.5
        u = np.random.normal(-sigma2/2, np.sqrt(sigma2), n)
        y = np.exp(1.0 + 0.5 * x + u)
        return pd.DataFrame({'y': y, 'x': x})
    
    def generate_heteroskedastic_data(self, n):
        """Generate data with heteroskedastic errors."""
        x = np.random.uniform(0.5, 2.5, n)
        # Variance increases with x
        v = np.random.normal(0, 0.3 * np.sqrt(x), n)
        y = np.exp(1.0 + 0.5 * x + v)
        return pd.DataFrame({'y': y, 'x': x})
    
    def generate_heavy_tailed_data(self, n):
        """Generate data with heavy-tailed errors (t-distribution)."""
        x = np.random.uniform(0.5, 2.5, n)
        # T-distribution with 3 degrees of freedom
        v = 0.3 * np.random.standard_t(df=3, size=n)
        y = np.exp(1.0 + 0.5 * x + v)
        return pd.DataFrame({'y': y, 'x': x})
    
    def generate_mixture_data(self, n):
        """Generate data from a mixture of normals."""
        x = np.random.uniform(0.5, 2.5, n)
        # Mixture: 80% N(0, 0.1²), 20% N(0, 0.5²)
        mixture_indicator = np.random.binomial(1, 0.2, n)
        v = np.where(mixture_indicator == 0,
                     np.random.normal(0, 0.1, n),
                     np.random.normal(0, 0.5, n))
        y = np.exp(1.0 + 0.5 * x + v)
        return pd.DataFrame({'y': y, 'x': x})
    
    def estimate_variances(self, df, method='ols', bootstrap_reps=200):
        """Estimate both asymptotic and bootstrap variances."""
        try:
            # Fit with asymptotic variance (kernel method)
            dre_asymp = DRE(df['y'], df[['x']], interest='x', 
                           estimator_type='kernel',  # Use kernel for nonparametric part
                           density_estimator='kernel')  # Use kernel for density
            results_asymp = dre_asymp.fit(method=method, 
                                         compute_asymptotic_variance=True,
                                         bootstrap=False)
            
            # Get asymptotic standard error at average
            x_avg = np.mean(df['x']).reshape(1, -1)
            asymp_se = results_asymp.asymptotic_standard_error_at_point(x_avg, 0)
            asymp_var = asymp_se ** 2
            
            # Fit with bootstrap (using neural network if specified)
            dre_boot = DRE(df['y'], df[['x']], interest='x', 
                          estimator_type='nn',  # Use neural network for nonparametric part
                          density_estimator='nn',  # Use neural network for density
                          nn_params={'epochs': 50, 'verbose': 0})
            results_boot = dre_boot.fit(method=method, 
                                       compute_asymptotic_variance=False,
                                       bootstrap=True,
                                       bootstrap_reps=bootstrap_reps)
            
            # Get bootstrap variance
            boot_se = results_boot.bootstrap_se_dict[0][3]  # SE at average
            boot_var = boot_se ** 2
            
            return {
                'asymptotic_variance': asymp_var,
                'bootstrap_variance': boot_var,
                'ratio': asymp_var / boot_var if boot_var > 0 else np.nan,
                'estimate_asymp': results_asymp.estimate_at_average(0),
                'estimate_boot': results_boot.estimate_at_average(0)
            }
            
        except Exception as e:
            print(f"Error in estimation: {e}")
            return None
    
    def run_comparison(self, sample_sizes, n_replications=10):
        """Run variance comparison across sample sizes and models."""
        results = {
            'DRSEE-OLS': {model: {n: [] for n in sample_sizes} for model in self.models},
            'DRSEE-PPML': {model: {n: [] for n in sample_sizes} for model in self.models}
        }
        
        for model_name, data_generator in self.models.items():
            print(f"\nProcessing {model_name} model...")
            
            for n in tqdm(sample_sizes, desc=f"{model_name}"):
                for rep in range(n_replications):
                    # Generate data
                    df = data_generator(n)
                    
                    # DRSEE-OLS
                    result_ols = self.estimate_variances(df, method='ols')
                    if result_ols is not None:
                        results['DRSEE-OLS'][model_name][n].append(result_ols['ratio'])
                    
                    # DRSEE-PPML
                    result_ppml = self.estimate_variances(df, method='ppml')
                    if result_ppml is not None:
                        results['DRSEE-PPML'][model_name][n].append(result_ppml['ratio'])
        
        return results


def create_variance_ratio_plot(results, sample_sizes):
    """Create plot showing ratio of asymptotic to bootstrap variance."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    model_names = list(results['DRSEE-OLS'].keys())
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        
        # Calculate mean ratios and confidence intervals
        for method, color, marker in [('DRSEE-OLS', 'blue', 'o'), 
                                      ('DRSEE-PPML', 'red', 's')]:
            mean_ratios = []
            lower_bounds = []
            upper_bounds = []
            valid_sizes = []
            
            for n in sample_sizes:
                ratios = [r for r in results[method][model_name][n] if not np.isnan(r)]
                if len(ratios) > 0:
                    mean_ratio = np.mean(ratios)
                    se_ratio = np.std(ratios) / np.sqrt(len(ratios))
                    
                    mean_ratios.append(mean_ratio)
                    lower_bounds.append(mean_ratio - 1.96 * se_ratio)
                    upper_bounds.append(mean_ratio + 1.96 * se_ratio)
                    valid_sizes.append(n)
            
            if len(valid_sizes) > 0:
                # Plot line with confidence band
                ax.plot(valid_sizes, mean_ratios, color=color, marker=marker, 
                       markersize=8, linewidth=2, label=method)
                ax.fill_between(valid_sizes, lower_bounds, upper_bounds, 
                               color=color, alpha=0.2)
        
        # Add reference line at 1
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, 
                  label='Equal variances')
        
        # Formatting
        ax.set_xlabel('Sample Size', fontsize=12)
        ax.set_ylabel('Asymptotic Var / Bootstrap Var', fontsize=12)
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Set y-axis limits to focus on relevant range
        ax.set_ylim(0, 3)
    
    plt.suptitle('Ratio of Asymptotic to Bootstrap Variance Estimates\n' + 
                 '(Kernel-based Asymptotic vs NN-based Bootstrap)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def create_summary_statistics(results, sample_sizes):
    """Create summary statistics table."""
    
    summary_data = []
    
    for method in ['DRSEE-OLS', 'DRSEE-PPML']:
        for model in results[method]:
            for n in sample_sizes:
                ratios = [r for r in results[method][model][n] if not np.isnan(r)]
                if len(ratios) > 0:
                    summary_data.append({
                        'Method': method,
                        'Model': model,
                        'Sample Size': n,
                        'Mean Ratio': np.mean(ratios),
                        'Median Ratio': np.median(ratios),
                        'Std Dev': np.std(ratios),
                        'Min Ratio': np.min(ratios),
                        'Max Ratio': np.max(ratios),
                        'N Valid': len(ratios)
                    })
    
    return pd.DataFrame(summary_data)


def main():
    """Run the complete bootstrap vs asymptotic variance comparison."""
    
    # Create output directory
    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print("Bootstrap vs Asymptotic Variance Comparison")
    print("=" * 60)
    print("This simulation compares:")
    print("- Asymptotic variance (kernel-based)")
    print("- Bootstrap variance (neural network-based)")
    print("for DRSEE-OLS and DRSEE-PPML estimators")
    print("=" * 60)
    
    # Simulation parameters
    sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    n_replications = 10
    
    print(f"\nSample sizes: {sample_sizes}")
    print(f"Replications per configuration: {n_replications}")
    print("\nRunning simulation...")
    
    # Run simulation
    sim = VarianceComparisonSimulation(seed=42)
    results = sim.run_comparison(sample_sizes, n_replications)
    
    # Create and save plot
    fig = create_variance_ratio_plot(results, sample_sizes)
    plot_path = os.path.join(output_dir, 'bootstrap_vs_asymptotic.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_path}")
    
    # Create and save summary statistics
    summary_df = create_summary_statistics(results, sample_sizes)
    csv_path = os.path.join(output_dir, 'bootstrap_vs_asymptotic.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"✓ Data saved to: {csv_path}")
    
    # Display summary
    print("\n" + "=" * 60)
    print("SUMMARY: Average Variance Ratios")
    print("(Asymptotic / Bootstrap)")
    print("=" * 60)
    
    avg_ratios = summary_df.groupby(['Method', 'Sample Size'])['Mean Ratio'].mean().reset_index()
    pivot_avg = avg_ratios.pivot(index='Sample Size', columns='Method', values='Mean Ratio')
    print(pivot_avg.round(2))
    
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("- Ratio > 1: Asymptotic variance is larger (conservative)")
    print("- Ratio < 1: Bootstrap variance is larger (conservative)")
    print("- Ratio ≈ 1: Both methods agree")
    print("=" * 60)
    
    # Display summary for largest sample size
    print("\n" + "=" * 60)
    print("Summary Statistics for n=5000")
    print("=" * 60)
    summary_5000 = summary_df[summary_df['Sample Size'] == 5000]
    print(summary_5000[['Method', 'Model', 'Mean Ratio', 'Std Dev']].to_string(index=False))
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        pass
    
    print(f"\nSimulation complete! Check the {output_dir} directory for outputs.")
    
    return summary_df


if __name__ == "__main__":
    summary_df = main()

