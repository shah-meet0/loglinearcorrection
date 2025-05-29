#!/usr/bin/env python
"""
Run the bootstrap vs asymptotic variance comparison simulation.
Saves outputs to ./output/bootstrap_vs_asymptotic.png and ./output/bootstrap_vs_asymptotic.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the simulation code
from variance_comparison_simulation import (
    VarianceComparisonSimulation, 
    create_variance_ratio_plot, 
    create_summary_statistics
)

def main():
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
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        pass
    
    return summary_df

if __name__ == "__main__":
    summary_df = main()

