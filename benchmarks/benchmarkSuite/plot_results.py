#!/usr/bin/env python3
"""
Post-processing script for NeoFOAM implicit operator benchmarks.
Visualizes performance comparisons between OpenFOAM and NeoN executors.
"""

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
plt.rcParams['figure.dpi'] = 150

def load_data(csv_file):
    """Load benchmark data from CSV file."""
    return pd.read_csv(csv_file)


def plot_speedup_barchart(df, operator_name, output_dir):
    """Create seaborn bar chart for speedup comparison."""
    # Exclude OpenFOAM and create combined column for hue
    df_filtered = df[df['benchmark_name'] != 'OpenFOAM'].copy()
    
    if df_filtered.empty:
        print(f"  Warning: No non-OpenFOAM data found for {operator_name}, skipping...")
        return
    
    df_filtered['Executor_Type'] = df_filtered['benchmark_name'] + ' - ' + df_filtered['section2']
    
    for mesh_type in df['MeshType'].unique():
        df_mesh = df_filtered[df_filtered['MeshType'] == mesh_type].copy()
        
        if df_mesh.empty:
            print(f"  Warning: No data for mesh type {mesh_type}, skipping...")
            continue
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Determine section2 types in the data
        section2_types = sorted(df_mesh['section2'].unique())
        
        # Custom order: Serial, CPU, GPU, each with all section2 types
        executor_base_order = ['SerialExecutor', 'CPUExecutor', 'GPUExecutor']
        hue_order = [f'{exec} - {sec}' for exec in executor_base_order for sec in section2_types]
        # Filter to only include executors present in the data
        hue_order = [h for h in hue_order if h in df_mesh['Executor_Type'].unique()]
        
        if not hue_order:
            print(f"  Warning: No matching executors found for {mesh_type}, skipping...")
            continue
        
        # Sort resolutions numerically (extract number from 'N128' format)
        resolution_order = sorted(df_mesh['Resolution'].unique(), 
                                 key=lambda x: int(x.replace('N', '')))
        print(f"  Plotting {operator_name} - {mesh_type} with resolution order: {resolution_order}")
        
        # Create grouped bar chart with seaborn
        sns.barplot(data=df_mesh, x='Resolution', y='normalized_speedup', 
                   hue='Executor_Type', hue_order=hue_order, order=resolution_order,
                   ax=ax, palette='Set2', edgecolor='black', linewidth=1.2)
        
        # Add reference line at speedup = 1
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                  alpha=0.7, label='OpenFOAM baseline', zorder=0)
        
        # Customize
        ax.set_xlabel('Resolution', fontsize=13, fontweight='bold')
        ax.set_ylabel('Normalized Speedup (vs OpenFOAM)', fontsize=13, fontweight='bold')
        ax.set_title(f'{operator_name} - {mesh_type} Speedup Comparison', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(title='Executor', fontsize=9, title_fontsize=10, 
                 loc='upper left', framealpha=0.95, ncol=1)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=7, padding=2)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{operator_name}_{mesh_type}_speedup_barchart.png', 
                   bbox_inches='tight', dpi=200)
        plt.close()

def main():
    """Main processing function."""
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <benchmark_suite>")
        print("Example: python plot_results.py implicitOperators")
        sys.exit(1)
    
    # Get the benchmark suite directory from command line argument
    root = Path(os.getcwd()) / sys.argv[1]
    results_dir = root / "results"
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    output_dir = results_dir / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    # Find all CSV files in the results directory (excluding summary files)
    csv_files = [f for f in results_dir.glob('*.csv') if not f.name.endswith('_summary.csv')]
    
    if not csv_files:
        print(f"Warning: No CSV files found in {results_dir}")
        sys.exit(1)
    
    print(f"Processing benchmark suite: {sys.argv[1]}")
    print(f"Found {len(csv_files)} CSV file(s) to process")
    
    for csv_file in csv_files:
        operator = csv_file.stem  # Get filename without extension
        
        print(f"\nProcessing {operator}...")
        df = load_data(csv_file)
        
        # Create visualizations
        plot_speedup_barchart(df, operator, output_dir)
        
        print(f"  Generated plots for {operator}")
    
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    for plot_file in sorted(output_dir.glob('*.png')):
        print(f"  - {plot_file.name}")
    for csv_file in sorted(output_dir.glob('*.csv')):
        print(f"  - {csv_file.name}")

if __name__ == '__main__':
    main()
