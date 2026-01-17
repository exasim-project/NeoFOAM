#!/usr/bin/env python3
"""
Post-processing script for NeoFOAM DSL operator fusion benchmarks.
Visualizes performance comparisons between composable DSL operators and fused kernels.
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


def plot_runtime_comparison(df, test_case, output_dir):
    """Create runtime comparison between implicit and fused-implicit integration."""
    # Filter to only implicit-time-integration and fused-implicit-integration
    df_filtered = df[df['section2'].isin(['implicit-time-integration', 'fused-implicit-integration'])].copy()
    df_filtered = df_filtered[df_filtered['benchmark_name'] != 'OpenFOAM'].copy()
    
    if df_filtered.empty:
        print(f"  Warning: No DSL data found for {test_case}, skipping...")
        return
    
    # Clean up benchmark names (remove -fused suffix for grouping)
    df_filtered['Executor'] = df_filtered['benchmark_name'].str.replace('-fused', '')
    df_filtered['Method'] = df_filtered['section2'].map({
        'implicit-time-integration': 'DSL Composable',
        'fused-implicit-integration': 'Fused Kernel'
    })
    
    for mesh_type in df['MeshType'].unique():
        df_mesh = df_filtered[df_filtered['MeshType'] == mesh_type].copy()
        
        if df_mesh.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort resolutions numerically
        resolution_order = sorted(df_mesh['Resolution'].unique(), 
                                 key=lambda x: int(x.replace('N', '')))
        
        # Plot grouped bars
        sns.barplot(data=df_mesh, x='Resolution', y='avg_runtime', 
                   hue='Method', order=resolution_order,
                   ax=ax, palette=['#e74c3c', '#2ecc71'], edgecolor='black', linewidth=1.2)
        
        # Split by executor with facets
        g = sns.catplot(data=df_mesh, x='Resolution', y='avg_runtime',
                       hue='Method', col='Executor', order=resolution_order,
                       kind='bar', palette=['#e74c3c', '#2ecc71'],
                       edgecolor='black', linewidth=1.2, height=5, aspect=1.2,
                       col_order=['SerialExecutor', 'CPUExecutor', 'GPUExecutor'])
        
        g.set_axis_labels('Resolution', 'Runtime (ns)', fontsize=12, fontweight='bold')
        g.set_titles('{col_name}', fontsize=13, fontweight='bold')
        g.fig.suptitle(f'{test_case} - {mesh_type}\nRuntime Comparison: DSL Composable vs Fused Kernel', 
                      fontsize=14, fontweight='bold', y=1.02)
        
        for ax in g.axes.flat:
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.tick_params(axis='x', rotation=45, labelsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{test_case}_{mesh_type}_runtime_comparison.png', 
                   bbox_inches='tight', dpi=200)
        plt.close()


def plot_fusion_speedup(df, test_case, output_dir):
    """Plot speedup gained from kernel fusion (fused vs composable DSL)."""
    # Get only NeoN executors
    df_neon = df[df['section1'] == 'NeoN'].copy()
    
    # Pivot to compare implicit vs fused
    df_implicit = df_neon[df_neon['section2'] == 'implicit-time-integration'].copy()
    df_fused = df_neon[df_neon['section2'] == 'fused-implicit-integration'].copy()
    
    # Clean executor names
    df_implicit['Executor'] = df_implicit['benchmark_name']
    df_fused['Executor'] = df_fused['benchmark_name'].str.replace('-fused', '')
    
    # Merge on common keys
    df_compare = pd.merge(
        df_implicit[['MeshType', 'Resolution', 'Executor', 'avg_runtime']],
        df_fused[['MeshType', 'Resolution', 'Executor', 'avg_runtime']],
        on=['MeshType', 'Resolution', 'Executor'],
        suffixes=('_implicit', '_fused')
    )
    
    # Calculate fusion speedup
    df_compare['fusion_speedup'] = df_compare['avg_runtime_implicit'] / df_compare['avg_runtime_fused']
    
    for mesh_type in df_compare['MeshType'].unique():
        df_mesh = df_compare[df_compare['MeshType'] == mesh_type].copy()
        
        if df_mesh.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        resolution_order = sorted(df_mesh['Resolution'].unique(), 
                                 key=lambda x: int(x.replace('N', '')))
        
        # Custom color palette
        colors = {'SerialExecutor': '#3498db', 'CPUExecutor': '#e67e22', 'GPUExecutor': '#9b59b6'}
        
        sns.barplot(data=df_mesh, x='Resolution', y='fusion_speedup',
                   hue='Executor', order=resolution_order,
                   hue_order=['SerialExecutor', 'CPUExecutor', 'GPUExecutor'],
                   ax=ax, palette=colors, edgecolor='black', linewidth=1.2)
        
        # Add reference line at speedup = 1 (no improvement)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                  alpha=0.7, label='No speedup (baseline)', zorder=0)
        
        ax.set_xlabel('Resolution', fontsize=13, fontweight='bold')
        ax.set_ylabel('Fusion Speedup\n(Fused Kernel / DSL Composable)', fontsize=13, fontweight='bold')
        ax.set_title(f'{test_case} - {mesh_type}\nSpeedup from Operator Fusion', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(title='Executor', fontsize=10, title_fontsize=11, 
                 loc='upper left', framealpha=0.95)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Add value labels on bars
        for container in ax.containers[:-1]:  # Skip the baseline line
            ax.bar_label(container, fmt='%.2fx', fontsize=8, padding=3)
        
        # Add annotation about speedup interpretation
        ax.text(0.98, 0.02, 'Speedup > 1.0 means fused kernel is faster',
               transform=ax.transAxes, fontsize=9, style='italic',
               ha='right', va='bottom', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{test_case}_{mesh_type}_fusion_speedup.png', 
                   bbox_inches='tight', dpi=200)
        plt.close()


def plot_openfoam_comparison_with_fusion(df, test_case, output_dir):
    """Compare implicit and fused methods against OpenFOAM baseline."""
    for mesh_type in df['MeshType'].unique():
        df_mesh = df[df['MeshType'] == mesh_type].copy()
        
        # Get OpenFOAM implicit baseline for recalculation
        df_baseline = df_mesh[
            (df_mesh['benchmark_name'] == 'OpenFOAM') & 
            (df_mesh['section2'] == 'implicit-time-integration')
        ].copy()
        
        # Get NeoN implementations (only implicit and fused)
        df_plot = df_mesh[
            (df_mesh['benchmark_name'].str.contains('Executor')) & 
            (df_mesh['section1'] == 'NeoN') & 
            (df_mesh['section2'].isin(['implicit-time-integration', 'fused-implicit-integration']))
        ].copy()
        
        if df_plot.empty or df_baseline.empty:
            continue
        
        # Recalculate normalized_speedup against OpenFOAM implicit baseline
        baseline_dict = df_baseline.set_index('Resolution')['avg_runtime'].to_dict()
        df_plot['normalized_speedup'] = df_plot.apply(
            lambda row: baseline_dict.get(row['Resolution'], 1.0) / row['avg_runtime'],
            axis=1
        )
        
        # Create combined label
        df_plot['Method'] = df_plot.apply(
            lambda row: f"{row['benchmark_name'].replace('-fused', '')} - {row['section2'].replace('-time-integration', '').replace('fused-', 'fused ')}",
            axis=1
        )
        
        fig, ax = plt.subplots(figsize=(16, 9))
        
        resolution_order = sorted(df_plot['Resolution'].unique(), 
                                 key=lambda x: int(x.replace('N', '')))
        
        sns.barplot(data=df_plot, x='Resolution', y='normalized_speedup',
                   hue='Method', order=resolution_order,
                   ax=ax, edgecolor='black', linewidth=1.0)
        
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2.5, 
                  alpha=0.8, label='OpenFOAM implicit baseline', zorder=0)
        
        ax.set_xlabel('Resolution', fontsize=13, fontweight='bold')
        ax.set_ylabel('Normalized Speedup (vs OpenFOAM implicit)', fontsize=13, fontweight='bold')
        ax.set_title(f'{test_case} - {mesh_type}\nPerformance vs OpenFOAM Implicit Baseline', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(title='Implementation', fontsize=8, title_fontsize=9, 
                 loc='upper left', framealpha=0.95, ncol=2)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=7, padding=2)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{test_case}_{mesh_type}_complete_comparison.png', 
                   bbox_inches='tight', dpi=200)
        plt.close()


def create_summary_table(df, test_case, output_dir):
    """Create summary statistics table for fusion performance."""
    df_neon = df[df['section1'] == 'NeoN'].copy()
    
    df_implicit = df_neon[df_neon['section2'] == 'implicit-time-integration'].copy()
    df_fused = df_neon[df_neon['section2'] == 'fused-implicit-integration'].copy()
    
    df_implicit['Executor'] = df_implicit['benchmark_name']
    df_fused['Executor'] = df_fused['benchmark_name'].str.replace('-fused', '')
    
    df_compare = pd.merge(
        df_implicit[['MeshType', 'Resolution', 'Executor', 'avg_runtime']],
        df_fused[['MeshType', 'Resolution', 'Executor', 'avg_runtime']],
        on=['MeshType', 'Resolution', 'Executor'],
        suffixes=('_implicit', '_fused')
    )
    
    df_compare['fusion_speedup'] = df_compare['avg_runtime_implicit'] / df_compare['avg_runtime_fused']
    df_compare['percent_improvement'] = (df_compare['fusion_speedup'] - 1.0) * 100
    
    # Create summary by executor and mesh type
    summary = df_compare.groupby(['MeshType', 'Executor']).agg({
        'fusion_speedup': ['mean', 'min', 'max'],
        'percent_improvement': ['mean', 'min', 'max']
    }).round(3)
    
    # Save to CSV
    summary.to_csv(output_dir / f'{test_case}_fusion_summary.csv')
    
    print(f"\n  Fusion Performance Summary for {test_case}:")
    print(summary.to_string())
    
    return summary


def main():
    """Main processing function."""
    root = Path(os.getcwd()) / "dsl"
    results_dir = root / "results"
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    output_dir = results_dir / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    # Find CSV files
    csv_files = [f for f in results_dir.glob('*.csv') if not f.name.endswith('_summary.csv')]
    
    if not csv_files:
        print(f"Warning: No CSV files found in {results_dir}")
        sys.exit(1)
    
    print(f"Processing DSL benchmark results...")
    print(f"Found {len(csv_files)} CSV file(s) to process")
    
    for csv_file in csv_files:
        test_case = csv_file.stem
        
        print(f"\nProcessing {test_case}...")
        df = load_data(csv_file)
        
        # Create visualizations
        plot_runtime_comparison(df, test_case, output_dir)
        plot_fusion_speedup(df, test_case, output_dir)
        plot_openfoam_comparison_with_fusion(df, test_case, output_dir)
        create_summary_table(df, test_case, output_dir)
        
        print(f"  Generated plots and summary for {test_case}")
    
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    for plot_file in sorted(output_dir.glob('*.png')):
        print(f"  - {plot_file.name}")
    for summary_file in sorted(output_dir.glob('*_summary.csv')):
        print(f"  - {summary_file.name}")

if __name__ == '__main__':
    main()
