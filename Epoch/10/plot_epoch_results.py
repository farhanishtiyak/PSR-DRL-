#!/usr/bin/env python3
"""
RLBEEP Epoch Results Comparison Plotter
========================================

This script reads epoch results from CSV files and generates comprehensive comparison visualizations
between Traditional RLBEEP and DQL-Enhanced RLBEEP approaches for first node death time analysis.

Usage:
    python plot_epoch_results.py                                    # Auto-detect CSV files
    python plot_epoch_results.py --single-plot                      # Single large comparison plot
    python plot_epoch_results.py --show                             # Show plot interactively
    python plot_epoch_results.py --traditional-csv file1.csv --dql-csv file2.csv  # Manual file specification
    python plot_epoch_results.py --help                             # Show help

Features:
    - Automatic detection of Traditional and DQL-Enhanced CSV files
    - Side-by-side comparison of first node death time vs epochs
    - Distribution analysis and statistical comparison
    - Network survival rate comparison
    - Performance improvement metrics

Author: Research Team
Date: July 2025
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate comparison plots from RLBEEP epoch results CSV files')
    
    parser.add_argument('--traditional-csv', type=str, default=None,
                        help='CSV file containing traditional RLBEEP epoch results')
    
    parser.add_argument('--dql-csv', type=str, default=None,
                        help='CSV file containing DQL-enhanced RLBEEP epoch results')
    
    parser.add_argument('--auto-detect', action='store_true', default=True,
                        help='Automatically detect CSV files in results directory (default: True)')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename for the plot (default: auto-generated with timestamp)')
    
    parser.add_argument('--title', type=str, default='RLBEEP Protocol Comparison - Traditional vs DQL-Enhanced',
                        help='Main title for the plot')
    
    parser.add_argument('--ylim', type=str, default='1300,1700',
                        help='Y-axis limits for first death time plot (format: min,max)')
    
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved plot (default: 300)')
    
    parser.add_argument('--show', action='store_true',
                        help='Show the plot interactively')
    
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the plot to file')
    
    parser.add_argument('--single-plot', action='store_true',
                        help='Show only the first death time vs epochs plot in a separate large window')
    
    return parser.parse_args()

def auto_detect_csv_files(results_dir):
    """
    Auto-detect traditional and DQL-enhanced CSV files in results directory
    
    Args:
        results_dir (str): Path to results directory
        
    Returns:
        tuple: (traditional_csv_path, dql_csv_path)
    """
    traditional_csv = None
    dql_csv = None
    
    # Look for CSV files in results directory
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith('.csv'):
                if 'traditional' in filename.lower():
                    traditional_csv = os.path.join(results_dir, filename)
                elif 'dql' in filename.lower() or 'enhanced' in filename.lower():
                    dql_csv = os.path.join(results_dir, filename)
    
    return traditional_csv, dql_csv

def load_epoch_results(csv_path):
    """
    Load epoch results from CSV file
    
    Args:
        csv_path (str): Path to the CSV file containing epoch results
        
    Returns:
        pandas.DataFrame: DataFrame containing epoch results
    """
    if not csv_path or not os.path.exists(csv_path):
        print(f"WARNING: CSV file not found at {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df)} epoch results from {os.path.basename(csv_path)}")
        
        # Display basic information about the data
        if 'epoch' in df.columns:
            print(f"  Epochs range: {df['epoch'].min()} to {df['epoch'].max()}")
        
        if 'first_death_time' in df.columns:
            death_times = df['first_death_time']
            print(f"  First death time range: {death_times.min():.1f} to {death_times.max():.1f} seconds")
        
        return df
        
    except Exception as e:
        print(f"ERROR loading CSV file {csv_path}: {e}")
        return None

def plot_comparison_analysis(df_traditional, df_dql, results_dir, args):
    """
    Create single comparison visualization for First Node Death Time between Traditional and DQL-Enhanced RLBEEP
    
    Args:
        df_traditional (pandas.DataFrame): Traditional RLBEEP epoch results
        df_dql (pandas.DataFrame): DQL-Enhanced RLBEEP epoch results
        results_dir (str): Directory to save the plots
        args: Command line arguments
    """
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    fig.suptitle(args.title + ' - First Node Death Time Comparison', fontsize=18, fontweight='bold', y=0.95)
    
    # Parse y-axis limits - check actual data range first
    all_death_times = []
    if df_traditional is not None:
        all_death_times.extend(df_traditional['first_death_time'].values)
    if df_dql is not None:
        all_death_times.extend(df_dql['first_death_time'].values)
    
    if all_death_times:
        data_min = min(all_death_times)
        data_max = max(all_death_times)
        # Add 5% padding
        padding = (data_max - data_min) * 0.05
        auto_ylim_min = max(0, data_min - padding)
        auto_ylim_max = data_max + padding
    else:
        auto_ylim_min, auto_ylim_max = 0, 2000
    
    try:
        ylim_min, ylim_max = map(float, args.ylim.split(','))
        # Use auto-detected range if it's better
        if auto_ylim_min < ylim_min or auto_ylim_max > ylim_max:
            print(f"Adjusting y-axis range from {ylim_min},{ylim_max} to {auto_ylim_min:.0f},{auto_ylim_max:.0f} to fit data")
            ylim_min, ylim_max = auto_ylim_min, auto_ylim_max
    except:
        ylim_min, ylim_max = auto_ylim_min, auto_ylim_max
        print(f"Using auto-detected y-axis range: {ylim_min:.0f}-{ylim_max:.0f}")
    
    # Plot Traditional RLBEEP
    if df_traditional is not None:
        epochs_trad = df_traditional['epoch'].values
        death_times_trad = df_traditional['first_death_time'].values
        
        print(f"Plotting Traditional RLBEEP: {len(epochs_trad)} epochs, death times range {death_times_trad.min():.1f}-{death_times_trad.max():.1f}")
        
        ax.plot(epochs_trad, death_times_trad, 'b-', alpha=0.8, linewidth=2.5, 
                label='Traditional RLBEEP')
    
    # Plot DQL-Enhanced RLBEEP
    if df_dql is not None:
        epochs_dql = df_dql['epoch'].values
        death_times_dql = df_dql['first_death_time'].values
        
        print(f"Plotting DQL-Enhanced RLBEEP: {len(epochs_dql)} epochs, death times range {death_times_dql.min():.1f}-{death_times_dql.max():.1f}")
        
        ax.plot(epochs_dql, death_times_dql, 'r-', alpha=0.8, linewidth=2.5, 
                label='DQL-Enhanced RLBEEP')
    
    # Set labels and formatting with larger fonts
    ax.set_xlabel('Epoch Number', fontsize=16, fontweight='bold')
    ax.set_ylabel('First Node Death Time (seconds)', fontsize=16, fontweight='bold')
    ax.set_title('First Node Death Time: Traditional vs DQL-Enhanced RLBEEP', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(ylim_min, ylim_max)
    
    # Enhanced grid
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3, which='minor')
    
    # Set x-axis with better ticks
    all_epochs = []
    if df_traditional is not None:
        all_epochs.extend(df_traditional['epoch'].values)
    if df_dql is not None:
        all_epochs.extend(df_dql['epoch'].values)
    
    if all_epochs:
        max_epoch = max(all_epochs)
        ax.set_xlim(0, max_epoch * 1.02)
        
        # Set major and minor ticks for better readability
        if max_epoch <= 50:
            major_ticks = np.arange(0, max_epoch + 1, 5)
            minor_ticks = np.arange(0, max_epoch + 1, 1)
        elif max_epoch <= 100:
            major_ticks = np.arange(0, max_epoch + 1, 10)
            minor_ticks = np.arange(0, max_epoch + 1, 2)
        elif max_epoch <= 300:
            major_ticks = np.arange(0, max_epoch + 1, 25)
            minor_ticks = np.arange(0, max_epoch + 1, 5)
        else:
            major_ticks = np.arange(0, max_epoch + 1, 50)
            minor_ticks = np.arange(0, max_epoch + 1, 10)
        
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
    
    # Larger tick labels
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=10, width=1, length=3)
    
    # Legend with larger font
    ax.legend(fontsize=14, loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Add comprehensive statistics text box
    stats_text_lines = []
    
    if df_traditional is not None:
        death_times_trad = df_traditional['first_death_time'].values
        stats_text_lines.append("Traditional RLBEEP:")
        stats_text_lines.append(f'  Count: {len(death_times_trad)}')
        stats_text_lines.append(f'  Mean: {np.mean(death_times_trad):.1f}s')
        stats_text_lines.append(f'  Std: {np.std(death_times_trad):.1f}s')
        stats_text_lines.append(f'  Min: {np.min(death_times_trad):.1f}s')
        stats_text_lines.append(f'  Max: {np.max(death_times_trad):.1f}s')
        stats_text_lines.append("")
    
    if df_dql is not None:
        death_times_dql = df_dql['first_death_time'].values
        stats_text_lines.append("DQL-Enhanced RLBEEP:")
        stats_text_lines.append(f'  Count: {len(death_times_dql)}')
        stats_text_lines.append(f'  Mean: {np.mean(death_times_dql):.1f}s')
        stats_text_lines.append(f'  Std: {np.std(death_times_dql):.1f}s')
        stats_text_lines.append(f'  Min: {np.min(death_times_dql):.1f}s')
        stats_text_lines.append(f'  Max: {np.max(death_times_dql):.1f}s')
        stats_text_lines.append("")
    
    # Add improvement calculation
    if df_traditional is not None and df_dql is not None:
        mean_trad = np.mean(death_times_trad)
        mean_dql = np.mean(death_times_dql)
        improvement = ((mean_dql - mean_trad) / mean_trad) * 100
        stats_text_lines.append("Performance Improvement:")
        stats_text_lines.append(f'  {improvement:+.1f}% vs Traditional')
    
    if stats_text_lines:
        stats_text = '\n'.join(stats_text_lines)
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.9, edgecolor='black'))
    
    # Tight layout
    plt.tight_layout(pad=3.0)
    
    # Save the plot
    if not args.no_save:
        if args.output:
            plot_path = os.path.join(results_dir, args.output)
        else:
            plot_path = os.path.join(results_dir, 'rlbeep_comparison.png')
        
        plt.savefig(plot_path, dpi=args.dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Comparison plot saved to: {plot_path}")
    
    # Show the plot
    if args.show:
        plt.show()
    
    return plot_path if not args.no_save else None
    """
    Create comprehensive visualizations for epoch analysis
    
    Args:
        df (pandas.DataFrame): DataFrame containing epoch results
        results_dir (str): Directory to save the plots
        args: Command line arguments
    """
    if df is None or df.empty:
        print("ERROR: No data available for plotting")
        return None
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(args.title, fontsize=16, fontweight='bold')
    
    # Extract data
    epochs = df['epoch'].values
    death_times = df['first_death_time'].values
    
    # Parse y-axis limits
    try:
        ylim_min, ylim_max = map(float, args.ylim.split(','))
    except:
        ylim_min, ylim_max = 1000, 2000
        print(f"Warning: Invalid ylim format, using default range: {ylim_min}-{ylim_max}")
    
    # 1. First Death Time vs Epochs (Main Plot)
    ax1 = axes[0, 0]
    
    # Handle no-death cases
    max_sim_time = df['simulation_duration'].max() if 'simulation_duration' in df.columns else 80000
    no_death_mask = death_times >= max_sim_time
    death_times_plot = death_times.copy()
    death_times_plot[no_death_mask] = max_sim_time
    
    # Plot main line and points
    ax1.plot(epochs, death_times_plot, 'b-', alpha=0.7, linewidth=1.5, label='First Death Time')
    ax1.scatter(epochs, death_times_plot, c='red', s=25, alpha=0.7, zorder=5)
    
    # Mark no-death epochs
    if np.any(no_death_mask):
        ax1.scatter(epochs[no_death_mask], death_times_plot[no_death_mask], 
                   c='green', s=40, marker='^', label='No Deaths', zorder=6)
    
    # Set labels and formatting
    ax1.set_xlabel('Epoch Number', fontsize=12)
    ax1.set_ylabel('First Node Death Time (seconds)', fontsize=12)
    ax1.set_title('First Node Death Time vs Epochs', fontsize=14, fontweight='bold')
    ax1.set_ylim(ylim_min, ylim_max)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add trend line
    valid_mask = ~no_death_mask
    if np.sum(valid_mask) > 1:  # Need at least 2 points for trend
        z = np.polyfit(epochs[valid_mask], death_times_plot[valid_mask], 1)
        p = np.poly1d(z)
        trend_line = p(epochs)
        # Only show trend line within the y-axis limits
        trend_line = np.clip(trend_line, ylim_min, ylim_max)
        ax1.plot(epochs, trend_line, "r--", alpha=0.8, linewidth=2, 
                label=f'Trend: {z[0]:.2f}x + {z[1]:.0f}')
        ax1.legend(fontsize=10)
    
    # Add statistics text box
    if np.sum(valid_mask) > 0:
        valid_deaths = death_times[valid_mask]
        stats_text = f'Mean: {np.mean(valid_deaths):.0f}s\nStd: {np.std(valid_deaths):.0f}s\nMin: {np.min(valid_deaths):.0f}s\nMax: {np.max(valid_deaths):.0f}s'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Distribution of First Death Times
    ax2 = axes[0, 1]
    valid_deaths = death_times[~no_death_mask]
    
    if len(valid_deaths) > 0:
        # Create histogram
        n_bins = min(20, len(valid_deaths)//2) if len(valid_deaths) > 10 else 10
        ax2.hist(valid_deaths, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1)
        ax2.set_xlabel('First Death Time (seconds)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of First Death Times', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics lines
        mean_death = np.mean(valid_deaths)
        std_death = np.std(valid_deaths)
        ax2.axvline(mean_death, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_death:.0f}s')
        ax2.axvline(mean_death + std_death, color='orange', linestyle='--', alpha=0.7,
                   label=f'Mean + STD: {mean_death + std_death:.0f}s')
        ax2.axvline(mean_death - std_death, color='orange', linestyle='--', alpha=0.7,
                   label=f'Mean - STD: {mean_death - std_death:.0f}s')
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No Node Deaths Recorded', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax2.set_title('Distribution of First Death Times', fontsize=14, fontweight='bold')
    
    # 3. Final Live Percentage vs Epochs
    ax3 = axes[1, 0]
    if 'final_live_percentage' in df.columns:
        final_percentages = df['final_live_percentage'].values
        ax3.plot(epochs, final_percentages, 'g-', alpha=0.7, linewidth=1.5, label='Network Survival')
        ax3.scatter(epochs, final_percentages, c='darkgreen', s=20, alpha=0.7)
        ax3.set_xlabel('Epoch Number', fontsize=12)
        ax3.set_ylabel('Final Live Percentage (%)', fontsize=12)
        ax3.set_title('Network Survival vs Epochs', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 105)
        
        # Add average line
        avg_survival = np.mean(final_percentages)
        ax3.axhline(avg_survival, color='red', linestyle='--', linewidth=2,
                   label=f'Average: {avg_survival:.1f}%')
        ax3.legend(fontsize=10)
        
        # Add statistics text box
        stats_text = f'Mean: {avg_survival:.1f}%\nStd: {np.std(final_percentages):.1f}%\nMin: {np.min(final_percentages):.1f}%\nMax: {np.max(final_percentages):.1f}%'
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'No Survival Data Available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Network Survival vs Epochs', fontsize=14, fontweight='bold')
    
    # 4. Energy Efficiency vs Epochs
    ax4 = axes[1, 1]
    if 'energy_efficiency' in df.columns:
        efficiency = df['energy_efficiency'].values * 100  # Convert to percentage
        ax4.plot(epochs, efficiency, 'purple', alpha=0.7, linewidth=1.5, label='Energy Efficiency')
        ax4.scatter(epochs, efficiency, c='darkviolet', s=20, alpha=0.7)
        ax4.set_xlabel('Epoch Number', fontsize=12)
        ax4.set_ylabel('Energy Efficiency (%)', fontsize=12)
        ax4.set_title('Energy Efficiency vs Epochs', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add average line
        avg_efficiency = np.mean(efficiency)
        ax4.axhline(avg_efficiency, color='red', linestyle='--', linewidth=2,
                   label=f'Average: {avg_efficiency:.1f}%')
        ax4.legend(fontsize=10)
        
        # Add statistics text box
        stats_text = f'Mean: {avg_efficiency:.1f}%\nStd: {np.std(efficiency):.1f}%\nMin: {np.min(efficiency):.1f}%\nMax: {np.max(efficiency):.1f}%'
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'No Energy Efficiency Data Available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Energy Efficiency vs Epochs', fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    if not args.no_save:
        if args.output:
            plot_path = os.path.join(results_dir, args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(results_dir, f'epoch_analysis_{timestamp}.png')
        
        plt.savefig(plot_path, dpi=args.dpi, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    
    # Show the plot
    if args.show:
        plt.show()
    
    return plot_path if not args.no_save else None

def plot_first_death_time_only(df, results_dir, args):
    """
    Create a single large plot showing only First Death Time vs Epochs
    
    Args:
        df (pandas.DataFrame): DataFrame containing epoch results
        results_dir (str): Directory to save the plots
        args: Command line arguments
    """
    if df is None or df.empty:
        print("ERROR: No data available for plotting")
        return None
    
    # Create a large figure for single plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    fig.suptitle(args.title + ' - First Node Death Time Analysis', fontsize=18, fontweight='bold', y=0.95)
    
    # Extract data
    epochs = df['epoch'].values
    death_times = df['first_death_time'].values
    
    # Parse y-axis limits
    try:
        ylim_min, ylim_max = map(float, args.ylim.split(','))
    except:
        ylim_min, ylim_max = 1300, 1700
        print(f"Warning: Invalid ylim format, using default range: {ylim_min}-{ylim_max}")
    
    # Handle no-death cases
    max_sim_time = df['simulation_duration'].max() if 'simulation_duration' in df.columns else 80000
    no_death_mask = death_times >= max_sim_time
    death_times_plot = death_times.copy()
    death_times_plot[no_death_mask] = max_sim_time
    
    # Plot main line and points with larger markers
    ax.plot(epochs, death_times_plot, 'b-', alpha=0.8, linewidth=2, label='First Death Time', markersize=4)
    ax.scatter(epochs, death_times_plot, c='red', s=35, alpha=0.8, zorder=5, edgecolors='darkred', linewidth=0.5)
    
    # Mark no-death epochs with larger markers
    if np.any(no_death_mask):
        ax.scatter(epochs[no_death_mask], death_times_plot[no_death_mask], 
                   c='green', s=60, marker='^', label='No Deaths', zorder=6, 
                   edgecolors='darkgreen', linewidth=1)
    
    # Set labels and formatting with larger fonts
    ax.set_xlabel('Epoch Number', fontsize=16, fontweight='bold')
    ax.set_ylabel('First Node Death Time (seconds)', fontsize=16, fontweight='bold')
    ax.set_title('First Node Death Time vs Epochs', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(ylim_min, ylim_max)
    
    # Enhanced grid
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3, which='minor')
    
    # Set x-axis with more space and better ticks
    ax.set_xlim(0, max(epochs) * 1.02)  # Add 2% padding on the right
    
    # Set major and minor ticks for better readability
    max_epoch = max(epochs)
    if max_epoch <= 50:
        major_ticks = np.arange(0, max_epoch + 1, 5)
        minor_ticks = np.arange(0, max_epoch + 1, 1)
    elif max_epoch <= 100:
        major_ticks = np.arange(0, max_epoch + 1, 10)
        minor_ticks = np.arange(0, max_epoch + 1, 2)
    elif max_epoch <= 300:
        major_ticks = np.arange(0, max_epoch + 1, 25)
        minor_ticks = np.arange(0, max_epoch + 1, 5)
    else:
        major_ticks = np.arange(0, max_epoch + 1, 50)
        minor_ticks = np.arange(0, max_epoch + 1, 10)
    
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    
    # Larger tick labels
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=10, width=1, length=3)
    
    # Legend with larger font
    ax.legend(fontsize=14, loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Add comprehensive statistics text box in a better position
    valid_mask = ~no_death_mask
    if np.sum(valid_mask) > 0:
        valid_deaths = death_times[valid_mask]
        
        # Calculate additional statistics
        median_death = np.median(valid_deaths)
        q25 = np.percentile(valid_deaths, 25)
        q75 = np.percentile(valid_deaths, 75)
        cv = (np.std(valid_deaths) / np.mean(valid_deaths)) * 100  # Coefficient of variation
        
        stats_text = (f'Count: {len(valid_deaths)}\n'
                     f'Mean: {np.mean(valid_deaths):.1f}s\n'
                     f'Median: {median_death:.1f}s\n'
                     f'Std Dev: {np.std(valid_deaths):.1f}s\n'
                     f'Min: {np.min(valid_deaths):.1f}s\n'
                     f'Max: {np.max(valid_deaths):.1f}s\n'
                     f'Q25: {q25:.1f}s\n'
                     f'Q75: {q75:.1f}s\n'
                     f'CV: {cv:.1f}%')
        
        # Position the stats box in the upper right
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.9, edgecolor='black'))
    
    # Add epoch count information
    epoch_info = f'Total Epochs: {len(epochs)}\nNo Deaths: {np.sum(no_death_mask)}\nWith Deaths: {np.sum(valid_mask)}'
    ax.text(0.02, 0.98, epoch_info, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, edgecolor='navy'))
    
    # Tight layout with more padding
    plt.tight_layout(pad=3.0)
    
    # Save the plot
    if not args.no_save:
        if args.output:
            # Modify filename for single plot
            base_name = os.path.splitext(args.output)[0]
            ext = os.path.splitext(args.output)[1] or '.png'
            plot_path = os.path.join(results_dir, f'{base_name}_single_plot{ext}')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(results_dir, f'first_death_time_analysis_{timestamp}.png')
        
        plt.savefig(plot_path, dpi=args.dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Single plot saved to: {plot_path}")
    
    # Show the plot
    if args.show:
        plt.show()
    
    return plot_path if not args.no_save else None

def plot_comparison_first_death_time_only(df_traditional, df_dql, results_dir, args):
    """
    Create a single large plot comparing First Death Time vs Epochs between both approaches
    
    Args:
        df_traditional (pandas.DataFrame): Traditional RLBEEP epoch results
        df_dql (pandas.DataFrame): DQL-Enhanced RLBEEP epoch results  
        results_dir (str): Directory to save the plots
        args: Command line arguments
    """
    # Create a large figure for single plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    fig.suptitle(args.title + ' - First Node Death Time Analysis', fontsize=18, fontweight='bold', y=0.95)
    
    # Parse y-axis limits
    try:
        ylim_min, ylim_max = map(float, args.ylim.split(','))
    except:
        ylim_min, ylim_max = 1300, 1700
        print(f"Warning: Invalid ylim format, using default range: {ylim_min}-{ylim_max}")
    
    # Plot Traditional RLBEEP
    if df_traditional is not None:
        epochs_trad = df_traditional['epoch'].values
        death_times_trad = df_traditional['first_death_time'].values
        
        # Handle no-death cases for traditional
        max_sim_time_trad = df_traditional['simulation_duration'].max() if 'simulation_duration' in df_traditional.columns else 80000
        no_death_mask_trad = death_times_trad >= max_sim_time_trad
        death_times_plot_trad = death_times_trad.copy()
        death_times_plot_trad[no_death_mask_trad] = max_sim_time_trad
        
        # Plot traditional line
        ax.plot(epochs_trad, death_times_plot_trad, 'b-', alpha=0.8, linewidth=2.5, 
               label='Traditional RLBEEP')
        
        # Mark no-death epochs for traditional
        if np.any(no_death_mask_trad):
            ax.scatter(epochs_trad[no_death_mask_trad], death_times_plot_trad[no_death_mask_trad], 
                      c='lightblue', s=60, marker='^', label='Traditional - No Deaths', zorder=6, 
                      edgecolors='darkblue', linewidth=1)
    
    # Plot DQL-Enhanced RLBEEP
    if df_dql is not None:
        epochs_dql = df_dql['epoch'].values
        death_times_dql = df_dql['first_death_time'].values
        
        # Handle no-death cases for DQL
        max_sim_time_dql = df_dql['simulation_duration'].max() if 'simulation_duration' in df_dql.columns else 80000
        no_death_mask_dql = death_times_dql >= max_sim_time_dql
        death_times_plot_dql = death_times_dql.copy()
        death_times_plot_dql[no_death_mask_dql] = max_sim_time_dql
        
        # Plot DQL line
        ax.plot(epochs_dql, death_times_plot_dql, 'r-', alpha=0.8, linewidth=2.5, 
               label='DQL-Enhanced RLBEEP')
        
        # Mark no-death epochs for DQL
        if np.any(no_death_mask_dql):
            ax.scatter(epochs_dql[no_death_mask_dql], death_times_plot_dql[no_death_mask_dql], 
                      c='lightcoral', s=60, marker='^', label='DQL-Enhanced - No Deaths', zorder=6, 
                      edgecolors='darkred', linewidth=1)
    
    # Set labels and formatting with larger fonts
    ax.set_xlabel('Epoch Number', fontsize=16, fontweight='bold')
    ax.set_ylabel('First Node Death Time (seconds)', fontsize=16, fontweight='bold')
    ax.set_title('First Node Death Time Comparison: Traditional vs DQL-Enhanced RLBEEP', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(ylim_min, ylim_max)
    
    # Enhanced grid
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3, which='minor')
    
    # Set x-axis with more space and better ticks
    all_epochs = []
    if df_traditional is not None:
        all_epochs.extend(df_traditional['epoch'].values)
    if df_dql is not None:
        all_epochs.extend(df_dql['epoch'].values)
    
    if all_epochs:
        max_epoch = max(all_epochs)
        ax.set_xlim(0, max_epoch * 1.02)  # Add 2% padding on the right
        
        # Set major and minor ticks for better readability
        if max_epoch <= 50:
            major_ticks = np.arange(0, max_epoch + 1, 5)
            minor_ticks = np.arange(0, max_epoch + 1, 1)
        elif max_epoch <= 100:
            major_ticks = np.arange(0, max_epoch + 1, 10)
            minor_ticks = np.arange(0, max_epoch + 1, 2)
        elif max_epoch <= 300:
            major_ticks = np.arange(0, max_epoch + 1, 25)
            minor_ticks = np.arange(0, max_epoch + 1, 5)
        else:
            major_ticks = np.arange(0, max_epoch + 1, 50)
            minor_ticks = np.arange(0, max_epoch + 1, 10)
        
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
    
    # Larger tick labels
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=10, width=1, length=3)
    
    # Legend with larger font
    ax.legend(fontsize=14, loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Add comprehensive statistics text boxes
    stats_text_lines = []
    
    if df_traditional is not None:
        valid_mask_trad = death_times_trad < max_sim_time_trad
        if np.sum(valid_mask_trad) > 0:
            valid_deaths_trad = death_times_trad[valid_mask_trad]
            stats_text_lines.append("Traditional RLBEEP:")
            stats_text_lines.append(f'  Count: {len(valid_deaths_trad)}')
            stats_text_lines.append(f'  Mean: {np.mean(valid_deaths_trad):.1f}s')
            stats_text_lines.append(f'  Std: {np.std(valid_deaths_trad):.1f}s')
            stats_text_lines.append(f'  Min: {np.min(valid_deaths_trad):.1f}s')
            stats_text_lines.append(f'  Max: {np.max(valid_deaths_trad):.1f}s')
            stats_text_lines.append("")
    
    if df_dql is not None:
        valid_mask_dql = death_times_dql < max_sim_time_dql
        if np.sum(valid_mask_dql) > 0:
            valid_deaths_dql = death_times_dql[valid_mask_dql]
            stats_text_lines.append("DQL-Enhanced RLBEEP:")
            stats_text_lines.append(f'  Count: {len(valid_deaths_dql)}')
            stats_text_lines.append(f'  Mean: {np.mean(valid_deaths_dql):.1f}s')
            stats_text_lines.append(f'  Std: {np.std(valid_deaths_dql):.1f}s')
            stats_text_lines.append(f'  Min: {np.min(valid_deaths_dql):.1f}s')
            stats_text_lines.append(f'  Max: {np.max(valid_deaths_dql):.1f}s')
            stats_text_lines.append("")
    
    # Add improvement calculation
    if (df_traditional is not None and df_dql is not None and 
        np.sum(valid_mask_trad) > 0 and np.sum(valid_mask_dql) > 0):
        mean_trad = np.mean(valid_deaths_trad)
        mean_dql = np.mean(valid_deaths_dql)
        improvement = ((mean_dql - mean_trad) / mean_trad) * 100
        stats_text_lines.append("Performance Improvement:")
        stats_text_lines.append(f'  {improvement:+.1f}% vs Traditional')
    
    if stats_text_lines:
        stats_text = '\n'.join(stats_text_lines)
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.9, edgecolor='black'))
    
    # Tight layout with more padding
    plt.tight_layout(pad=3.0)
    
    # Save the plot
    if not args.no_save:
        if args.output:
            # Modify filename for single plot
            base_name = os.path.splitext(args.output)[0]
            ext = os.path.splitext(args.output)[1] or '.png'
            plot_path = os.path.join(results_dir, f'{base_name}_single{ext}')
        else:
            plot_path = os.path.join(results_dir, 'rlbeep_first_death_comparison.png')
        
        plt.savefig(plot_path, dpi=args.dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Comparison plot saved to: {plot_path}")
    
    # Show the plot
    if args.show:
        plt.show()
    
    return plot_path if not args.no_save else None

def generate_quick_summary(df):
    """
    Generate a quick summary of the epoch results
    
    Args:
        df (pandas.DataFrame): DataFrame containing epoch results
    """
    if df is None or df.empty:
        print("No data available for summary")
        return
    
    print("\n" + "="*50)
    print("  EPOCH RESULTS SUMMARY")
    print("="*50)
    
    # Basic info
    total_epochs = len(df)
    print(f"Total Epochs: {total_epochs}")
    
    if 'first_death_time' in df.columns:
        death_times = df['first_death_time'].values
        max_sim_time = df['simulation_duration'].max() if 'simulation_duration' in df.columns else 80000
        valid_deaths = death_times[death_times < max_sim_time]
        no_death_count = len(death_times[death_times >= max_sim_time])
        
        print(f"Epochs with Node Deaths: {len(valid_deaths)}")
        print(f"Epochs with No Deaths: {no_death_count}")
        
        if len(valid_deaths) > 0:
            print(f"Average First Death Time: {np.mean(valid_deaths):.1f} seconds")
            print(f"Death Time Range: {np.min(valid_deaths):.1f} - {np.max(valid_deaths):.1f} seconds")
            print(f"Standard Deviation: {np.std(valid_deaths):.1f} seconds")
    
    if 'final_live_percentage' in df.columns:
        survival_rates = df['final_live_percentage'].values
        avg_survival = np.mean(survival_rates)
        perfect_survival_count = len(survival_rates[survival_rates == 100.0])
        
        print(f"Average Network Survival: {avg_survival:.1f}%")
        print(f"Perfect Survival Rate: {(perfect_survival_count/total_epochs)*100:.1f}%")
    
    if 'energy_efficiency' in df.columns:
        efficiency_rates = df['energy_efficiency'].values * 100
        avg_efficiency = np.mean(efficiency_rates)
        print(f"Average Energy Efficiency: {avg_efficiency:.1f}%")
    
    print("="*50)

def main():
    """Main function to generate comparison plots from epoch results CSV"""
    args = parse_arguments()
    
    print("RLBEEP Epoch Results Comparison Plotter")
    print("=" * 50)
    
    # Get paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    
    # Ensure results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    
    # Determine CSV paths
    traditional_csv = None
    dql_csv = None
    
    if args.auto_detect or (not args.traditional_csv and not args.dql_csv):
        print("Auto-detecting CSV files...")
        traditional_csv, dql_csv = auto_detect_csv_files(results_dir)
        
        if traditional_csv:
            print(f"Found Traditional RLBEEP CSV: {os.path.basename(traditional_csv)}")
        if dql_csv:
            print(f"Found DQL-Enhanced RLBEEP CSV: {os.path.basename(dql_csv)}")
    else:
        # Use manually specified paths
        if args.traditional_csv:
            if os.path.isabs(args.traditional_csv):
                traditional_csv = args.traditional_csv
            else:
                traditional_csv = os.path.join(results_dir, args.traditional_csv)
        
        if args.dql_csv:
            if os.path.isabs(args.dql_csv):
                dql_csv = args.dql_csv
            else:
                dql_csv = os.path.join(results_dir, args.dql_csv)
    
    # Check if we have at least one CSV file
    if not traditional_csv and not dql_csv:
        print("ERROR: No CSV files found or specified")
        print("Available files in results directory:")
        if os.path.exists(results_dir):
            for f in os.listdir(results_dir):
                if f.endswith('.csv'):
                    print(f"  - {f}")
        sys.exit(1)
    
    print(f"Y-axis Range: {args.ylim}")
    print(f"Output DPI: {args.dpi}")
    
    # Load data
    print("\nLoading data...")
    df_traditional = load_epoch_results(traditional_csv) if traditional_csv else None
    df_dql = load_epoch_results(dql_csv) if dql_csv else None
    
    if df_traditional is None and df_dql is None:
        print("Cannot proceed without valid data")
        sys.exit(1)
    
    # Generate quick summary for both datasets
    if df_traditional is not None:
        print("\n" + "="*30)
        print("  TRADITIONAL RLBEEP SUMMARY")
        print("="*30)
        generate_quick_summary(df_traditional)
    
    if df_dql is not None:
        print("\n" + "="*30)
        print("  DQL-ENHANCED RLBEEP SUMMARY")  
        print("="*30)
        generate_quick_summary(df_dql)
    
    # Create plots
    print("\nGenerating comparison visualizations...")
    
    if args.single_plot:
        # Create single large comparison plot for first death time vs epochs
        print("Creating single comparison plot for First Death Time vs Epochs...")
        plot_path = plot_comparison_first_death_time_only(df_traditional, df_dql, results_dir, args)
        
        if plot_path:
            print(f"\nSingle comparison plot complete!")
            print(f"Plot saved to: {plot_path}")
    else:
        # Create comprehensive 4-panel comparison plot
        plot_path = plot_comparison_analysis(df_traditional, df_dql, results_dir, args)
        
        if plot_path:
            print(f"\nComparison visualization complete!")
            print(f"Plot saved to: {plot_path}")
    
    if args.show:
        print("\nDisplaying plot...")
    else:
        print("\nUse --show flag to display the plot interactively")
    
    print("\nComparison plotting complete!")

if __name__ == "__main__":
    main()
