#!/usr/bin/env python3
"""
Script to compare PSR-DRL vs RLBEEP vs EER-RL (50 Nodes)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_comparison(dql_csv, traditional_csv):
    """
    Plot comparison between PSR-DRL and RLBEEP
    """
    # Read both CSV files
    df_dql = pd.read_csv(dql_csv)
    df_traditional = pd.read_csv(traditional_csv)
    
    # Extract data
    epochs_dql = df_dql['epoch']
    death_times_dql = df_dql['first_death_time']
    
    epochs_traditional = df_traditional['epoch']
    death_times_traditional = df_traditional['first_death_time']
    
    # Calculate statistics
    mean_dql = np.mean(death_times_dql)
    std_dql = np.std(death_times_dql)
    mean_traditional = np.mean(death_times_traditional)
    std_traditional = np.std(death_times_traditional)
    
    # Create comparison plot
    plt.figure(figsize=(14, 10))
    
    # Plot PSR-DRL
    plt.subplot(2, 1, 1)
    plt.plot(epochs_dql, death_times_dql, 'b-', linewidth=1, alpha=0.7, label='PSR-DRL')
    plt.scatter(epochs_dql, death_times_dql, c='blue', s=8, alpha=0.5)
    plt.fill_between(epochs_dql, mean_dql - std_dql, mean_dql + std_dql, 
                     alpha=0.2, color='blue', label=f'±1 STD: {std_dql:.1f}s')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('First Node Death Time (s)', fontsize=14)
    plt.title('PSR-DRL - 50 Nodes', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Plot RLBEEP
    plt.subplot(2, 1, 2)
    plt.plot(epochs_traditional, death_times_traditional, 'g-', linewidth=1, alpha=0.7, label='RLBEEP')
    plt.scatter(epochs_traditional, death_times_traditional, c='green', s=8, alpha=0.5)
    plt.fill_between(epochs_traditional, mean_traditional - std_traditional, 
                     mean_traditional + std_traditional, 
                     alpha=0.2, color='green', label=f'±1 STD: {std_traditional:.1f}s')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('First Node Death Time (s)', fontsize=14)
    plt.title('RLBEEP - 50 Nodes', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_file = 'results/comparison_dql_vs_traditional_50nodes.png'
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved as: {comparison_file}")
    plt.show()
    
    # Create side-by-side comparison
    plt.figure(figsize=(14, 8))
    
    plt.plot(epochs_dql, death_times_dql, 'b-', linewidth=1, alpha=0.7, label='PSR-DRL')
    plt.plot(epochs_traditional, death_times_traditional, 'g-', linewidth=1, alpha=0.7, label='RLBEEP')
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('First Node Death Time (seconds)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper right')
    
    # Calculate improvement for stats
    improvement = ((mean_dql - mean_traditional) / mean_traditional) * 100
    
    plt.tight_layout()
    
    # Save overlay comparison
    overlay_file = 'results/overlay_comparison_50nodes.png'
    plt.savefig(overlay_file, dpi=300, bbox_inches='tight')
    print(f"Overlay comparison saved as: {overlay_file}")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY - 50 Nodes (300 Epochs)")
    print("="*60)
    print(f"PSR-DRL:")
    print(f"  Mean: {mean_dql:.1f} seconds")
    print(f"  Std:  {std_dql:.1f} seconds")
    print(f"  Min:  {np.min(death_times_dql)} seconds")
    print(f"  Max:  {np.max(death_times_dql)} seconds")
    print()
    print(f"RLBEEP:")
    print(f"  Mean: {mean_traditional:.1f} seconds")
    print(f"  Std:  {std_traditional:.1f} seconds")
    print(f"  Min:  {np.min(death_times_traditional)} seconds")
    print(f"  Max:  {np.max(death_times_traditional)} seconds")
    print()
    print(f"Performance Improvement: {improvement:.1f}%")
    print(f"Network Lifetime Extension: {mean_dql - mean_traditional:.1f} seconds")
    print("="*60)

def plot_comparison_with_eerrl(dql_csv, traditional_csv, eerrl_csv):
    """
    Plot comparison between DQL-Enhanced, Traditional RLBEEP, and EER-RL
    """
    # Read all CSV files
    df_dql = pd.read_csv(dql_csv)
    df_traditional = pd.read_csv(traditional_csv)
    
    # Read EER-RL data with different column names
    df_eerrl = pd.read_csv(eerrl_csv)
    # Rename columns to match expected format if needed
    if 'Epoch' in df_eerrl.columns:
        df_eerrl = df_eerrl.rename(columns={'Epoch': 'epoch', 'FirstNodeDeathTime': 'first_death_time'})
    
    # Extract data
    epochs_dql = df_dql['epoch']
    death_times_dql = df_dql['first_death_time']
    
    epochs_traditional = df_traditional['epoch']
    death_times_traditional = df_traditional['first_death_time']
    
    epochs_eerrl = df_eerrl['epoch']
    death_times_eerrl = df_eerrl['first_death_time']
    
    # Calculate statistics
    mean_dql = np.mean(death_times_dql)
    std_dql = np.std(death_times_dql)
    mean_traditional = np.mean(death_times_traditional)
    std_traditional = np.std(death_times_traditional)
    mean_eerrl = np.mean(death_times_eerrl)
    std_eerrl = np.std(death_times_eerrl)
    
    # Create side-by-side comparison with all three methods
    plt.figure(figsize=(14, 8))
    
    plt.plot(epochs_dql, death_times_dql, 'b-', linewidth=1, alpha=0.7, label='PSR-DRL')
    plt.plot(epochs_traditional, death_times_traditional, 'g-', linewidth=1, alpha=0.7, label='RLBEEP')
    plt.plot(epochs_eerrl, death_times_eerrl, 'r-', linewidth=1, alpha=0.7, label='EER-RL')
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('First Node Death Time (seconds)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper right')
    
    # Add improvement statistics
    improvement_dql = ((mean_dql - mean_traditional) / mean_traditional) * 100
    improvement_eerrl_vs_trad = ((mean_eerrl - mean_traditional) / mean_traditional) * 100
    improvement_eerrl_vs_dql = ((mean_eerrl - mean_dql) / mean_dql) * 100
    
    plt.tight_layout()
    
    # Save overlay comparison with all three methods
    overlay_file = 'results/overlay_comparison_all_methods_50nodes.png'
    overlay_file_pdf = 'results/overlay_comparison_all_methods_50nodes.pdf'
    plt.savefig(overlay_file, dpi=300, bbox_inches='tight')
    plt.savefig(overlay_file_pdf, dpi=300, bbox_inches='tight')
    print(f"Overlay comparison with EER-RL saved as: {overlay_file}")
    print(f"Overlay comparison with EER-RL saved as PDF: {overlay_file_pdf}")
    plt.show()
    
    # Print summary statistics including EER-RL
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY - 50 Nodes (300 Epochs)")
    print("="*60)
    print(f"PSR-DRL:")
    print(f"  Mean: {mean_dql:.1f} seconds")
    print(f"  Std:  {std_dql:.1f} seconds")
    print(f"  Min:  {np.min(death_times_dql)} seconds")
    print(f"  Max:  {np.max(death_times_dql)} seconds")
    print()
    print(f"RLBEEP:")
    print(f"  Mean: {mean_traditional:.1f} seconds")
    print(f"  Std:  {std_traditional:.1f} seconds")
    print(f"  Min:  {np.min(death_times_traditional)} seconds")
    print(f"  Max:  {np.max(death_times_traditional)} seconds")
    print()
    print(f"EER-RL:")
    print(f"  Mean: {mean_eerrl:.1f} seconds")
    print(f"  Std:  {std_eerrl:.1f} seconds")
    print(f"  Min:  {np.min(death_times_eerrl)} seconds")
    print(f"  Max:  {np.max(death_times_eerrl)} seconds")
    print()
    print(f"Improvement (PSR-DRL vs RLBEEP): {improvement_dql:.1f}%")
    print(f"Improvement (EER-RL vs RLBEEP): {improvement_eerrl_vs_trad:.1f}%")
    print(f"Improvement (EER-RL vs PSR-DRL): {improvement_eerrl_vs_dql:.1f}%")
    print("="*60)

def main():
    """
    Main function to create comparison plots
    """
    results_dir = 'results'
    
    # Find the CSV files
    dql_file = None
    traditional_file = None
    eerrl_file = None
    
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if '50nodes' in file and file.endswith('.csv'):
                if 'dql_enhanced' in file:
                    dql_file = os.path.join(results_dir, file)
                elif 'traditional' in file:
                    traditional_file = os.path.join(results_dir, file)
                elif 'eer_rl' in file:
                    eerrl_file = os.path.join(results_dir, file)
    
    # Check if all files are found
    if dql_file and traditional_file and eerrl_file:
        print(f"DQL File: {dql_file}")
        print(f"Traditional File: {traditional_file}")
        print(f"EER-RL File: {eerrl_file}")
        plot_comparison_with_eerrl(dql_file, traditional_file, eerrl_file)
        plot_comparison(dql_file, traditional_file)
    # If EER-RL file is missing but the others are present
    elif dql_file and traditional_file:
        print(f"DQL File: {dql_file}")
        print(f"Traditional File: {traditional_file}")
        print(f"EER-RL File not found. Running standard comparison only.")
        plot_comparison(dql_file, traditional_file)
    else:
        print("Could not find required CSV files for comparison!")
        if dql_file:
            print(f"Found DQL file: {dql_file}")
        if traditional_file:
            print(f"Found Traditional file: {traditional_file}")
        if eerrl_file:
            print(f"Found EER-RL file: {eerrl_file}")

if __name__ == "__main__":
    main()
