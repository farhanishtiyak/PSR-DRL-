#!/usr/bin/env python3
"""
Simple script to plot first node death times from CSV file
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_first_death_times(csv_file):
    """
    Plot first node death times vs epochs from CSV file
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract data
    epochs = df['epoch']
    first_death_times = df['first_death_time']
    method = df['method'].iloc[0]  # Get the method name
    
    # Calculate statistics
    mean_death_time = np.mean(first_death_times)
    std_death_time = np.std(first_death_times)
    min_death_time = np.min(first_death_times)
    max_death_time = np.max(first_death_times)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the data
    plt.plot(epochs, first_death_times, 'b-', linewidth=1, alpha=0.7, label=f'{method}')
    plt.scatter(epochs, first_death_times, c='blue', s=10, alpha=0.5)
    
    # Add mean line
    plt.axhline(y=mean_death_time, color='red', linestyle='--', 
                label=f'Mean: {mean_death_time:.1f}s')
    
    # Fill area between mean ± std
    plt.fill_between(epochs, mean_death_time - std_death_time, 
                     mean_death_time + std_death_time, 
                     alpha=0.2, color='gray', label=f'±1 STD: {std_death_time:.1f}s')
    
    # Customize the plot
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('First Node Death Time (seconds)', fontsize=12)
    plt.title(f'First Node Death Time vs Epoch\n{method} - 30 Nodes', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add statistics text box
    stats_text = f"""Statistics:
Mean: {mean_death_time:.1f}s
Std: {std_death_time:.1f}s
Min: {min_death_time}s
Max: {max_death_time}s
Range: {max_death_time - min_death_time}s"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = csv_file.replace('.csv', '_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")
    
    # Show the plot
    plt.show()
    
    return mean_death_time, std_death_time

def main():
    """
    Main function to run the plotting script
    """
    # Get the current directory
    current_dir = os.getcwd()
    results_dir = os.path.join(current_dir, 'results')
    
    # Look for CSV files in results directory
    csv_files = []
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.csv') and 'first_death_times' in file:
                csv_files.append(os.path.join(results_dir, file))
    
    if not csv_files:
        print("No CSV files found in results directory!")
        print("Please provide the path to your CSV file:")
        csv_file = input().strip()
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            return
        csv_files = [csv_file]
    
    # Plot each CSV file
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        try:
            mean_time, std_time = plot_first_death_times(csv_file)
            print(f"Mean first death time: {mean_time:.1f} ± {std_time:.1f} seconds")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    main()
