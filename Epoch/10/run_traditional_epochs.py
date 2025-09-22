#!/usr/bin/env python3
"""
RLBEEP Competitor Epoch Runner - 10 Nodes
==========================================

This script runs the traditional RLBEEP simulation for 300 epochs and collects
first node death time data for analysis.

Author: Research Team
Date: July 2025
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import importlib.util
import csv

# Import the simulation module
def load_simulation_module():
    """Load the simulation.py module dynamically"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    simulation_path = os.path.join(current_dir, 'simulation.py')
    
    spec = importlib.util.spec_from_file_location("simulation", simulation_path)
    simulation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(simulation_module)
    
    return simulation_module

def run_epoch_simulation(epoch_num, simulation_module):
    """
    Run a single epoch of the traditional RLBEEP simulation
    
    Args:
        epoch_num (int): Current epoch number
        simulation_module: The loaded simulation module
        
    Returns:
        dict: Results from this epoch
    """
    print(f"Running Traditional RLBEEP Epoch {epoch_num}/300...")
    
    try:
        # Set paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, "Dataset")
        
        # Create simulation instance
        simulation = simulation_module.RLBEEPSimulation(
            dataset_path=dataset_path,
            num_nodes=10,
            num_clusters=4,
            max_longitude=60.0,
            max_latitude=60.0,
            send_range=10,
            alpha=0.5,
            dfr_min=5.0,
            dfr_max=55.0,
            total_time=3000,
            sleep_threshold=10,
            change_threshold=10.0,
            ch_rotation_interval=200
        )
        
        # Run simulation
        results = simulation.run_simulation()
        
        # Extract key metrics
        first_death_time = results.get('first_death_time', -1)
        final_live_percentage = results.get('final_live_percentage', 0.0)
        
        # Handle case where no nodes died
        if first_death_time == -1:
            first_death_time = 3000  # Use max simulation time
        
        epoch_results = {
            'epoch': epoch_num,
            'first_death_time': first_death_time,
            'final_live_percentage': final_live_percentage,
            'simulation_duration': 3000,
            'method': 'Traditional_RLBEEP'
        }
        
        print(f"Epoch {epoch_num} completed - First death: {first_death_time}s")
        return epoch_results
        
    except Exception as e:
        print(f"Error in epoch {epoch_num}: {e}")
        return {
            'epoch': epoch_num,
            'first_death_time': 3000,  # Default to max time on error
            'final_live_percentage': 0.0,
            'simulation_duration': 3000,
            'method': 'Traditional_RLBEEP',
            'error': str(e)
        }

def save_epoch_results(results_list, filename):
    """
    Save epoch results to CSV file
    
    Args:
        results_list (list): List of epoch result dictionaries
        filename (str): Output CSV filename
    """
    if not results_list:
        print("No results to save")
        return
    
    # Create results directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    csv_path = os.path.join(results_dir, filename)
    
    # Define fieldnames
    fieldnames = ['epoch', 'first_death_time', 'final_live_percentage', 'simulation_duration', 'method']
    if any('error' in result for result in results_list):
        fieldnames.append('error')
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)
    
    print(f"Results saved to: {csv_path}")
    return csv_path

def plot_first_death_time_epochs(results_list, save_path):
    """
    Plot first node death time vs epochs
    
    Args:
        results_list (list): List of epoch result dictionaries
        save_path (str): Path to save the plot
    """
    if not results_list:
        print("No results to plot")
        return
    
    # Extract data
    epochs = [r['epoch'] for r in results_list]
    death_times = [r['first_death_time'] for r in results_list]
    
    # Create figure
    plt.figure(figsize=(16, 8))
    
    # Plot data
    plt.plot(epochs, death_times, 'b-', alpha=0.8, linewidth=2, label='First Death Time')
    plt.scatter(epochs, death_times, c='red', s=35, alpha=0.8, zorder=5, edgecolors='darkred', linewidth=0.5)
    
    # Formatting
    plt.xlabel('Epoch Number', fontsize=16, fontweight='bold')
    plt.ylabel('First Node Death Time (seconds)', fontsize=16, fontweight='bold')
    plt.title('Traditional RLBEEP - First Node Death Time vs Epochs\n10-Node Network', fontsize=18, fontweight='bold', pad=20)
    
    # Grid
    plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.2, linestyle='--', linewidth=0.3, which='minor')
    
    # Set limits and ticks
    plt.xlim(0, max(epochs) * 1.02)
    plt.ylim(min(death_times) * 0.95, max(death_times) * 1.05)
    
    # Major and minor ticks
    if max(epochs) <= 50:
        major_ticks = np.arange(0, max(epochs) + 1, 5)
        minor_ticks = np.arange(0, max(epochs) + 1, 1)
    elif max(epochs) <= 100:
        major_ticks = np.arange(0, max(epochs) + 1, 10)
        minor_ticks = np.arange(0, max(epochs) + 1, 2)
    else:
        major_ticks = np.arange(0, max(epochs) + 1, 25)
        minor_ticks = np.arange(0, max(epochs) + 1, 5)
    
    plt.xticks(major_ticks)
    plt.xticks(minor_ticks, minor=True)
    
    # Tick params
    plt.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    plt.tick_params(axis='both', which='minor', labelsize=10, width=1, length=3)
    
    # Statistics
    valid_deaths = [d for d in death_times if d < 3000]  # Exclude max time defaults
    if valid_deaths:
        mean_death = np.mean(valid_deaths)
        std_death = np.std(valid_deaths)
        min_death = np.min(valid_deaths)
        max_death = np.max(valid_deaths)
        
        stats_text = (f'Total Epochs: {len(epochs)}\n'
                     f'Valid Deaths: {len(valid_deaths)}\n'
                     f'Mean: {mean_death:.1f}s\n'
                     f'Std Dev: {std_death:.1f}s\n'
                     f'Min: {min_death:.1f}s\n'
                     f'Max: {max_death:.1f}s')
        
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.9, edgecolor='black'))
    
    # Method info
    method_info = f'Method: Traditional RLBEEP\nNodes: 10\nTotal Time: 3000s'
    plt.text(0.02, 0.98, method_info, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, edgecolor='navy'))
    
    plt.tight_layout(pad=3.0)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Plot saved to: {save_path}")
    
    plt.show()

def main():
    """Main function to run 300 epochs of traditional RLBEEP simulation"""
    print("="*60)
    print("   TRADITIONAL RLBEEP - 300 EPOCH ANALYSIS")
    print("   10-Node Wireless Sensor Network")
    print("="*60)
    
    # Load simulation module
    try:
        simulation_module = load_simulation_module()
        print("✓ Simulation module loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load simulation module: {e}")
        sys.exit(1)
    
    # Run epochs
    results_list = []
    start_time = datetime.now()
    
    print(f"\nStarting 300-epoch simulation at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    for epoch in range(1, 301):
        epoch_result = run_epoch_simulation(epoch, simulation_module)
        results_list.append(epoch_result)
        
        # Progress update every 25 epochs
        if epoch % 25 == 0:
            elapsed = datetime.now() - start_time
            avg_time_per_epoch = elapsed.total_seconds() / epoch
            estimated_remaining = (300 - epoch) * avg_time_per_epoch
            
            print(f"Progress: {epoch}/300 epochs completed ({epoch/300*100:.1f}%)")
            print(f"Elapsed time: {elapsed}")
            print(f"Estimated remaining: {estimated_remaining//60:.0f}m {estimated_remaining%60:.0f}s")
            print("-" * 60)
    
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print(f"\n✓ All 300 epochs completed!")
    print(f"Total execution time: {total_time}")
    print(f"Average time per epoch: {total_time.total_seconds()/300:.1f} seconds")
    
    # Save results
    print("\nSaving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"traditional_rlbeep_epochs_300_{timestamp}.csv"
    csv_path = save_epoch_results(results_list, csv_filename)
    
    # Generate plot
    print("\nGenerating visualization...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    plot_filename = f"traditional_rlbeep_first_death_vs_epochs_{timestamp}.png"
    plot_path = os.path.join(results_dir, plot_filename)
    
    plot_first_death_time_epochs(results_list, plot_path)
    
    # Summary statistics
    death_times = [r['first_death_time'] for r in results_list]
    valid_deaths = [d for d in death_times if d < 3000]
    
    print("\n" + "="*60)
    print("   EXECUTION SUMMARY")
    print("="*60)
    print(f"Total Epochs: 300")
    print(f"Epochs with Node Deaths: {len(valid_deaths)}")
    print(f"Epochs with No Deaths: {300 - len(valid_deaths)}")
    
    if valid_deaths:
        print(f"Average First Death Time: {np.mean(valid_deaths):.1f} seconds")
        print(f"Death Time Range: {np.min(valid_deaths):.1f} - {np.max(valid_deaths):.1f} seconds")
        print(f"Standard Deviation: {np.std(valid_deaths):.1f} seconds")
    
    print(f"\nResults saved to: {csv_path}")
    print(f"Plot saved to: {plot_path}")
    print("="*60)

if __name__ == "__main__":
    main()
