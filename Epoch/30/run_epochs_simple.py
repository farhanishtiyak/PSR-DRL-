#!/usr/bin/env python3
"""
Simple Epoch Runner for 30-Node RLBEEP
Runs epochs and saves only the first node death time to CSV
"""

import os
import sys
import csv
import time
import importlib.util
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def run_single_epoch(epoch_num, use_dql_enhanced=True):
    """
    Run a single epoch and return first node death time
    
    Args:
        epoch_num (int): Current epoch number
        use_dql_enhanced (bool): Whether to use DQL-enhanced or traditional RLBEEP
        
    Returns:
        dict: Results containing epoch number and first death time
    """
    print(f"Running Epoch {epoch_num}/300...")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, "Dataset")
        
        if use_dql_enhanced:
            # Load and use main.py (Our Proposed DQL-Enhanced)
            main_path = os.path.join(current_dir, "main.py")
            spec = importlib.util.spec_from_file_location("main", main_path)
            main_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main_module)
            
            # Create simulation instance
            simulation = main_module.RLBEEPSimulation(
                dataset_path=dataset_path,
                num_nodes=30,
                num_clusters=12,
                max_longitude=100.0,
                max_latitude=100.0,
                send_range=20,
                alpha=0.5,
                dfr_min=5.0,
                dfr_max=55.0,
                total_time=3000,
                sleep_threshold=10,
                change_threshold=10.0,
                ch_rotation_interval=200,
                debug=False  # Disable debug output
            )
            method = "DQL_Enhanced_RLBEEP"
        else:
            # Load and use simulation.py (Traditional RLBEEP)
            sim_path = os.path.join(current_dir, "simulation.py")
            spec = importlib.util.spec_from_file_location("simulation", sim_path)
            sim_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sim_module)
            
            # Create simulation instance
            simulation = sim_module.RLBEEPSimulation(
                dataset_path=dataset_path,
                num_nodes=30,
                num_clusters=12,
                max_longitude=100.0,
                max_latitude=100.0,
                send_range=20,
                alpha=0.5,
                dfr_min=5.0,
                dfr_max=55.0,
                total_time=3000,
                sleep_threshold=10,
                change_threshold=10.0,
                ch_rotation_interval=200
            )
            method = "Traditional_RLBEEP"
        
        # Run simulation (suppress CSV file creation)
        # Temporarily disable output by redirecting stdout
        import contextlib
        import io
        import glob
        
        # Get current CSV files count
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_files_before = set(glob.glob(os.path.join(current_dir, "*.csv")))
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            results = simulation.run_simulation()
        
        # Remove any new CSV files created by the simulation
        csv_files_after = set(glob.glob(os.path.join(current_dir, "*.csv")))
        new_csv_files = csv_files_after - csv_files_before
        for csv_file in new_csv_files:
            try:
                os.remove(csv_file)
            except:
                pass  # Ignore errors
        
        # Extract first death time
        first_death_time = results.get('first_death_time', -1)
        
        # Handle case where no nodes died
        if first_death_time == -1:
            first_death_time = 3000  # Use max simulation time
        
        return {
            'epoch': epoch_num,
            'first_death_time': first_death_time,
            'method': method
        }
        
    except Exception as e:
        print(f"Error in epoch {epoch_num}: {str(e)}")
        return {
            'epoch': epoch_num,
            'first_death_time': -1,
            'method': method if 'method' in locals() else 'Unknown',
            'error': str(e)
        }

def plot_first_death_times(results_list, method_name, save_path):
    """
    Plot first node death time vs epochs
    
    Args:
        results_list (list): List of epoch result dictionaries
        method_name (str): Name of the method for plot title
        save_path (str): Path to save the plot
    """
    epochs = [r['epoch'] for r in results_list]
    death_times = [r['first_death_time'] for r in results_list if 'error' not in r]
    epoch_nums = [r['epoch'] for r in results_list if 'error' not in r]
    
    if not death_times:
        print("No valid death times to plot")
        return
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot data
    plt.plot(epoch_nums, death_times, 'b-', linewidth=1, alpha=0.7, label='First Death Time')
    plt.scatter(epoch_nums, death_times, c='red', s=20, alpha=0.6)
    
    # Add mean line
    mean_death_time = np.mean(death_times)
    plt.axhline(y=mean_death_time, color='green', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_death_time:.1f}s')
    
    # Formatting
    plt.title(f'{method_name} - First Node Death Time vs Epochs\n30-Node Network', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch Number', fontsize=14)
    plt.ylabel('First Node Death Time (seconds)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set axis limits
    plt.xlim(0, max(epoch_nums) + 10)
    if death_times:
        y_min = min(death_times) * 0.95
        y_max = max(death_times) * 1.05
        plt.ylim(y_min, y_max)
    
    # Add statistics text box
    stats_text = f'Statistics:\n'
    stats_text += f'Total Epochs: {len(death_times)}\n'
    stats_text += f'Mean: {np.mean(death_times):.1f}s\n'
    stats_text += f'Std Dev: {np.std(death_times):.1f}s\n'
    stats_text += f'Min: {np.min(death_times):.1f}s\n'
    stats_text += f'Max: {np.max(death_times):.1f}s'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to: {save_path}")

def save_results_to_csv(results_list, filename):
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
    fieldnames = ['epoch', 'first_death_time', 'method']
    if any('error' in result for result in results_list):
        fieldnames.append('error')
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)
    
    print(f"Results saved to: {csv_path}")
    return csv_path

def run_epochs_dql_enhanced():
    """Run 300 epochs of DQL-Enhanced RLBEEP and save first death times"""
    print("="*60)
    print("   DQL-ENHANCED RLBEEP EPOCH RUNNER")
    print("   30-Node Wireless Sensor Network")
    print("="*60)
    
    results_list = []
    start_time = datetime.now()
    
    # Run 300 epochs
    for epoch in range(1, 301):
        epoch_result = run_single_epoch(epoch, use_dql_enhanced=True)
        results_list.append(epoch_result)
        
        # Progress update every 50 epochs
        if epoch % 50 == 0:
            print(f"Progress: {epoch}/300 epochs completed ({epoch/300*100:.1f}%)")
    
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print(f"\n✓ All 300 epochs completed!")
    print(f"Total execution time: {total_time}")
    print(f"Average time per epoch: {total_time.total_seconds()/300:.1f} seconds")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"dql_enhanced_rlbeep_first_death_times_30nodes_{timestamp}.csv"
    save_results_to_csv(results_list, csv_filename)
    
    # Generate plot
    print("\nGenerating visualization...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    plot_filename = f"dql_enhanced_first_death_vs_epochs_{timestamp}.png"
    plot_path = os.path.join(results_dir, plot_filename)
    
    plot_first_death_times(results_list, "DQL-Enhanced RLBEEP", plot_path)
    
    # Summary statistics
    death_times = [r['first_death_time'] for r in results_list if r['first_death_time'] != -1]
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
    
    print("="*60)

def run_epochs_traditional():
    """Run 300 epochs of Traditional RLBEEP and save first death times"""
    print("="*60)
    print("   TRADITIONAL RLBEEP EPOCH RUNNER")
    print("   30-Node Wireless Sensor Network")
    print("="*60)
    
    results_list = []
    start_time = datetime.now()
    
    # Run 300 epochs
    for epoch in range(1, 301):
        epoch_result = run_single_epoch(epoch, use_dql_enhanced=False)
        results_list.append(epoch_result)
        
        # Progress update every 50 epochs
        if epoch % 50 == 0:
            print(f"Progress: {epoch}/300 epochs completed ({epoch/300*100:.1f}%)")
    
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print(f"\n✓ All 300 epochs completed!")
    print(f"Total execution time: {total_time}")
    print(f"Average time per epoch: {total_time.total_seconds()/300:.1f} seconds")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"traditional_rlbeep_first_death_times_30nodes_{timestamp}.csv"
    save_results_to_csv(results_list, csv_filename)
    
    # Generate plot
    print("\nGenerating visualization...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    plot_filename = f"traditional_first_death_vs_epochs_{timestamp}.png"
    plot_path = os.path.join(results_dir, plot_filename)
    
    plot_first_death_times(results_list, "Traditional RLBEEP", plot_path)
    
    # Summary statistics
    death_times = [r['first_death_time'] for r in results_list if r['first_death_time'] != -1]
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
    
    print("="*60)

def main():
    """Main function to run epochs"""
    if len(sys.argv) < 2:
        print("Usage: python run_epochs_simple.py [dql|traditional|both]")
        print("  dql        - Run DQL-Enhanced RLBEEP epochs")
        print("  traditional - Run Traditional RLBEEP epochs")
        print("  both       - Run both methods")
        sys.exit(1)
    
    method = sys.argv[1].lower()
    
    if method == "dql":
        run_epochs_dql_enhanced()
    elif method == "traditional":
        run_epochs_traditional()
    elif method == "both":
        print("Running both DQL-Enhanced and Traditional RLBEEP...")
        run_epochs_dql_enhanced()
        print("\n" + "="*80 + "\n")
        run_epochs_traditional()
    else:
        print("Invalid method. Use 'dql', 'traditional', or 'both'")
        sys.exit(1)

if __name__ == "__main__":
    main()
