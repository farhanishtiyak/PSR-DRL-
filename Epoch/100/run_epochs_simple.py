#!/usr/bin/env python3
"""
Simple Epoch Runner for 100-Node RLBEEP
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
import contextlib
import glob

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

        # Suppress stdout during simulation execution
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                if use_dql_enhanced:
                    # Import and run DQL-Enhanced RLBEEP
                    spec = importlib.util.spec_from_file_location("main", os.path.join(current_dir, "main.py"))
                    main_module = importlib.util.module_from_spec(spec)
                    sys.modules["main"] = main_module
                    spec.loader.exec_module(main_module)
                    
                    # Run simulation
                    simulation = main_module.RLBEEPSimulation(
                        dataset_path=dataset_path,
                        num_nodes=100,
                        num_clusters=40,
                        max_longitude=200.0,
                        max_latitude=200.0,
                        send_range=30,
                        total_time=3000,
                        sleep_threshold=10,
                        change_threshold=10.0,
                        ch_rotation_interval=300
                    )
                    
                    results = simulation.run_simulation()
                    first_death_time = results.get('first_death_time', -1)
                    method = "DQL_Enhanced_RLBEEP"
                    
                else:
                    # Import and run Traditional RLBEEP
                    spec = importlib.util.spec_from_file_location("simulation", os.path.join(current_dir, "simulation.py"))
                    sim_module = importlib.util.module_from_spec(spec)
                    sys.modules["simulation"] = sim_module
                    spec.loader.exec_module(sim_module)
                    
                    # Run simulation
                    simulation = sim_module.RLBEEPSimulation(
                        dataset_path=dataset_path,
                        num_nodes=100,
                        num_clusters=40,
                        max_longitude=200.0,
                        max_latitude=200.0,
                        send_range=30,
                        alpha=0.5,
                        dfr_min=5.0,
                        dfr_max=55.0,
                        total_time=3000,
                        sleep_threshold=10,
                        change_threshold=10.0,
                        ch_rotation_interval=300
                    )
                    
                    results = simulation.run_simulation()
                    first_death_time = results.get('first_death_time', -1)
                    method = "Traditional_RLBEEP"
        
        # Clean up any unwanted CSV files that might have been created during simulation
        cleanup_unwanted_files()
        
        return {
            'epoch': epoch_num,
            'first_death_time': first_death_time,
            'method': method
        }
        
    except Exception as e:
        print(f"Error in epoch {epoch_num}: {e}")
        return {
            'epoch': epoch_num,
            'first_death_time': -1,
            'method': "DQL_Enhanced_RLBEEP" if use_dql_enhanced else "Traditional_RLBEEP"
        }

def cleanup_unwanted_files():
    """Remove unwanted CSV files created during individual simulations"""
    try:
        # Look for simulation data/summary files in current directory
        unwanted_patterns = [
            "rlbeep_simulation_data.csv",
            "rlbeep_simulation_summary.csv",
            "*simulation_data.csv",
            "*simulation_summary.csv"
        ]
        
        for pattern in unwanted_patterns:
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                    print(f"Cleaned up: {file_path}")
                except Exception:
                    pass  # Ignore cleanup errors
    except Exception:
        pass  # Ignore all cleanup errors

def save_results_to_csv(results, filename):
    """Save results to CSV file"""
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'first_death_time', 'method']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to: {filepath}")
    return filepath

def plot_first_death_times(results, method_name):
    """Plot first node death times vs epochs"""
    epochs = [r['epoch'] for r in results]
    death_times = [r['first_death_time'] for r in results if r['first_death_time'] > 0]
    valid_epochs = [r['epoch'] for r in results if r['first_death_time'] > 0]
    
    if not death_times:
        print(f"No valid death times found for {method_name}")
        return
    
    # Calculate statistics
    mean_death_time = np.mean(death_times)
    std_death_time = np.std(death_times)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the data
    plt.plot(valid_epochs, death_times, 'b-' if 'DQL' in method_name else 'g-', 
             linewidth=1, alpha=0.7, label=f'{method_name}')
    plt.scatter(valid_epochs, death_times, 
                c='blue' if 'DQL' in method_name else 'green', s=10, alpha=0.5)
    
    # Add mean line
    plt.axhline(y=mean_death_time, color='red', linestyle='--', 
                label=f'Mean: {mean_death_time:.1f}s')
    
    # Fill area between mean ± std
    plt.fill_between(valid_epochs, mean_death_time - std_death_time, 
                     mean_death_time + std_death_time, 
                     alpha=0.2, color='gray', label=f'±1 STD: {std_death_time:.1f}s')
    
    # Customize the plot
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('First Node Death Time (seconds)', fontsize=12)
    plt.title(f'First Node Death Time vs Epoch\n{method_name} - 100 Nodes', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add statistics text box
    stats_text = f"""Statistics (100 Nodes):
Mean: {mean_death_time:.1f}s
Std: {std_death_time:.1f}s
Min: {min(death_times)}s
Max: {max(death_times)}s
Valid epochs: {len(death_times)}/300"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"results/{method_name.lower().replace(' ', '_')}_100nodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    
    plt.close()  # Close the figure to free memory

def run_dql_enhanced_epochs():
    """Run 300 epochs using DQL-Enhanced RLBEEP"""
    print("Starting DQL-Enhanced RLBEEP Epochs (100 Nodes)...")
    print("="*60)
    
    results = []
    start_time = time.time()
    
    for epoch in range(1, 301):
        result = run_single_epoch(epoch, use_dql_enhanced=True)
        results.append(result)
        
        # Progress update every 50 epochs
        if epoch % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Completed {epoch}/300 epochs in {elapsed/60:.1f} minutes")
            if result['first_death_time'] > 0:
                print(f"Latest first death time: {result['first_death_time']} seconds")
    
    # Save results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"dql_enhanced_rlbeep_first_death_times_100nodes_{timestamp}.csv"
    csv_path = save_results_to_csv(results, filename)
    
    # Create plot
    plot_first_death_times(results, "DQL_Enhanced_RLBEEP")
    
    # Calculate and print statistics
    valid_results = [r for r in results if r['first_death_time'] > 0]
    if valid_results:
        death_times = [r['first_death_time'] for r in valid_results]
        print(f"\nDQL-Enhanced RLBEEP Results (100 Nodes):")
        print(f"Mean first death time: {np.mean(death_times):.1f} seconds")
        print(f"Standard deviation: {np.std(death_times):.1f} seconds")
        print(f"Valid epochs: {len(valid_results)}/300")
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time/60:.1f} minutes")
    print("="*60)
    
    return results

def run_traditional_epochs():
    """Run 300 epochs using Traditional RLBEEP"""
    print("Starting Traditional RLBEEP Epochs (100 Nodes)...")
    print("="*60)
    
    results = []
    start_time = time.time()
    
    for epoch in range(1, 301):
        result = run_single_epoch(epoch, use_dql_enhanced=False)
        results.append(result)
        
        # Progress update every 50 epochs
        if epoch % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Completed {epoch}/300 epochs in {elapsed/60:.1f} minutes")
            if result['first_death_time'] > 0:
                print(f"Latest first death time: {result['first_death_time']} seconds")
    
    # Save results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"traditional_rlbeep_first_death_times_100nodes_{timestamp}.csv"
    csv_path = save_results_to_csv(results, filename)
    
    # Create plot
    plot_first_death_times(results, "Traditional_RLBEEP")
    
    # Calculate and print statistics
    valid_results = [r for r in results if r['first_death_time'] > 0]
    if valid_results:
        death_times = [r['first_death_time'] for r in valid_results]
        print(f"\nTraditional RLBEEP Results (100 Nodes):")
        print(f"Mean first death time: {np.mean(death_times):.1f} seconds")
        print(f"Standard deviation: {np.std(death_times):.1f} seconds")
        print(f"Valid epochs: {len(valid_results)}/300")
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time/60:.1f} minutes")
    print("="*60)
    
    return results

def main():
    """Main function to run epochs"""
    print("RLBEEP Epoch Runner - 100 Nodes")
    print("This will run 300 epochs for each method")
    print("Choose which method to run:")
    print("1. DQL-Enhanced RLBEEP only")
    print("2. Traditional RLBEEP only")
    print("3. Both methods")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        run_dql_enhanced_epochs()
    elif choice == "2":
        run_traditional_epochs()
    elif choice == "3":
        print("Running both methods sequentially...")
        run_dql_enhanced_epochs()
        print("\n" + "="*60 + "\n")
        run_traditional_epochs()
    else:
        print("Invalid choice. Running DQL-Enhanced RLBEEP by default.")
        run_dql_enhanced_epochs()

if __name__ == "__main__":
    main()
