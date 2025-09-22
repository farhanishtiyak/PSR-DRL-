#!/usr/bin/env python3
"""
Comparison script for 30, 50, and 100 node simulations
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def compare_simulations():
    """Compare the three simulation results"""
    
    # Load summary data
    sim_30 = pd.read_csv('30/rlbeep_simulation_summary.csv')
    sim_50 = pd.read_csv('50/rlbeep_simulation_summary.csv')
    sim_100 = pd.read_csv('100/rlbeep_simulation_summary.csv')
    
    # Convert to dictionaries for easier access
    data_30 = dict(zip(sim_30['Parameter'], sim_30['Value']))
    data_50 = dict(zip(sim_50['Parameter'], sim_50['Value']))
    data_100 = dict(zip(sim_100['Parameter'], sim_100['Value']))
    
    # Create comparison table
    print("=== RLBEEP SIMULATION COMPARISON ===\n")
    print(f"{'Metric':<20} {'30 Nodes':<12} {'50 Nodes':<12} {'100 Nodes':<12}")
    print("-" * 60)
    print(f"{'Nodes':<20} {data_30['Num_Nodes']:<12} {data_50['Num_Nodes']:<12} {data_100['Num_Nodes']:<12}")
    print(f"{'Cluster Heads':<20} {data_30['Num_Clusters']:<12} {data_50['Num_Clusters']:<12} {data_100['Num_Clusters']:<12}")
    print(f"{'Network Area':<20} {data_30['Network_Area']:<12} {data_50['Network_Area']:<12} {data_100['Network_Area']:<12}")
    print(f"{'Send Range (m)':<20} {data_30['Send_Range']:<12} {data_50['Send_Range']:<12} {data_100['Send_Range']:<12}")
    print(f"{'First Death (s)':<20} {data_30['First_Death_Time']:<12} {data_50['First_Death_Time']:<12} {data_100['First_Death_Time']:<12}")
    
    # Load time-series data for plotting
    try:
        ts_30 = pd.read_csv('30/rlbeep_simulation_data.csv')
        ts_50 = pd.read_csv('50/rlbeep_simulation_data.csv')
        ts_100 = pd.read_csv('100/rlbeep_simulation_data.csv')
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Live Node Percentage
        ax1.plot(ts_30['Time'], ts_30['Live_Node_Percentage'], 'b-', label='30 Nodes', linewidth=2)
        ax1.plot(ts_50['Time'], ts_50['Live_Node_Percentage'], 'r-', label='50 Nodes', linewidth=2)
        ax1.plot(ts_100['Time'], ts_100['Live_Node_Percentage'], 'g-', label='100 Nodes', linewidth=2)
        ax1.set_title('Live Node Percentage Comparison')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Live Node Percentage (%)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Total Network Energy
        ax2.plot(ts_30['Time'], ts_30['Total_Energy'], 'b-', label='30 Nodes', linewidth=2)
        ax2.plot(ts_50['Time'], ts_50['Total_Energy'], 'r-', label='50 Nodes', linewidth=2)
        ax2.plot(ts_100['Time'], ts_100['Total_Energy'], 'g-', label='100 Nodes', linewidth=2)
        ax2.set_title('Total Network Energy Comparison')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Total Energy')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Packets Received at Sink
        ax3.plot(ts_30['Time'], ts_30['Packets_Received'], 'b-', label='30 Nodes', linewidth=2)
        ax3.plot(ts_50['Time'], ts_50['Packets_Received'], 'r-', label='50 Nodes', linewidth=2)
        ax3.plot(ts_100['Time'], ts_100['Packets_Received'], 'g-', label='100 Nodes', linewidth=2)
        ax3.set_title('Packets Received at Sink Comparison')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Packets Received')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Active Nodes
        ax4.plot(ts_30['Time'], ts_30['Active_Nodes'], 'b-', label='30 Nodes', linewidth=2)
        ax4.plot(ts_50['Time'], ts_50['Active_Nodes'], 'r-', label='50 Nodes', linewidth=2)
        ax4.plot(ts_100['Time'], ts_100['Active_Nodes'], 'g-', label='100 Nodes', linewidth=2)
        ax4.set_title('Active Nodes Comparison')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Number of Active Nodes')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('simulation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\nComparison plots saved to simulation_comparison.png")
        
    except Exception as e:
        print(f"Could not create plots: {e}")

if __name__ == "__main__":
    compare_simulations()
