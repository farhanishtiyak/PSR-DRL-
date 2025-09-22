import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set up matplotlib for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_competitor_data():
    """Load competitor simulation data"""
    competitor_file = 'competitor_simulation_data.csv'
    if not os.path.exists(competitor_file):
        print(f"Warning: {competitor_file} not found")
        return None
    
    df = pd.read_csv(competitor_file)
    return df

def load_proposed_data():
    """Load proposed method data"""
    # Load network lifetime data
    network_file = 'proposed_network_lifetime_data.csv'
    death_file = 'proposed_node_death_times.csv'
    
    network_df = None
    death_df = None
    
    if os.path.exists(network_file):
        network_df = pd.read_csv(network_file)
    else:
        print(f"Warning: {network_file} not found")
    
    if os.path.exists(death_file):
        death_df = pd.read_csv(death_file)
    else:
        print(f"Warning: {death_file} not found")
    
    return network_df, death_df

def load_eerrl_data():
    """Load EER-RL simulation data"""
    eerrl_file = 'eerrl_simulation_data.csv'
    if not os.path.exists(eerrl_file):
        print(f"Warning: {eerrl_file} not found")
        return None
    
    df = pd.read_csv(eerrl_file)
    return df

def extract_first_death_time(df, method_type):
    """Extract first node death time from data"""
    if method_type == 'competitor':
        # For competitor data, find when Live_Node_Percentage first drops below 100%
        if df is not None and 'Live_Node_Percentage' in df.columns:
            first_death_idx = df[df['Live_Node_Percentage'] < 100].index
            if len(first_death_idx) > 0:
                return df.loc[first_death_idx[0], 'Time']
        return None
    
    elif method_type == 'proposed':
        # For proposed data, find when live_percentage first drops below 100%
        network_df, death_df = df
        if network_df is not None and 'live_percentage' in network_df.columns:
            first_death_idx = network_df[network_df['live_percentage'] < 100].index
            if len(first_death_idx) > 0:
                return network_df.loc[first_death_idx[0], 'time']
        # Alternative: use death times data
        elif death_df is not None and 'Death_Time' in death_df.columns:
            return death_df['Death_Time'].min()
        return None
    
    elif method_type == 'eerrl':
        # For EER-RL data, find when OperatingNodes first drops below 100
        if df is not None and 'OperatingNodes' in df.columns:
            first_death_idx = df[df['OperatingNodes'] < 100].index
            if len(first_death_idx) > 0:
                return df.loc[first_death_idx[0], 'Times']
        return None

def plot_first_node_death_comparison():
    """Plot comparison of first node death times"""
    # Load data
    competitor_df = load_competitor_data()
    proposed_data = load_proposed_data()
    eerrl_df = load_eerrl_data()
    
    # Extract first death times
    competitor_first_death = extract_first_death_time(competitor_df, 'competitor')
    proposed_first_death = extract_first_death_time(proposed_data, 'proposed')
    eerrl_first_death = extract_first_death_time(eerrl_df, 'eerrl')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = []
    death_times = []
    colors = []
    
    if competitor_first_death is not None:
        methods.append('RLBEEP')
        death_times.append(competitor_first_death)
        colors.append('#ff7f0e')  # Orange
    
    if proposed_first_death is not None:
        methods.append('PSR-DRL')
        death_times.append(proposed_first_death)
        colors.append('#2ca02c')  # Green
    
    if eerrl_first_death is not None:
        methods.append('EER-RL')
        death_times.append(eerrl_first_death)
        colors.append('#1f77b4')  # Blue
    
    if len(methods) > 0:
        bars = ax.bar(methods, death_times, color=colors, alpha=0.7, edgecolor='black', linewidth=1, width=0.6)
        
        # Add value labels on bars
        for bar, value in zip(bars, death_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Time (simulation steps)', fontweight='bold')
        ax.set_title('First Node Death Time Comparison\n100-Node Wireless Sensor Network', 
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        
    else:
        ax.text(0.5, 0.5, 'No data available for comparison', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    plt.tight_layout()
    return fig

def plot_network_lifetime_comparison():
    """Plot network lifetime comparison over time"""
    # Load data
    competitor_df = load_competitor_data()
    proposed_data = load_proposed_data()
    eerrl_df = load_eerrl_data()
    network_df, _ = proposed_data
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot competitor data
    # Plot proposed data (PSR-DRL first)
    if network_df is not None and 'time' in network_df.columns and 'live_percentage' in network_df.columns:
        ax.plot(network_df['time'], network_df['live_percentage'], 
               label='PSR-DRL', color='blue', linewidth=1.5)
    
    # Plot competitor data (RLBEEP second)
    if competitor_df is not None and 'Time' in competitor_df.columns and 'Live_Node_Percentage' in competitor_df.columns:
        ax.plot(competitor_df['Time'], competitor_df['Live_Node_Percentage'], 
               label='RLBEEP', color='green', linewidth=1.5)
    
    # Plot EER-RL data (EER-RL third)
    if eerrl_df is not None and 'Times' in eerrl_df.columns and 'OperatingNodes' in eerrl_df.columns:
        # Convert OperatingNodes to percentage (assuming max is 100 nodes)
        eerrl_percentage = (eerrl_df['OperatingNodes'] / 100) * 100
        ax.plot(eerrl_df['Times'], eerrl_percentage, 
               label='EER-RL', color='red', linewidth=1.5)
    
    ax.set_xlabel('Time (seconds)',  fontsize=14)
    ax.set_ylabel('Alive Sensors (%)',  fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Ensure lines appear with equal thickness in both directions
    ax.set_aspect('auto')
    
    plt.tight_layout()
    return fig

def generate_summary_statistics():
    """Generate and print summary statistics"""
    print("="*60)
    print("NETWORK PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    # Load data
    competitor_df = load_competitor_data()
    proposed_data = load_proposed_data()
    eerrl_df = load_eerrl_data()
    network_df, death_df = proposed_data
    
    # First node death time analysis
    competitor_first_death = extract_first_death_time(competitor_df, 'competitor')
    proposed_first_death = extract_first_death_time(proposed_data, 'proposed')
    eerrl_first_death = extract_first_death_time(eerrl_df, 'eerrl')
    
    print("\n1. FIRST NODE DEATH TIME ANALYSIS:")
    print("-" * 40)
    if competitor_first_death is not None:
        print(f"RLBEEP: {competitor_first_death:.0f} time steps")
    if proposed_first_death is not None:
        print(f"PSR-DRL: {proposed_first_death:.0f} time steps")
    if eerrl_first_death is not None:
        print(f"EER-RL: {eerrl_first_death:.0f} time steps")
    
    if competitor_first_death is not None and proposed_first_death is not None:
        improvement_prop_vs_comp = ((proposed_first_death - competitor_first_death) / competitor_first_death) * 100
        print(f"Improvement (PSR-DRL vs RLBEEP): {improvement_prop_vs_comp:+.2f}%")
    
    if competitor_first_death is not None and eerrl_first_death is not None:
        improvement_eerrl_vs_comp = ((eerrl_first_death - competitor_first_death) / competitor_first_death) * 100
        print(f"Improvement (EER-RL vs RLBEEP): {improvement_eerrl_vs_comp:+.2f}%")
    
    if proposed_first_death is not None and eerrl_first_death is not None:
        improvement_eerrl_vs_prop = ((eerrl_first_death - proposed_first_death) / proposed_first_death) * 100
        print(f"Improvement (EER-RL vs PSR-DRL): {improvement_eerrl_vs_prop:+.2f}%")
    
    # Network lifetime analysis
    print("\n2. NETWORK LIFETIME ANALYSIS:")
    print("-" * 40)
    
    if competitor_df is not None:
        final_time_comp = competitor_df['Time'].max()
        final_percentage_comp = competitor_df['Live_Node_Percentage'].iloc[-1]
        print(f"RLBEEP - Final time: {final_time_comp:.0f}, Final live %: {final_percentage_comp:.1f}%")
    
    if network_df is not None:
        final_time_prop = network_df['time'].max()
        final_percentage_prop = network_df['live_percentage'].iloc[-1]
        print(f"PSR-DRL - Final time: {final_time_prop:.0f}, Final live %: {final_percentage_prop:.1f}%")
    
    if eerrl_df is not None:
        final_time_eerrl = eerrl_df['Times'].max()
        final_nodes_eerrl = eerrl_df['OperatingNodes'].iloc[-1]
        final_percentage_eerrl = (final_nodes_eerrl / 100) * 100  # Assuming 100 nodes total
        print(f"EER-RL - Final time: {final_time_eerrl:.0f}, Final live %: {final_percentage_eerrl:.1f}%")
    
    # Node death pattern analysis
    if death_df is not None:
        print("\n3. NODE DEATH PATTERN (PSR-DRL Method):")
        print("-" * 40)
        print(f"Total nodes died: {len(death_df)}")
        print(f"Average death time: {death_df['Death_Time'].mean():.1f}")
        print(f"Standard deviation: {death_df['Death_Time'].std():.1f}")
        
        # Death by node type
        if 'Node_Type' in death_df.columns:
            death_by_type = death_df['Node_Type'].value_counts()
            print(f"Deaths by type: {dict(death_by_type)}")

def main():
    """Main function to generate all plots and analysis"""
    print("Generating 100-Node Network Performance Comparison...")
    
    # Change to the correct directory
    os.chdir('/home/ishtiyak/Desktop/Thesis/MatLab/Comparison/100')
    
    # Generate summary statistics
    generate_summary_statistics()
    
    # Create plots
    print("\nGenerating plots...")
    
    # Plot 1: First node death time comparison
    fig1 = plot_first_node_death_comparison()
    fig1.savefig('first_node_death_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ First node death comparison plot saved as 'first_node_death_comparison.png'")
    
    # Plot 2: Network lifetime comparison
    fig2 = plot_network_lifetime_comparison()
    fig2.savefig('network_lifetime_comparison.png', dpi=300, bbox_inches='tight')
    fig2.savefig('network_lifetime_comparison.pdf', dpi=300, bbox_inches='tight')
    print("✓ Network lifetime comparison plot saved as 'network_lifetime_comparison.png'")
    print("✓ Network lifetime comparison plot saved as 'network_lifetime_comparison.pdf'")
    
    # Show plots
    plt.show()
    
    print("\nComparison analysis completed!")
    print("Check the generated PNG files for detailed visualizations.")

if __name__ == "__main__":
    main()
