# RLBEEP 300 Epochs Analysis

This directory contains scripts to run the RLBEEP protocol for 300 epochs and analyze the first node death times across epochs.

## Files

-   `main.py` - Main RLBEEP simulation code (with minor CSV logging addition)
-   `run_300_epochs.py` - Script to run 300 epochs and generate analysis
-   `test_epochs.py` - Test script to verify setup with 5 epochs
-   `DRL_DOCUMENTATION.md` - Comprehensive DRL documentation
-   `Dataset/` - Directory containing sensor data files
-   `results/` - Directory where results and visualizations are saved

## Usage

### 1. Test the Setup (Recommended First)

Before running the full 300 epochs, test your setup:

```bash
python test_epochs.py
```

This will:

-   Run 5 short epochs to verify everything works
-   Show estimated time for 300 epochs
-   Create a test CSV file
-   Display basic results

### 2. Run 300 Epochs Analysis

Once testing is successful, run the full analysis:

```bash
python run_300_epochs.py
```

This will:

-   Run 300 complete epochs of the RLBEEP simulation
-   Track first node death time for each epoch
-   Save results to `results/epoch_results_300.csv`
-   Generate comprehensive visualizations
-   Create a detailed summary report

**Note**: This may take several hours depending on your system performance.

### 3. Results Generated

After completion, you'll find:

1. **CSV File**: `results/epoch_results_300.csv`

    - Contains data for all 300 epochs
    - Columns: epoch, first_death_time, final_live_percentage, etc.

2. **Visualization**: `results/epoch_analysis_TIMESTAMP.png`

    - Four subplots showing:
        - First Death Time vs Epochs
        - Distribution of Death Times
        - Network Survival vs Epochs
        - Energy Efficiency vs Epochs

3. **Summary Report**: `results/300_epochs_summary_TIMESTAMP.txt`
    - Statistical analysis of all epochs
    - Performance trends
    - Key insights

## Configuration

The default configuration in `run_300_epochs.py`:

```python
config = {
    'num_nodes': 10,           # Number of sensor nodes
    'num_clusters': 4,         # Number of clusters
    'total_time': 80000,       # Simulation time per epoch (seconds)
    'send_range': 10.0,        # Transmission range (meters)
    'alpha': 0.5,              # Learning rate
    'change_threshold': 1.5,   # Data change threshold
    'debug': False,            # Debug output
    'csv_filename': 'epoch_results_300.csv'
}
```

You can modify these values in the script as needed.

## Understanding the Results

### First Node Death Time

-   **Higher values**: Better network lifetime
-   **Trend analysis**: Shows if the DRL is improving over epochs
-   **Distribution**: Shows consistency of performance

### Network Survival Percentage

-   **100%**: Perfect epoch (no node deaths)
-   **Lower values**: Some nodes died during simulation
-   **Trend**: Shows overall network resilience

### Energy Efficiency

-   **Higher values**: Better energy utilization
-   **Trend**: Shows if energy management improves over epochs

## Troubleshooting

### Common Issues

1. **Dataset Not Found**

    - Ensure `Dataset/` directory exists
    - Check that node CSV files are present (node1.csv, node2.csv, etc.)

2. **Memory Issues**

    - Reduce `total_time` in configuration
    - Reduce number of epochs for testing

3. **Long Runtime**

    - Each epoch takes 30-60 seconds typically
    - 300 epochs may take 5-10 hours
    - Consider running overnight

4. **Import Errors**
    - Ensure all required packages are installed:
        ```bash
        pip install torch pandas matplotlib numpy
        ```

### Progress Monitoring

The script shows progress every 10 epochs:

```
Progress: 50/300 epochs (16.7%)
Elapsed: 1.2h, Estimated remaining: 6.1h
```

## Output Examples

### Console Output

```
RLBEEP Protocol - 300 Epochs Analysis
==================================================

Dataset Path: /path/to/Dataset
Results Directory: /path/to/results
Nodes: 10, Clusters: 4
Simulation Time: 80000 seconds per epoch

============================================================
  EPOCH 1/300 - RLBEEP Simulation
============================================================
Starting RLBEEP simulation with 10 nodes, 4 clusters
...
Epoch 1 Summary:
  Duration: 45.23 seconds
  First Death Time: 15234
  Final Live Percentage: 80.0%
```

### CSV Output

```csv
epoch,first_death_time,final_live_percentage,simulation_duration,total_transmissions,avg_remaining_energy,energy_efficiency,num_nodes,num_clusters
1,15234,80.0,80000,1250,45.67,0.54,10,4
2,16789,90.0,80000,1180,52.34,0.48,10,4
...
```

## Analysis Features

The analysis includes:

1. **Trend Analysis**: Linear regression on death times vs epochs
2. **Statistical Summary**: Mean, median, standard deviation
3. **Performance Metrics**: Survival rates, energy efficiency
4. **Visualization**: Multiple plots showing different aspects
5. **Comprehensive Report**: Detailed text summary

## Research Applications

This analysis is useful for:

-   **DRL Performance Evaluation**: How well the learning improves over time
-   **Protocol Comparison**: Baseline metrics for comparison with other protocols
-   **Network Optimization**: Identifying optimal configurations
-   **Academic Research**: Data for papers and thesis work

## Citation

If you use this analysis in your research, please cite:

```
RLBEEP: Reinforcement Learning Based Energy Efficient Protocol for Wireless Sensor Networks
[Your research paper citation]
```

## Support

For questions or issues:

1. Check the troubleshooting section above
2. Review the DRL documentation in `DRL_DOCUMENTATION.md`
3. Examine the console output for error messages
4. Test with shorter epochs using `test_epochs.py`
