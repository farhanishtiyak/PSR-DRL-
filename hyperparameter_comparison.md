# Hyperparameter Comparison Across Node Configurations

This document provides a comprehensive comparison of hyperparameters used in the PSWR-DRL (Power Saving Wireless Routing based on Deep Reinforcement Learning) system across different network scales.

## Network Parameters

| Parameter | 10 Nodes | 30 Nodes | 50 Nodes | 100 Nodes |
|-----------|----------|----------|----------|-----------|
| Number of Nodes | 10 | 30 | 50 | 100 |
| Number of Clusters | 4 | 12 | 20 | 40 |
| Send Range (meters) | 10 | 20 | 25 | 30 |
| Max Longitude (meters) | 60.0 | 100.0 | 150.0 | 200.0 |
| Max Latitude (meters) | 60.0 | 100.0 | 150.0 | 200.0 |

## Energy Parameters

| Parameter | 10 Nodes | 30 Nodes | 50 Nodes | 100 Nodes |
|-----------|----------|----------|----------|-----------|
| Initial Energy (Joules) | 100 | 150 | 200 | 300 |
| Power for Sending (J) | 0.3 | 0.3 | 0.4 | 0.5 |
| Power for Receiving (J) | 0.2 | 0.2 | 0.25 | 0.3 |
| Power for Active State (J/s) | 0.1 | 0.1 | 0.12 | 0.15 |
| Power for Sleep State (J/s) | 0.05 | 0.05 | 0.06 | 0.08 |

## Routing Parameters

| Parameter | 10 Nodes | 30 Nodes | 50 Nodes | 100 Nodes |
|-----------|----------|----------|----------|-----------|
| Learning Rate (Alpha) | 0.5 | 0.5 | 0.5 | 0.5 |
| DFR Minimum | 5.0 | 5.0 | 5.0 | 5.0 |
| DFR Maximum | 55.0 | 55.0 | 55.0 | 55.0 |

## Sleep Scheduling Parameters

| Parameter | 10 Nodes | 30 Nodes | 50 Nodes | 100 Nodes |
|-----------|----------|----------|----------|-----------|
| Sleep Restrict Threshold | 5 | 5 | 5 | 5 |
| Sleep Duration (seconds) | 30 | 30 | 30 | 30 |
| Wake Up Check Interval (seconds) | 30 | 30 | 30 | 30 |

## Data Transmission Parameters

| Parameter | 10 Nodes | 30 Nodes | 50 Nodes | 100 Nodes |
|-----------|----------|----------|----------|-----------|
| Change Threshold | 1.5 | 1.5 | 1.5 | 1.5 |
| Transmission Period (seconds) | 6 | 6 | 6 | 6 |

## DQL Parameters

| Parameter | 10 Nodes | 30 Nodes | 50 Nodes | 100 Nodes |
|-----------|----------|----------|----------|-----------|
| Batch Size | 32 | 32 | 32 | 32 |
| Gamma (Discount Factor) | 0.99 | 0.99 | 0.99 | 0.99 |
| Epsilon Start | 0.9 | 0.9 | 0.9 | 0.9 |
| Epsilon End | 0.05 | 0.05 | 0.05 | 0.05 |
| Epsilon Decay | 200 | 200 | 200 | 200 |
| Target Network Update Frequency | 10 | 10 | 10 | 10 |
| Memory Size | 10000 | 10000 | 10000 | 10000 |

## Simulation Parameters

| Parameter | 10 Nodes | 30 Nodes | 50 Nodes | 100 Nodes |
|-----------|----------|----------|----------|-----------|
| Total Simulation Time (seconds) | 3000 | 3000 | 3000 | 3000 |

## Key Observations

1. **Network Density Scaling**: As the number of nodes increases, both the geographical area (longitude/latitude) and transmission range increase proportionally, maintaining appropriate network density.

2. **Energy Scaling**: Initial energy allocation increases with network size:
   - 10 nodes: 100 Joules per node
   - 30 nodes: 150 Joules per node
   - 50 nodes: 200 Joules per node
   - 100 nodes: 300 Joules per node

3. **Power Consumption Adjustment**: Power requirements for all operations (sending, receiving, active state, sleep state) increase with network size to account for longer transmission distances and higher network complexity.

4. **Cluster Ratio Consistency**: The ratio of clusters to nodes remains relatively consistent across configurations (approximately 1:2.5), ensuring appropriate clustering density.

5. **Algorithm Parameters Consistency**: Most DQL parameters remain constant across all network scales, demonstrating the algorithm's inherent scalability.

These configurations demonstrate careful parameter scaling to maintain consistent network behavior and performance across different network sizes, while accounting for the increased complexity and energy requirements of larger deployments.
