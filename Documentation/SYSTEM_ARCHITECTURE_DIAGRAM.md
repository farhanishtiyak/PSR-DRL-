# PSWR-DRL System Model Diagram

This document provides a simplified system model diagram for the Power Saving Wireless Routing based on Deep Reinforcement Learning (PSWR-DRL) system.

## PSWR-DRL System Model

```blockdiag
blockdiag {
  default_fontsize = 12;
  default_textcolor = black;
  
  // External Components
  "WSN Dataset" [color = orange];
  "Results Output" [color = orange];
  
  // Main System
  group {
    label = "PSWR-DRL System";
    color = lightsteelblue;
    
    // Network Module
    group {
      label = "Network Module";
      color = lightblue;
      
      "Network Topology" [color = skyblue];
      "Regular Nodes" [color = lightcyan];
      "Cluster Heads" [color = steelblue, textcolor = white];
      "Sink Node" [color = darkblue, textcolor = white];
    }
    
    // Power Management Module
    group {
      label = "Power Management Module";
      color = lightcoral;
      
      "Sleep Scheduling" [color = coral];
      "Data Restriction" [color = lightpink];
      "Energy Monitoring" [color = peachpuff];
    }
    
    // DRL Module
    group {
      label = "DRL Module";
      color = lightyellow;
      
      "Deep Q-Network\n(9→64→64→3)" [color = gold];
      "State Space\n(9 Features)" [color = yellow];
      "Action Space\n(3 Actions)" [color = wheat];
      "Learning Engine" [color = khaki];
    }
  }
  
  // System Flow
  "WSN Dataset" -> "Network Topology";
  "Network Topology" -> "Regular Nodes";
  "Regular Nodes" -> "Sleep Scheduling";
  "Sleep Scheduling" -> "State Space\n(9 Features)";
  "State Space\n(9 Features)" -> "Deep Q-Network\n(9→64→64→3)";
  "Deep Q-Network\n(9→64→64→3)" -> "Action Space\n(3 Actions)";
  "Action Space\n(3 Actions)" -> "Energy Monitoring";
  "Energy Monitoring" -> "Results Output";
  
  // Feedback
  "Learning Engine" -> "Deep Q-Network\n(9→64→64→3)";
  "Energy Monitoring" -> "Learning Engine";
}
```

## PSWR-DRL Process Flow Diagram

```blockdiag
blockdiag {
  default_fontsize = 11;
  default_textcolor = black;
  
  // Start/End nodes
  "START" [color = green, shape = ellipse];
  "END" [color = red, shape = ellipse];
  
  // Input/Output
  "Load WSN Dataset" [color = orange];
  "Performance Results" [color = orange];
  
  // Network Setup Phase
  group {
    label = "Network Initialization";
    color = lightblue;
    
    "Deploy Nodes" [color = skyblue];
    "Select Cluster Heads" [color = skyblue];
    "Initialize Energy" [color = skyblue];
  }
  
  // Main Simulation Loop
  group {
    label = "Simulation Loop (Each Time Step)";
    color = lightgray;
    
    // Sensor Data Processing
    "Get Sensor Data" [color = wheat];
    
    // For Each Node Decision Process
    group {
      label = "For Each Node";
      color = lightyellow;
      
      "Extract State\n(9 Features)" [color = yellow];
      "DQN Decision\n(3 Actions)" [color = gold];
      "Execute Action" [color = khaki];
    }
    
    // Power Management Decisions
    group {
      label = "Power Management";
      color = lightcoral;
      
      "Sleep Check?" [color = coral, shape = diamond];
      "Enter Sleep Mode" [color = mistyrose];
      "Data Transmission?" [color = lightpink, shape = diamond];
      "Send Data" [color = peachpuff];
      "Update Energy" [color = salmon];
    }
    
    // Network Management
    group {
      label = "Network Management";
      color = lightsteelblue;
      
      "Check Connectivity" [color = steelblue];
      "CH Rotation?" [color = skyblue, shape = diamond];
      "Rotate Cluster Heads" [color = lightcyan];
    }
    
    // Learning Update
    "Calculate Reward" [color = gold];
    "Update DQN" [color = yellow];
    
    // Termination Check
    "Network Alive?" [color = gray, shape = diamond];
  }
  
  // Flow Connections
  "START" -> "Load WSN Dataset";
  "Load WSN Dataset" -> "Deploy Nodes";
  "Deploy Nodes" -> "Select Cluster Heads";
  "Select Cluster Heads" -> "Initialize Energy";
  "Initialize Energy" -> "Get Sensor Data";
  
  "Get Sensor Data" -> "Extract State\n(9 Features)";
  "Extract State\n(9 Features)" -> "DQN Decision\n(3 Actions)";
  "DQN Decision\n(3 Actions)" -> "Execute Action";
  "Execute Action" -> "Sleep Check?";
  
  "Sleep Check?" -> "Enter Sleep Mode" [label = "Yes"];
  "Sleep Check?" -> "Data Transmission?" [label = "No"];
  "Enter Sleep Mode" -> "Update Energy";
  
  "Data Transmission?" -> "Send Data" [label = "Yes"];
  "Data Transmission?" -> "Update Energy" [label = "No"];
  "Send Data" -> "Update Energy";
  
  "Update Energy" -> "Check Connectivity";
  "Check Connectivity" -> "CH Rotation?";
  "CH Rotation?" -> "Rotate Cluster Heads" [label = "Yes"];
  "CH Rotation?" -> "Calculate Reward" [label = "No"];
  "Rotate Cluster Heads" -> "Calculate Reward";
  
  "Calculate Reward" -> "Update DQN";
  "Update DQN" -> "Network Alive?";
  
  "Network Alive?" -> "Get Sensor Data" [label = "Yes", style = dashed];
  "Network Alive?" -> "Performance Results" [label = "No"];
  "Performance Results" -> "END";
}
```

## Detailed Algorithm Flow

### Phase 1: System Initialization
1. **Load Dataset**: Import real WSN sensor data
2. **Deploy Nodes**: Strategic grid-based placement (10-100 nodes)
3. **Select Cluster Heads**: K-means inspired selection
4. **Initialize Energy**: Set initial energy (100J per node)

### Phase 2: Main Simulation Loop (Every Time Step)

#### 2.1 Node Decision Process
```
For each alive node:
  1. Extract 9-dimensional state vector:
     - Energy level, CH distance, Sink distance
     - Hop count, Data urgency, Network congestion
     - Sleep pressure, Cluster health, Temporal factor
  
  2. DQN makes routing decision:
     - Forward to Cluster Head (Action 0)
     - Forward to Sink (Action 1) 
     - Enter Sleep Mode (Action 2)
  
  3. Execute selected action
```

#### 2.2 Power Management Process
```
Sleep Decision:
  IF no_send_count >= threshold (3-7):
    - Enter sleep mode (0.05 J/s)
    - Sleep duration: 20-40 seconds + random offset
  ELSE:
    - Check data transmission need
    
Data Transmission:
  IF significant data change OR random transmission:
    - Send data (0.3J cost)
    - Receive acknowledgment (0.2J cost)
  
Energy Update:
  - Deduct energy based on activity
  - Apply node-specific efficiency (0-30% variation)
```

#### 2.3 Network Management
```
Connectivity Check:
  - Monitor network partitions
  - Check cluster head status
  
Cluster Head Rotation (Every 300 seconds):
  - Select new CHs based on energy levels
  - Reassign nodes to clusters
  - Update routing tables
  
Emergency Recovery:
  IF network partitioned:
    - Attempt connectivity restoration
    - Extend communication range if needed
```

#### 2.4 Learning Update
```
Reward Calculation:
  reward = base_reward + energy_reward + lifetime_reward + distance_reward
  
DQN Update:
  - Store experience in replay memory (10,000 capacity)
  - Sample batch for training
  - Update neural network weights
  - Update target network periodically
```

### Phase 3: Termination Conditions
- All nodes dead (energy = 0)
- Network partitioned (no path to sink)
- Maximum simulation time reached (3000 seconds)

## System Overview

### Core Components
- **Network Module**: Manages network topology, nodes, cluster heads, and sink
- **Power Management Module**: Controls sleep scheduling, data restriction, and energy monitoring
- **DRL Module**: Implements Deep Q-Network for intelligent routing decisions

### Key Features
- **205% First Node Death Improvement**
- **157% Network Lifetime Extension** 
- **95% Sleep Energy Savings** (0.05 J/s vs 0.1 J/s)
- **85% Transmission Reduction**

### Technical Specs
| Component | Value |
|-----------|-------|
| Nodes | 10-100 |
| Neural Network | 9→64→64→3 |
| State Features | 9 dimensions |
| Actions | 3 (CH/Sink/Sleep) |
| Energy: Active | 0.1 J/s |
| Energy: Sleep | 0.05 J/s |
