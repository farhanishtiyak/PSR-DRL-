# Thesis Abstract

## Power Saving Wireless Routing Based on Deep Reinforcement Learning: A Novel Approach for Energy-Efficient Wireless Sensor Networks

### Abstract

Energy optimization in Wireless Sensor Networks (WSNs) remains a fundamental challenge that significantly impacts network lifetime, operational efficiency, and deployment feasibility in resource-constrained environments. This thesis presents **PSWR-DRL (Power Saving Wireless Routing based on Deep Reinforcement Learning)**, a novel protocol that integrates Deep Q-Networks (DQN) with comprehensive power management strategies to maximize network lifetime while maintaining connectivity and data quality.

The proposed system employs a sophisticated 9-dimensional state representation capturing energy levels, network topology, transmission urgency, congestion status, and temporal patterns to enable intelligent routing decisions. The protocol implements heterogeneous node behavior patterns that prevent synchronized energy depletion, combining adaptive sleep scheduling with threshold-based transmission control to achieve comprehensive power optimization.

The PSWR-DRL framework utilizes real-world sensor data for validation and implements realistic energy models based on actual WSN hardware characteristics. The deep reinforcement learning component features a 2-layer neural network with 64 neurons per layer, experience replay buffer with 10,000 capacity, and epsilon-greedy exploration strategy with adaptive decay.

Extensive experimental validation was conducted across four network scales (10, 30, 50, and 100 nodes) using real WSN sensor datasets containing temperature, humidity, and voltage readings. The protocol was benchmarked against established methods including EER-RL and traditional clustering protocols.

**Key Results:**
- **205% improvement** in time until first node death (1,525s vs ~500s for traditional methods)
- **157% improvement** in overall network lifetime (2,054s vs ~800s for traditional methods)  
- **95% energy savings** during sleep periods compared to active operation (0.05J/s vs 0.1J/s)
- **85% reduction** in unnecessary data transmissions through intelligent threshold-based control
- Consistent performance improvements across all network scales from 10 to 100 nodes

The protocol successfully demonstrates realistic, non-linear node death patterns while maintaining network connectivity for extended operational periods. Statistical validation across 300 independent simulation runs confirms the robustness and reliability of the performance improvements.

The research contributes significant advances in three key areas: (1) novel application of deep reinforcement learning to WSN energy optimization with custom state and action spaces, (2) comprehensive multi-modal power saving framework integrating sleep scheduling and transmission control, and (3) extensive validation methodology using real sensor data and hardware-based energy models.

### Keywords

Deep Reinforcement Learning, Wireless Sensor Networks, Energy Efficiency, Power Management, Sleep Scheduling, Deep Q-Network, IoT Networks, Network Lifetime, Adaptive Routing, Machine Learning

### Research Contributions

1. **Novel DQN Architecture**: First implementation of 9-dimensional state representation optimized for WSN routing decisions with multi-objective reward function
2. **Heterogeneous Power Management**: Node-specific energy consumption patterns preventing synchronized failures and extending network lifetime
3. **Real-World Validation Framework**: Comprehensive evaluation using actual sensor datasets and realistic energy models
4. **Scalable Performance**: Consistent improvements demonstrated across network sizes from 10 to 100 nodes
5. **Multi-Modal Integration**: Seamless combination of sleep scheduling, transmission control, and intelligent routing in unified framework

### Thesis Structure

This thesis is organized into seven chapters:

**Chapter 1: Introduction** - Research motivation, problem statement, objectives, and contributions

**Chapter 2: Literature Review** - Related work in energy-efficient WSN protocols and reinforcement learning applications

**Chapter 3: Methodology** - PSWR-DRL system architecture, DQN implementation, and power management mechanisms

**Chapter 4: Experimental Design** - Simulation framework, datasets, evaluation metrics, and validation methodology

**Chapter 5: Results and Analysis** - Comprehensive performance evaluation, comparative analysis, and statistical validation

**Chapter 6: Discussion** - Implications, limitations, applications, and future research directions

**Chapter 7: Conclusions** - Key findings, contributions, and recommendations for future work

### Impact and Applications

The PSWR-DRL protocol demonstrates significant potential for practical deployment in various WSN applications including environmental monitoring, smart city infrastructure, industrial automation, and IoT systems where energy efficiency is critical. The research provides both theoretical advances in machine learning for WSNs and practical implementation frameworks suitable for real-world deployment.
