# Chapter 4: Experimental Design and Implementation

## 4.1 Experimental Framework Overview

The experimental validation of the PSWR-DRL system employs a comprehensive multi-scale evaluation framework designed to assess performance across diverse network configurations and operational scenarios. The experimental design incorporates real-world sensor data, realistic energy models, and rigorous statistical validation methodologies to ensure the reliability and practical applicability of the research findings.

### 4.1.1 Experimental Objectives

**Primary Objectives:**
1. **Performance Validation**: Quantify improvements in network lifetime, energy efficiency, and data delivery performance
2. **Scalability Assessment**: Evaluate protocol effectiveness across different network sizes (10, 30, 50, 100 nodes)
3. **Comparative Analysis**: Benchmark PSWR-DRL against established WSN energy management protocols
4. **Statistical Significance**: Validate results through comprehensive statistical analysis across multiple simulation runs
5. **Real-World Applicability**: Demonstrate practical effectiveness using actual sensor datasets and realistic operational conditions

**Secondary Objectives:**
1. **Learning Convergence Analysis**: Assess DQN training effectiveness and stability
2. **Parameter Sensitivity**: Evaluate system robustness to configuration variations
3. **Operational Robustness**: Test performance under varying network conditions and failure scenarios
4. **Implementation Feasibility**: Validate computational requirements and deployment considerations

### 4.1.2 Experimental Design Principles

**Controlled Variables:**
- Network topology and deployment patterns
- Initial energy allocations and hardware specifications
- Data generation patterns and transmission requirements
- Environmental conditions and simulation parameters

**Independent Variables:**
- Network size (10, 30, 50, 100 nodes)
- Protocol type (PSWR-DRL, EER-RL, Traditional Clustering)
- Configuration parameters (learning rates, thresholds, sleep schedules)
- Operational scenarios (normal, high-load, failure conditions)

**Dependent Variables:**
- Network lifetime and first node death time
- Energy consumption patterns and efficiency metrics
- Data delivery performance and quality measures
- Learning convergence and adaptation effectiveness

## 4.2 Network Configurations and Deployment Scenarios

### 4.2.1 Multi-Scale Network Configurations

The experimental evaluation encompasses four distinct network scales, each representing different deployment scenarios and operational requirements:

**10-Node Configuration (Small-Scale Deployment)**
```python
config_10_nodes = {
    'num_nodes': 10,
    'num_clusters': 4,
    'network_area': (50, 50),          # meters
    'transmission_range': 10.0,         # meters
    'sink_position': (25, 25),          # center placement
    'initial_energy': 100.0,            # Joules per node
    'simulation_duration': 80000        # seconds
}
```

**30-Node Configuration (Medium-Scale Deployment)**
```python
config_30_nodes = {
    'num_nodes': 30,
    'num_clusters': 12,
    'network_area': (100, 100),         # meters
    'transmission_range': 20.0,         # meters
    'sink_position': (50, 50),          # center placement
    'initial_energy': 150.0,            # Joules per node
    'simulation_duration': 80000        # seconds
}
```

**50-Node Configuration (Large-Scale Deployment)**
```python
config_50_nodes = {
    'num_nodes': 50,
    'num_clusters': 20,
    'network_area': (150, 150),         # meters
    'transmission_range': 25.0,         # meters
    'sink_position': (75, 75),          # center placement
    'initial_energy': 200.0,            # Joules per node
    'simulation_duration': 80000        # seconds
}
```

**100-Node Configuration (Enterprise-Scale Deployment)**
```python
config_100_nodes = {
    'num_nodes': 100,
    'num_clusters': 40,
    'network_area': (200, 200),         # meters
    'transmission_range': 30.0,         # meters
    'sink_position': (100, 100),        # center placement
    'initial_energy': 300.0,            # Joules per node
    'simulation_duration': 80000        # seconds
}
```

### 4.2.2 Node Deployment Strategy

**Strategic Node Placement:**
```python
def deploy_nodes(config):
    """Strategic node deployment ensuring connectivity and coverage"""
    nodes = []
    
    # Sink node placement (unlimited energy)
    sink = Node(id=0, position=config.sink_position, energy=float('inf'))
    nodes.append(sink)
    
    # Sensor node deployment with spatial distribution
    for i in range(1, config.num_nodes + 1):
        position = generate_strategic_position(i, config)
        energy = calculate_initial_energy(i, config)
        node = Node(id=i, position=position, energy=energy)
        nodes.append(node)
    
    return nodes

def generate_strategic_position(node_id, config):
    """Generate node positions ensuring network connectivity"""
    # Implement spatial distribution algorithm
    # Ensure minimum distance between nodes
    # Guarantee connectivity to at least one cluster head
    # Consider coverage optimization
    pass
```

**Cluster Formation Algorithm:**
```python
def form_clusters(nodes, config):
    """Dynamic cluster formation based on energy and connectivity"""
    clusters = []
    
    # Select initial cluster heads based on energy and position
    cluster_heads = select_cluster_heads(nodes, config.num_clusters)
    
    # Assign nodes to clusters based on proximity and energy
    for node in nodes:
        if node not in cluster_heads:
            best_cluster = find_optimal_cluster(node, cluster_heads)
            best_cluster.add_member(node)
    
    return clusters

def select_cluster_heads(nodes, num_clusters):
    """Energy-aware cluster head selection"""
    candidates = [node for node in nodes if node.energy > 50.0]
    
    # Select based on energy level and spatial distribution
    cluster_heads = []
    for i in range(num_clusters):
        best_candidate = max(candidates, 
                           key=lambda n: n.energy * spatial_distribution_factor(n))
        cluster_heads.append(best_candidate)
        candidates.remove(best_candidate)
    
    return cluster_heads
```

### 4.2.3 Real-World Data Integration

**Sensor Dataset Integration:**

The experimental framework integrates actual sensor data collected from operational WSN deployments, ensuring realistic data patterns and transmission requirements:

```python
class RealWorldDataLoader:
    def __init__(self, dataset_path="/Dataset"):
        self.dataset_path = dataset_path
        self.sensor_files = [f"node{i}.csv" for i in range(1, 11)]
        self.sensor_data = self.load_all_sensor_data()
        
    def load_all_sensor_data(self):
        """Load sensor data from CSV files"""
        data = {}
        for i, filename in enumerate(self.sensor_files, 1):
            filepath = os.path.join(self.dataset_path, filename)
            try:
                df = pd.read_csv(filepath)
                data[i] = self.preprocess_sensor_data(df)
            except FileNotFoundError:
                print(f"Warning: {filename} not found, using synthetic data")
                data[i] = self.generate_synthetic_data()
        return data
    
    def preprocess_sensor_data(self, df):
        """Preprocess and validate sensor readings"""
        # Remove outliers and invalid readings
        # Interpolate missing values
        # Normalize temporal sampling
        # Calculate change rates for transmission decisions
        return processed_df
    
    def get_sensor_reading(self, node_id, timestamp):
        """Get sensor reading for specific node and time"""
        node_data = self.sensor_data.get(node_id, self.sensor_data[1])
        
        # Find closest timestamp or interpolate
        reading = self.interpolate_reading(node_data, timestamp)
        
        return {
            'temperature': reading.temperature,
            'humidity': reading.humidity,
            'voltage': reading.voltage,
            'timestamp': timestamp,
            'change_magnitude': self.calculate_change_magnitude(reading)
        }
```

**Data Characteristics Analysis:**
- **Temperature Range**: 15°C to 35°C with diurnal variations
- **Humidity Range**: 30% to 90% with seasonal patterns
- **Voltage Range**: 2.7V to 3.3V with energy depletion curves
- **Sampling Rate**: 1 reading per minute with irregular intervals
- **Data Quality**: Real-world noise, missing values, and outliers

## 4.3 Energy Models and Hardware Specifications

### 4.3.1 Realistic Energy Consumption Models

The experimental framework implements energy models based on actual WSN hardware specifications, ensuring realistic power consumption patterns:

**Hardware Reference: MICAz Mote Specifications**
```python
class MICAzEnergyModel:
    def __init__(self):
        # Based on Crossbow MICAz mote specifications
        self.supply_voltage = 3.0           # Volts
        self.battery_capacity = 3600        # mAh (2 AA batteries)
        
        # Current consumption by operational state
        self.current_active = 19.7e-3       # 19.7 mA (CPU active, radio RX)
        self.current_idle = 2.4e-3          # 2.4 mA (CPU idle, radio RX)
        self.current_sleep = 1e-6           # 1 µA (deep sleep mode)
        self.current_tx = 21.0e-3           # 21 mA (radio TX at 0 dBm)
        
        # Transition costs
        self.wakeup_energy = 0.01           # Joules
        self.sleep_transition_time = 0.1    # seconds
        
    def calculate_energy_consumption(self, state, duration_seconds):
        """Calculate energy consumption for specific state and duration"""
        current_map = {
            'active': self.current_active,
            'idle': self.current_idle,
            'sleep': self.current_sleep,
            'transmit': self.current_tx
        }
        
        current = current_map.get(state, self.current_active)
        energy_joules = current * self.supply_voltage * duration_seconds
        
        return energy_joules
    
    def calculate_transmission_energy(self, packet_size_bits, distance_meters):
        """Calculate energy for data transmission based on distance and packet size"""
        # Energy model: E_tx = E_elec * k + E_amp * k * d^2
        E_elec = 50e-9      # 50 nJ/bit (electronics energy)
        E_amp = 100e-12     # 100 pJ/bit/m² (amplifier energy)
        
        electronics_energy = E_elec * packet_size_bits
        amplifier_energy = E_amp * packet_size_bits * (distance_meters ** 2)
        
        return electronics_energy + amplifier_energy
```

### 4.3.2 Heterogeneous Energy Characteristics

To prevent synchronized energy depletion and create realistic diversity in node behavior, the system implements heterogeneous energy characteristics:

```python
class HeterogeneousEnergyManager:
    def __init__(self, num_nodes):
        self.node_energy_factors = self.generate_energy_factors(num_nodes)
        self.node_sleep_efficiencies = self.generate_sleep_efficiencies(num_nodes)
        
    def generate_energy_factors(self, num_nodes):
        """Generate node-specific energy consumption factors (0-30% variation)"""
        factors = {}
        for node_id in range(1, num_nodes + 1):
            # Consistent but varied factors based on node ID
            seed_value = hash(f"energy_{node_id}") % 1000
            random.seed(seed_value)
            variation = (random.random() - 0.5) * 0.3  # ±15% variation
            factors[node_id] = 1.0 + variation
        return factors
    
    def generate_sleep_efficiencies(self, num_nodes):
        """Generate node-specific sleep efficiency (95-98% energy reduction)"""
        efficiencies = {}
        for node_id in range(1, num_nodes + 1):
            seed_value = hash(f"sleep_{node_id}") % 1000
            random.seed(seed_value)
            efficiency = 0.95 + random.random() * 0.03  # 95-98%
            efficiencies[node_id] = efficiency
        return efficiencies
    
    def get_node_energy_consumption(self, node_id, base_consumption, state):
        """Calculate node-specific energy consumption"""
        factor = self.node_energy_factors.get(node_id, 1.0)
        
        if state == 'sleep':
            efficiency = self.node_sleep_efficiencies.get(node_id, 0.95)
            return base_consumption * (1.0 - efficiency)
        else:
            return base_consumption * factor
```

### 4.3.3 Energy Monitoring and Logging

**Real-Time Energy Tracking:**
```python
class EnergyMonitor:
    def __init__(self, nodes):
        self.nodes = nodes
        self.energy_history = {node.id: [] for node in nodes}
        self.state_history = {node.id: [] for node in nodes}
        
    def log_energy_state(self, timestamp):
        """Log current energy state of all nodes"""
        for node in self.nodes:
            energy_record = {
                'timestamp': timestamp,
                'node_id': node.id,
                'energy_level': node.energy,
                'energy_percentage': node.energy / node.initial_energy,
                'state': node.current_state,
                'alive': node.is_alive()
            }
            self.energy_history[node.id].append(energy_record)
            
    def calculate_network_metrics(self, timestamp):
        """Calculate network-wide energy metrics"""
        alive_nodes = [node for node in self.nodes if node.is_alive()]
        
        if not alive_nodes:
            return None
            
        metrics = {
            'timestamp': timestamp,
            'total_nodes': len(self.nodes),
            'alive_nodes': len(alive_nodes),
            'alive_percentage': len(alive_nodes) / len(self.nodes) * 100,
            'total_energy': sum(node.energy for node in alive_nodes),
            'average_energy': np.mean([node.energy for node in alive_nodes]),
            'min_energy': min(node.energy for node in alive_nodes),
            'max_energy': max(node.energy for node in alive_nodes),
            'energy_std': np.std([node.energy for node in alive_nodes])
        }
        
        return metrics
```

## 4.4 Deep Reinforcement Learning Configuration

### 4.4.1 DQN Hyperparameter Configuration

**Neural Network Architecture:**
```python
DQN_CONFIG = {
    'network_architecture': {
        'input_size': 9,                    # 9-dimensional state vector
        'hidden_layers': [64, 64],          # Two hidden layers with 64 neurons each
        'output_size': 3,                   # 3 possible actions
        'activation': 'relu',               # ReLU activation for hidden layers
        'output_activation': 'linear'       # Linear output for Q-values
    },
    
    'learning_parameters': {
        'learning_rate': 0.001,             # Adam optimizer learning rate
        'batch_size': 32,                   # Experience replay batch size
        'memory_size': 10000,               # Experience replay buffer capacity
        'target_update_frequency': 10,      # Target network update interval
        'training_frequency': 4             # Training every 4 experiences
    },
    
    'exploration_parameters': {
        'epsilon_start': 0.9,               # Initial exploration rate
        'epsilon_end': 0.05,                # Final exploration rate
        'epsilon_decay': 0.995,             # Exponential decay rate
        'exploration_episodes': 1000        # Episodes for exploration decay
    },
    
    'reward_parameters': {
        'energy_weight': 0.4,               # Energy optimization priority
        'connectivity_weight': 0.3,         # Network connectivity importance
        'performance_weight': 0.2,          # Data delivery performance
        'lifetime_weight': 0.1              # Long-term network lifetime
    }
}
```

**Experience Replay Implementation:**
```python
class ExperienceReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Store experience tuple in buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        experience = (state, action, reward, next_state, done)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)
```

### 4.4.2 State and Action Space Implementation

**State Vector Construction:**
```python
def construct_state_vector(node, network):
    """Construct 9-dimensional state vector for DQN input"""
    
    # s1: Normalized energy level (0-1)
    energy_level = node.energy / node.initial_energy
    
    # s2: Distance to cluster head (normalized)
    cluster_head = network.get_cluster_head(node.cluster_id)
    distance_to_ch = calculate_distance(node.position, cluster_head.position)
    normalized_distance_ch = min(distance_to_ch / node.transmission_range, 1.0)
    
    # s3: Distance to sink (normalized)
    sink_node = network.get_sink_node()
    distance_to_sink = calculate_distance(node.position, sink_node.position)
    normalized_distance_sink = distance_to_sink / network.max_distance
    
    # s4: Hop count to destination (normalized)
    hop_count = network.calculate_hop_count(node, sink_node)
    normalized_hop_count = hop_count / network.max_possible_hops
    
    # s5: Data transmission urgency (0-1)
    transmission_urgency = calculate_transmission_urgency(node)
    
    # s6: Network congestion level (0-1)
    congestion_level = network.calculate_local_congestion(node)
    
    # s7: Sleep pressure indicator (0-1)
    sleep_pressure = calculate_sleep_pressure(node)
    
    # s8: Cluster health status (0-1)
    cluster_health = network.calculate_cluster_health(node.cluster_id)
    
    # s9: Temporal factor (0-1)
    temporal_factor = calculate_temporal_factor(network.current_time)
    
    state_vector = np.array([
        energy_level, normalized_distance_ch, normalized_distance_sink,
        normalized_hop_count, transmission_urgency, congestion_level,
        sleep_pressure, cluster_health, temporal_factor
    ])
    
    return state_vector

def calculate_transmission_urgency(node):
    """Calculate urgency of data transmission"""
    if not node.data_queue:
        return 0.0
    
    urgency_factors = []
    for data_packet in node.data_queue:
        age_factor = min(data_packet.age / 300.0, 1.0)  # Max 5 minutes
        change_factor = data_packet.change_magnitude / 10.0
        priority_factor = data_packet.priority_level
        
        packet_urgency = (age_factor + change_factor + priority_factor) / 3.0
        urgency_factors.append(packet_urgency)
    
    return np.mean(urgency_factors)

def calculate_sleep_pressure(node):
    """Calculate node's inclination to enter sleep mode"""
    energy_pressure = 1.0 - (node.energy / node.initial_energy)
    activity_pressure = 1.0 - node.recent_activity_level
    queue_pressure = 1.0 - min(len(node.data_queue) / 10.0, 1.0)
    
    sleep_pressure = (energy_pressure + activity_pressure + queue_pressure) / 3.0
    return min(sleep_pressure, 1.0)
```

**Action Execution Framework:**
```python
class ActionExecutor:
    def __init__(self, network):
        self.network = network
        
    def execute_action(self, node, action):
        """Execute selected action and return reward"""
        if action == 0:  # Route to Cluster Head
            return self.route_to_cluster_head(node)
        elif action == 1:  # Route to Sink
            return self.route_to_sink(node)
        elif action == 2:  # Enter Sleep Mode
            return self.enter_sleep_mode(node)
        else:
            raise ValueError(f"Invalid action: {action}")
    
    def route_to_cluster_head(self, node):
        """Route data to cluster head"""
        if not node.data_queue:
            return -1.0  # Penalty for unnecessary routing
        
        cluster_head = self.network.get_cluster_head(node.cluster_id)
        if not cluster_head or not cluster_head.is_alive():
            return -10.0  # Penalty for routing to dead cluster head
        
        # Calculate transmission cost
        distance = calculate_distance(node.position, cluster_head.position)
        energy_cost = calculate_transmission_energy(distance)
        
        if node.energy < energy_cost:
            return -15.0  # Penalty for insufficient energy
        
        # Execute transmission
        success = self.transmit_data(node, cluster_head, energy_cost)
        
        if success:
            reward = self.calculate_routing_reward(node, energy_cost, 'cluster_head')
            return reward
        else:
            return -5.0  # Penalty for failed transmission
    
    def route_to_sink(self, node):
        """Route data directly to sink"""
        if not node.data_queue:
            return -1.0  # Penalty for unnecessary routing
        
        sink_node = self.network.get_sink_node()
        distance = calculate_distance(node.position, sink_node.position)
        
        if distance > node.transmission_range:
            return -8.0  # Penalty for out-of-range transmission
        
        energy_cost = calculate_transmission_energy(distance) * 1.5  # Higher cost
        
        if node.energy < energy_cost:
            return -15.0  # Penalty for insufficient energy
        
        # Execute direct transmission
        success = self.transmit_data(node, sink_node, energy_cost)
        
        if success:
            reward = self.calculate_routing_reward(node, energy_cost, 'sink')
            return reward + 2.0  # Bonus for direct delivery
        else:
            return -5.0  # Penalty for failed transmission
    
    def enter_sleep_mode(self, node):
        """Transition node to sleep mode"""
        if node.data_queue:
            return -3.0  # Penalty for sleeping with pending data
        
        if node.is_cluster_head():
            return -10.0  # Penalty for cluster head sleeping
        
        if node.energy < 0.1:
            return 10.0  # High reward for energy conservation when critical
        
        # Calculate sleep duration
        sleep_duration = self.calculate_optimal_sleep_duration(node)
        
        # Execute sleep transition
        node.enter_sleep_mode(sleep_duration)
        
        # Calculate sleep reward
        energy_saved = node.calculate_sleep_energy_savings(sleep_duration)
        reward = min(energy_saved * 50, 8.0)  # Scale and cap reward
        
        return reward
```

## 4.5 Evaluation Metrics and Performance Assessment

### 4.5.1 Primary Performance Metrics

**1. Network Lifetime Metrics:**
```python
class NetworkLifetimeMetrics:
    def __init__(self):
        self.first_node_death_time = None
        self.network_partition_time = None
        self.final_alive_percentage = 0.0
        
    def update_lifetime_metrics(self, timestamp, network_state):
        """Update network lifetime metrics"""
        alive_nodes = [node for node in network_state.nodes if node.is_alive()]
        alive_percentage = len(alive_nodes) / len(network_state.nodes)
        
        # Record first node death
        if self.first_node_death_time is None and alive_percentage < 1.0:
            self.first_node_death_time = timestamp
        
        # Check for network partition
        if self.network_partition_time is None:
            if not self.is_network_connected(alive_nodes):
                self.network_partition_time = timestamp
        
        self.final_alive_percentage = alive_percentage
    
    def is_network_connected(self, alive_nodes):
        """Check if network remains connected"""
        if len(alive_nodes) < 2:
            return False
        
        # Implement connectivity check using graph traversal
        connectivity_graph = self.build_connectivity_graph(alive_nodes)
        return self.is_graph_connected(connectivity_graph)
```

**2. Energy Efficiency Metrics:**
```python
class EnergyEfficiencyMetrics:
    def __init__(self):
        self.total_energy_consumed = 0.0
        self.total_data_transmitted = 0
        self.sleep_energy_savings = 0.0
        self.transmission_efficiency = 0.0
        
    def calculate_energy_per_bit(self):
        """Calculate energy consumed per bit transmitted"""
        if self.total_data_transmitted == 0:
            return float('inf')
        return self.total_energy_consumed / self.total_data_transmitted
    
    def calculate_sleep_efficiency(self):
        """Calculate energy savings from sleep mode"""
        total_possible_energy = self.calculate_total_possible_consumption()
        if total_possible_energy == 0:
            return 0.0
        return self.sleep_energy_savings / total_possible_energy
    
    def update_efficiency_metrics(self, energy_consumed, data_transmitted, sleep_savings):
        """Update energy efficiency metrics"""
        self.total_energy_consumed += energy_consumed
        self.total_data_transmitted += data_transmitted
        self.sleep_energy_savings += sleep_savings
```

**3. Data Delivery Performance:**
```python
class DataDeliveryMetrics:
    def __init__(self):
        self.packets_sent = 0
        self.packets_delivered = 0
        self.total_latency = 0.0
        self.delivery_failures = 0
        
    def calculate_delivery_ratio(self):
        """Calculate packet delivery ratio"""
        if self.packets_sent == 0:
            return 0.0
        return self.packets_delivered / self.packets_sent
    
    def calculate_average_latency(self):
        """Calculate average end-to-end latency"""
        if self.packets_delivered == 0:
            return 0.0
        return self.total_latency / self.packets_delivered
    
    def update_delivery_metrics(self, packet_info):
        """Update data delivery metrics"""
        self.packets_sent += 1
        
        if packet_info.delivered:
            self.packets_delivered += 1
            self.total_latency += packet_info.latency
        else:
            self.delivery_failures += 1
```

### 4.5.2 Learning and Adaptation Metrics

**DQN Learning Performance:**
```python
class LearningMetrics:
    def __init__(self):
        self.episode_rewards = []
        self.learning_losses = []
        self.epsilon_values = []
        self.q_value_evolution = []
        
    def track_learning_progress(self, episode, reward, loss, epsilon, avg_q_value):
        """Track DQN learning progress"""
        self.episode_rewards.append(reward)
        self.learning_losses.append(loss)
        self.epsilon_values.append(epsilon)
        self.q_value_evolution.append(avg_q_value)
        
    def calculate_learning_convergence(self):
        """Calculate learning convergence metrics"""
        if len(self.episode_rewards) < 100:
            return None
        
        recent_rewards = self.episode_rewards[-100:]
        reward_std = np.std(recent_rewards)
        reward_trend = np.polyfit(range(100), recent_rewards, 1)[0]
        
        convergence_metrics = {
            'reward_stability': 1.0 / (1.0 + reward_std),
            'learning_trend': max(0.0, reward_trend),
            'convergence_score': self.calculate_convergence_score()
        }
        
        return convergence_metrics
```

### 4.5.3 Statistical Validation Framework

**Multi-Run Statistical Analysis:**
```python
class StatisticalValidator:
    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level
        self.results_database = []
        
    def add_experiment_result(self, experiment_config, results):
        """Add experiment result to database"""
        result_entry = {
            'config': experiment_config,
            'network_lifetime': results.network_lifetime,
            'first_node_death': results.first_node_death_time,
            'energy_efficiency': results.energy_per_bit,
            'delivery_ratio': results.packet_delivery_ratio,
            'timestamp': time.time()
        }
        self.results_database.append(result_entry)
    
    def perform_statistical_analysis(self, metric_name, group_by='protocol'):
        """Perform statistical analysis on collected results"""
        grouped_data = self.group_results(metric_name, group_by)
        
        analysis_results = {
            'descriptive_stats': self.calculate_descriptive_stats(grouped_data),
            'significance_tests': self.perform_significance_tests(grouped_data),
            'confidence_intervals': self.calculate_confidence_intervals(grouped_data),
            'effect_sizes': self.calculate_effect_sizes(grouped_data)
        }
        
        return analysis_results
    
    def perform_significance_tests(self, grouped_data):
        """Perform statistical significance tests"""
        if len(grouped_data) < 2:
            return None
        
        # Perform t-tests for pairwise comparisons
        protocols = list(grouped_data.keys())
        test_results = {}
        
        for i in range(len(protocols)):
            for j in range(i+1, len(protocols)):
                protocol_a, protocol_b = protocols[i], protocols[j]
                data_a = grouped_data[protocol_a]
                data_b = grouped_data[protocol_b]
                
                # Perform Welch's t-test (unequal variances)
                t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)
                
                test_results[f"{protocol_a}_vs_{protocol_b}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level,
                    'effect_size': self.cohen_d(data_a, data_b)
                }
        
        return test_results
    
    def cohen_d(self, data_a, data_b):
        """Calculate Cohen's d effect size"""
        mean_a, mean_b = np.mean(data_a), np.mean(data_b)
        std_a, std_b = np.std(data_a, ddof=1), np.std(data_b, ddof=1)
        
        # Pooled standard deviation
        n_a, n_b = len(data_a), len(data_b)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        
        return (mean_a - mean_b) / pooled_std
```

## 4.6 Comparative Benchmarking Framework

### 4.6.1 Baseline Protocol Implementation

**EER-RL Protocol Implementation:**
```python
class EERRLProtocol:
    """Energy Efficient Routing using Reinforcement Learning"""
    
    def __init__(self, config):
        self.config = config
        self.q_table = {}  # State-action Q-table
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        
    def select_action(self, state):
        """Select action using epsilon-greedy Q-learning"""
        state_key = self.discretize_state(state)
        
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])  # Random exploration
        
        q_values = self.q_table.get(state_key, [0.0, 0.0, 0.0])
        return np.argmax(q_values)
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning rule"""
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0, 0.0]
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0, 0.0, 0.0]
        
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
    
    def discretize_state(self, state):
        """Convert continuous state to discrete key for Q-table"""
        # Simple discretization strategy
        discretized = []
        for value in state:
            discrete_value = int(value * 10)  # 10 bins per dimension
            discretized.append(min(discrete_value, 9))
        return tuple(discretized)
```

**Traditional Clustering Protocol:**
```python
class TraditionalClustering:
    """Traditional LEACH-based clustering protocol"""
    
    def __init__(self, config):
        self.config = config
        self.cluster_head_probability = 0.1
        self.round_duration = 300  # seconds
        
    def select_cluster_heads(self, nodes, round_number):
        """Select cluster heads using LEACH algorithm"""
        cluster_heads = []
        
        for node in nodes:
            if not node.is_alive():
                continue
            
            # LEACH cluster head selection probability
            threshold = self.calculate_threshold(node, round_number)
            
            if random.random() < threshold:
                cluster_heads.append(node)
        
        return cluster_heads
    
    def calculate_threshold(self, node, round_number):
        """Calculate cluster head selection threshold"""
        if node.was_cluster_head_recently(round_number):
            return 0.0
        
        P = self.cluster_head_probability
        r = round_number % (1.0 / P)
        
        return P / (1.0 - P * (r % (1.0 / P)))
    
    def route_data(self, source_node, network):
        """Route data using traditional clustering"""
        if source_node.is_cluster_head():
            # Cluster head routes to sink
            sink = network.get_sink_node()
            return self.transmit_to_sink(source_node, sink)
        else:
            # Regular node routes to cluster head
            cluster_head = network.get_cluster_head(source_node.cluster_id)
            return self.transmit_to_cluster_head(source_node, cluster_head)
```

### 4.6.2 Performance Comparison Framework

**Comparative Evaluation Protocol:**
```python
class ComparativeEvaluator:
    def __init__(self, protocols, configurations):
        self.protocols = protocols
        self.configurations = configurations
        self.results = {}
        
    def run_comparative_evaluation(self, num_runs=300):
        """Run comparative evaluation across all protocols and configurations"""
        
        for config_name, config in self.configurations.items():
            print(f"Evaluating configuration: {config_name}")
            
            for protocol_name, protocol_class in self.protocols.items():
                print(f"  Testing protocol: {protocol_name}")
                
                protocol_results = []
                
                for run in range(num_runs):
                    # Initialize protocol and network
                    protocol = protocol_class(config)
                    network = NetworkSimulation(config)
                    
                    # Run simulation
                    result = self.run_single_simulation(protocol, network, config)
                    protocol_results.append(result)
                    
                    if (run + 1) % 50 == 0:
                        print(f"    Completed {run + 1}/{num_runs} runs")
                
                # Store results
                self.results[f"{config_name}_{protocol_name}"] = protocol_results
        
        return self.analyze_comparative_results()
    
    def analyze_comparative_results(self):
        """Analyze and compare results across protocols"""
        analysis = {}
        
        for config_name in self.configurations.keys():
            config_analysis = {}
            
            # Extract results for this configuration
            config_results = {
                protocol: self.results[f"{config_name}_{protocol}"]
                for protocol in self.protocols.keys()
            }
            
            # Calculate performance metrics
            for metric in ['network_lifetime', 'first_node_death', 'energy_efficiency']:
                metric_analysis = self.compare_metric(config_results, metric)
                config_analysis[metric] = metric_analysis
            
            analysis[config_name] = config_analysis
        
        return analysis
    
    def compare_metric(self, config_results, metric_name):
        """Compare specific metric across protocols"""
        metric_data = {}
        
        for protocol, results in config_results.items():
            metric_values = [getattr(result, metric_name) for result in results]
            
            metric_data[protocol] = {
                'mean': np.mean(metric_values),
                'std': np.std(metric_values),
                'median': np.median(metric_values),
                'min': np.min(metric_values),
                'max': np.max(metric_values),
                'confidence_interval': self.calculate_confidence_interval(metric_values)
            }
        
        # Calculate relative improvements
        baseline_protocol = 'traditional_clustering'
        if baseline_protocol in metric_data:
            baseline_mean = metric_data[baseline_protocol]['mean']
            
            for protocol in metric_data:
                if protocol != baseline_protocol:
                    improvement = (metric_data[protocol]['mean'] - baseline_mean) / baseline_mean * 100
                    metric_data[protocol]['improvement_percentage'] = improvement
        
        return metric_data
```

## 4.7 Experimental Execution Framework

### 4.7.1 Simulation Control System

**Master Simulation Controller:**
```python
class SimulationController:
    def __init__(self, experiment_config):
        self.experiment_config = experiment_config
        self.current_simulation = None
        self.results_logger = ResultsLogger()
        
    def execute_experiment_suite(self):
        """Execute complete experimental suite"""
        total_experiments = self.calculate_total_experiments()
        completed_experiments = 0
        
        print(f"Starting experimental suite: {total_experiments} total experiments")
        
        for config_name, network_config in self.experiment_config.network_configs.items():
            for protocol_name, protocol_class in self.experiment_config.protocols.items():
                for run_number in range(self.experiment_config.num_runs):
                    
                    print(f"Experiment {completed_experiments + 1}/{total_experiments}: "
                          f"{config_name}_{protocol_name}_run_{run_number}")
                    
                    # Execute single experiment
                    result = self.execute_single_experiment(
                        network_config, protocol_class, run_number
                    )
                    
                    # Log results
                    self.results_logger.log_experiment_result(
                        config_name, protocol_name, run_number, result
                    )
                    
                    completed_experiments += 1
                    
                    # Progress reporting
                    if completed_experiments % 10 == 0:
                        progress = (completed_experiments / total_experiments) * 100
                        print(f"Progress: {progress:.1f}% ({completed_experiments}/{total_experiments})")
        
        print("Experimental suite completed successfully")
        return self.results_logger.generate_final_report()
    
    def execute_single_experiment(self, network_config, protocol_class, run_number):
        """Execute single experimental run"""
        
        # Set random seed for reproducibility
        random.seed(run_number)
        np.random.seed(run_number)
        
        # Initialize network and protocol
        network = NetworkSimulation(network_config)
        protocol = protocol_class(network_config)
        
        # Initialize performance monitors
        energy_monitor = EnergyMonitor(network.nodes)
        performance_monitor = PerformanceMonitor()
        learning_monitor = LearningMetrics()
        
        # Run simulation
        simulation_result = self.run_simulation_loop(
            network, protocol, energy_monitor, performance_monitor, learning_monitor
        )
        
        return simulation_result
    
    def run_simulation_loop(self, network, protocol, energy_monitor, 
                           performance_monitor, learning_monitor):
        """Main simulation execution loop"""
        
        start_time = time.time()
        
        for timestamp in range(network.config.simulation_duration):
            
            # Process network tick
            network.process_tick(timestamp)
            
            # Update node states and make decisions
            for node in network.nodes:
                if node.is_alive():
                    state = network.get_node_state(node)
                    action = protocol.select_action(node, state)
                    reward = network.execute_action(node, action)
                    
                    if hasattr(protocol, 'learn'):
                        protocol.learn(state, action, reward)
            
            # Monitor performance
            energy_monitor.log_energy_state(timestamp)
            performance_monitor.update_metrics(timestamp, network)
            
            if hasattr(protocol, 'get_learning_metrics'):
                learning_metrics = protocol.get_learning_metrics()
                learning_monitor.track_learning_progress(
                    timestamp, learning_metrics.episode_reward,
                    learning_metrics.loss, learning_metrics.epsilon,
                    learning_metrics.avg_q_value
                )
            
            # Check termination conditions
            if self.should_terminate_simulation(network, timestamp):
                break
        
        # Generate simulation result
        simulation_time = time.time() - start_time
        
        result = SimulationResult(
            network_lifetime=timestamp,
            first_node_death=energy_monitor.first_node_death_time,
            energy_efficiency=performance_monitor.calculate_energy_efficiency(),
            delivery_ratio=performance_monitor.calculate_delivery_ratio(),
            learning_convergence=learning_monitor.calculate_learning_convergence(),
            simulation_time=simulation_time
        )
        
        return result
    
    def should_terminate_simulation(self, network, timestamp):
        """Determine if simulation should terminate early"""
        alive_nodes = [node for node in network.nodes if node.is_alive()]
        
        # Terminate if network is partitioned or insufficient nodes
        if len(alive_nodes) < 3:
            return True
        
        # Terminate if sink is disconnected
        if not network.is_sink_connected():
            return True
        
        return False
```

### 4.7.2 Results Collection and Analysis

**Comprehensive Results Logger:**
```python
class ResultsLogger:
    def __init__(self, output_directory="experimental_results"):
        self.output_directory = output_directory
        self.experiment_results = []
        self.raw_data_storage = {}
        
        # Create output directory structure
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(f"{output_directory}/raw_data", exist_ok=True)
        os.makedirs(f"{output_directory}/analysis", exist_ok=True)
        os.makedirs(f"{output_directory}/figures", exist_ok=True)
    
    def log_experiment_result(self, config_name, protocol_name, run_number, result):
        """Log individual experiment result"""
        
        experiment_record = {
            'timestamp': time.time(),
            'config_name': config_name,
            'protocol_name': protocol_name,
            'run_number': run_number,
            'network_lifetime': result.network_lifetime,
            'first_node_death': result.first_node_death,
            'energy_efficiency': result.energy_efficiency,
            'delivery_ratio': result.delivery_ratio,
            'learning_convergence': result.learning_convergence,
            'simulation_time': result.simulation_time
        }
        
        self.experiment_results.append(experiment_record)
        
        # Save raw data
        raw_data_key = f"{config_name}_{protocol_name}"
        if raw_data_key not in self.raw_data_storage:
            self.raw_data_storage[raw_data_key] = []
        
        self.raw_data_storage[raw_data_key].append(result.detailed_data)
        
        # Periodic saves to prevent data loss
        if len(self.experiment_results) % 50 == 0:
            self.save_interim_results()
    
    def generate_final_report(self):
        """Generate comprehensive experimental report"""
        
        print("Generating final experimental report...")
        
        # Save all experimental data
        self.save_all_experimental_data()
        
        # Generate statistical analysis
        statistical_analysis = self.perform_statistical_analysis()
        
        # Generate visualizations
        self.generate_visualization_suite()
        
        # Create summary report
        summary_report = self.create_summary_report(statistical_analysis)
        
        # Save final report
        report_filename = f"{self.output_directory}/EXPERIMENTAL_REPORT_{int(time.time())}.md"
        with open(report_filename, 'w') as f:
            f.write(summary_report)
        
        print(f"Final experimental report saved to: {report_filename}")
        
        return {
            'summary_report': summary_report,
            'statistical_analysis': statistical_analysis,
            'raw_data_location': self.output_directory
        }
```

## 4.8 Chapter Summary

This chapter has presented a comprehensive experimental design and implementation framework for validating the PSWR-DRL system across multiple network configurations and operational scenarios. The experimental methodology encompasses rigorous statistical validation, realistic modeling, and comprehensive performance assessment.

**Key Experimental Design Features:**

1. **Multi-Scale Validation**: Four network configurations (10, 30, 50, 100 nodes) ensuring scalability assessment across diverse deployment scenarios.

2. **Real-World Integration**: Incorporation of actual sensor datasets and hardware-based energy models ensuring realistic evaluation conditions.

3. **Comprehensive Metrics**: Multi-dimensional performance assessment including network lifetime, energy efficiency, data delivery performance, and learning convergence.

4. **Statistical Rigor**: 300 independent simulation runs per configuration with significance testing and confidence interval analysis.

5. **Comparative Benchmarking**: Systematic comparison with EER-RL and traditional clustering protocols using standardized evaluation frameworks.

The experimental framework provides robust validation of the PSWR-DRL system's effectiveness while ensuring statistical significance and practical relevance of the research findings. The implementation details presented enable reproducible research and provide a foundation for future extensions and improvements.

The following chapter will present the comprehensive results obtained through this experimental methodology, demonstrating the significant performance improvements achieved by the PSWR-DRL system across all evaluated metrics and network configurations.
