import os
import sys
import math
import random
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from collections import deque
from datetime import datetime

# Ensure all necessary modules are imported
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import pandas as pd
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install required packages: pip install torch pandas matplotlib numpy")
    sys.exit(1)

# Make sure matplotlib doesn't require a GUI
try:
    plt.switch_backend('agg')
    print("Using matplotlib backend: agg (non-interactive)")
except Exception as e:
    print(f"Warning: Failed to switch matplotlib backend: {e}")
    print("Visualizations might not work correctly.")

#------------------------------------------------------------------------------
# CONFIGURATION SETTINGS
#------------------------------------------------------------------------------

# Network parameters
NUM_NODES = 100
NUM_CLUSTERS = 40
SEND_RANGE = 30  # meters
MAX_LONGITUDE = 200.0  # meters
MAX_LATITUDE = 200.0  # meters

# Energy parameters (adjusted for more realistic WSN behavior)
NODE_INITIAL_ENERGY = 300  # Joules (to match competitor's 300 ± 20 range for 100 nodes)
POWER_SEND = 0.5  # Joules per transmission (increased for 100-node network)
POWER_RECEIVE = 0.3  # Joules per reception (increased for 100-node network)
POWER_ACTIVE = 0.15  # Joules per second for active nodes (increased for 100-node network)
POWER_SLEEP = 0.08  # Joules per second for sleeping nodes (increased for 100-node network)

# Routing parameters
ALPHA = 0.5  # Learning rate
DFR_MIN = 5.0
DFR_MAX = 55.0

# Sleep scheduling parameters (node-specific diversity)
SLEEP_RESTRICT_THRESHOLD = 5  # Base threshold - nodes will vary this
SLEEP_DURATION = 30  # Base duration - nodes will vary this
WAKE_UP_CHECK_INTERVAL = 30  # seconds

# Data transmission parameters (more sensitive to create diversity)
CHANGE_THRESHOLD = 1.0  # Base threshold - nodes will vary this
TRANSMISSION_PERIOD = 6  # seconds (from dataset)

# Simulation parameters
TOTAL_SIMULATION_TIME = 5000  # seconds

# DQL parameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
MEMORY_SIZE = 10000

#------------------------------------------------------------------------------
# UTILITY FUNCTIONS
#------------------------------------------------------------------------------

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def normalize_distance(distance, max_distance):
    """Normalize distance value to [0,1] range"""
    if max_distance == 0:
        return 0
    return distance / max_distance

def calculate_n_parameter(normalized_distance, dfr_min, dfr_max):
    """Calculate n parameter based on normalized distance and DFR range"""
    return normalized_distance * (dfr_max - dfr_min) + dfr_min

def calculate_reward(energy, distance, hop_count, n_param):
    """Calculate reward for RL-based routing"""
    if distance == 0:
        distance = 0.1  # Avoid division by zero
    
    if hop_count == 0:
        hop_count = 1  # Avoid division by zero
    
    return energy / ((distance ** n_param) * hop_count)

def update_q_value(old_q, reward, best_next_q, alpha):
    """Update Q-value based on reward and next best Q-value"""
    return (1 - alpha) * old_q + alpha * (reward + best_next_q)

#------------------------------------------------------------------------------
# DATASET LOADER
#------------------------------------------------------------------------------

class DatasetLoader:
    """Class for loading and processing the WSN dataset"""
    
    def __init__(self, dataset_path):
        """Initialize the dataset loader with the path to the dataset directory"""
        self.dataset_path = dataset_path
        self.node_data = {}
        self.dataset_info = self._get_dataset_info()
    
    def _get_dataset_info(self):
        """Extract basic dataset information"""
        info = {
            'name': 'wsn-indfeat-dataset',
            'transmission_period': 6,
            'columns': {
                'rssi_mean': 0,
                'rssi_std': 1,
                'lqi_mean': 2,
                'lqi_std': 3,
                'noise_mean': 4,
                'noise_std': 5,
                'tx_rate': 6,
                'rx_rate': 7,
                'path_length_mean': 8,
                'path_length_std': 9,
                'estimated_prr': 10,
                'temperature_mean': 11,
                'temperature_std': 12,
                'humidity_mean': 13,
                'humidity.std': 14,
                'voltage_mean': 15,
                'voltage_std': 16,
                'recorded_prr': 17
            }
        }
        
        # Count actual available nodes in dataset
        count = 0
        while True:
            file_path = os.path.join(self.dataset_path, f"node{count+1}.csv")
            if os.path.exists(file_path):
                count += 1
            else:
                break
        
        info['available_nodes'] = count
        return info
    
    def load_node_data(self, node_id):
        """Load data for a specific node"""
        file_path = os.path.join(self.dataset_path, f"node{node_id}.csv")
        if os.path.exists(file_path):
            try:
                data = pd.read_csv(file_path, header=None)
                self.node_data[node_id] = data
                return data
            except Exception as e:
                print(f"Error loading data for node {node_id}: {e}")
                self.node_data[node_id] = None
                return None
        else:
            # Use modular arithmetic to reuse existing dataset files
            # Map node_id to available files (node1.csv to node10.csv)
            mapped_node_id = ((node_id - 1) % 10) + 1
            mapped_file_path = os.path.join(self.dataset_path, f"node{mapped_node_id}.csv")
            
            if os.path.exists(mapped_file_path):
                try:
                    data = pd.read_csv(mapped_file_path, header=None)
                    self.node_data[node_id] = data
                    return data
                except Exception as e:
                    print(f"Error loading data for node {node_id} (mapped to node{mapped_node_id}): {e}")
                    self.node_data[node_id] = None
                    return None
            else:
                print(f"WARNING: Dataset file not found: {file_path}")
                self.node_data[node_id] = None
                return None
    
    def load_all_nodes(self, num_nodes):
        """Load data for all nodes up to num_nodes"""
        loaded_count = 0
        for i in range(1, num_nodes + 1):
            data = self.load_node_data(i)
            if data is not None:
                loaded_count += 1
        
        print(f"Successfully loaded data for {loaded_count}/{num_nodes} nodes")
        return self.node_data
    
    def get_sensor_reading(self, node_id, time_index, column='temperature_mean'):
        """Get a sensor reading for a specific node at a specific time index with enhanced data generation for 100 nodes"""
        if node_id not in self.node_data:
            self.load_node_data(node_id)
        
        if node_id not in self.node_data or self.node_data[node_id] is None:
            # Enhanced data generation for 100-node network
            base_node_id = ((node_id - 1) % 10) + 1  # Map to nodes 1-10
            
            # Load base node data if available
            if base_node_id not in self.node_data:
                self.load_node_data(base_node_id)
            
            if base_node_id in self.node_data and self.node_data[base_node_id] is not None:
                # Get base value from existing node
                col_idx = self.dataset_info['columns'].get(column, 11)
                if time_index >= len(self.node_data[base_node_id]):
                    time_index = time_index % len(self.node_data[base_node_id])
                base_value = self.node_data[base_node_id].iloc[time_index, col_idx]
                
                # Apply variations for 100-node network
                seasonal_variation = math.cos(time_index * 0.01) * 1.5  # Seasonal pattern
                daily_variation = math.sin(time_index * 0.1) * 2  # Daily pattern
                node_offset = (node_id % 101) * 0.25  # Node-specific offset for 100 nodes
                
                return base_value + seasonal_variation + daily_variation + node_offset
            else:
                # Fallback random generation with realistic patterns
                base_temp = 25 + (node_id % 10) * 0.5  # Node-specific base
                daily_cycle = 5 * math.sin(time_index * 0.2)  # Daily temperature cycle
                noise = random.uniform(-2, 2)  # Random noise
                return base_temp + daily_cycle + noise
        
        # Get column index from dataset_info
        col_idx = self.dataset_info['columns'].get(column, 11)  # Default to temperature_mean
        
        # Make sure time_index is within bounds
        if time_index >= len(self.node_data[node_id]):
            time_index = time_index % len(self.node_data[node_id])
        
        return self.node_data[node_id].iloc[time_index, col_idx]
    
    def get_path_length(self, node_id, time_index):
        """Get path length for a specific node at a specific time index"""
        return self.get_sensor_reading(node_id, time_index, 'path_length_mean')
    
    def get_link_quality(self, node_id, time_index):
        """Get link quality for a specific node at a specific time index"""
        return self.get_sensor_reading(node_id, time_index, 'lqi_mean')
    
    def get_voltage(self, node_id, time_index):
        """Get voltage for a specific node at a specific time index"""
        return self.get_sensor_reading(node_id, time_index, 'voltage_mean')

#------------------------------------------------------------------------------
# DEEP Q-LEARNING IMPLEMENTATION
#------------------------------------------------------------------------------

class ReplayMemory:
    """Experience replay memory for DQL agent"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Save a transition to memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, min(len(self.memory), batch_size))
    
    def __len__(self):
        return len(self.memory)

class DQLNetwork(nn.Module):
    """Neural network for DQL"""
    
    def __init__(self, input_size, output_size, hidden_size=64):
        super(DQLNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQLAgent:
    """Deep Q-Learning Agent for RLBEEP"""
    
    def __init__(self, state_size, action_size):
        """Initialize DQL agent with state and action space size"""
        self.state_size = state_size
        self.action_size = action_size
        
        # DQL parameters
        self.gamma = GAMMA
        self.epsilon = EPS_START
        self.epsilon_min = EPS_END
        self.epsilon_decay = EPS_DECAY
        self.batch_size = BATCH_SIZE
        self.target_update = TARGET_UPDATE
        
        # Networks
        self.policy_net = DQLNetwork(state_size, action_size)
        self.target_net = DQLNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Replay memory
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Tracking variables
        self.steps_done = 0
        self.episode_rewards = []
    
    def select_action(self, state):
        """Select action based on epsilon-greedy policy"""
        sample = random.random()
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
                        math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                return self.policy_net(state_tensor).max(1)[1].item()
        else:
            return random.randrange(self.action_size)
    
    def memorize(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Experience replay training"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor([s[0] for s in batch])
        actions = torch.LongTensor([[s[1]] for s in batch])
        rewards = torch.FloatTensor([[s[2]] for s in batch])
        next_states = torch.FloatTensor([s[3] for s in batch])
        dones = torch.FloatTensor([[s[4]] for s in batch])
        
        # Compute Q(s_t, a)
        curr_q = self.policy_net(states).gather(1, actions)
        
        # Compute max Q(s_{t+1}, a) for all next states
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1).detach()
        
        # Compute expected Q values
        expected_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.SmoothL1Loss()(curr_q, expected_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

#------------------------------------------------------------------------------
# SIMULATION COMPONENTS
#------------------------------------------------------------------------------

class NodeMode(Enum):
    """Enum for node operation modes"""
    ACTIVE = 1
    SLEEP = 2

class NodeType(Enum):
    """Enum for node types"""
    REGULAR = 1
    CLUSTER_HEAD = 2
    SINK = 3

class Packet:
    """Class representing a data packet in the WSN"""
    def __init__(self, source_id, data, sender_id=None, destination_id=None):
        self.source_id = source_id
        self.sender_id = sender_id if sender_id is not None else source_id
        self.destination_id = destination_id
        self.data = data
        self.hops = 0
        self.created_time = 0
        self.energy_cost = 0.0

class Node:
    """Base class representing a node in the WSN"""
    def __init__(self, node_id, x, y, initial_energy=NODE_INITIAL_ENERGY):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.energy = initial_energy
        self.initial_energy = initial_energy
        self.type = NodeType.REGULAR
        self.mode = NodeMode.ACTIVE
        self.cluster_id = None
        self.neighbors = {}  # neighbor_id -> (hop_count, energy)
        self.send_queue = deque()
        self.received_data = {}
        self.last_data = None
        self.no_send_count = 0
        self.last_active_time = 0
        self.sleep_until = 0
        self.transmit_count = 0
        self.receive_count = 0
        self.data_history = []
        
        # DQL state variables
        self.current_state = None
        self.previous_state = None
        self.previous_action = None
        
        # Create DQL agent for routing decisions
        self.state_size = 9  # Enhanced state representation
        self.action_size = 3  # Forward to CH, Forward to Sink, Sleep
        self.dql_agent = DQLAgent(self.state_size, self.action_size)
        
    def copy(self):
        """Create a copy of this node"""
        new_node = Node(self.node_id, self.x, self.y, self.energy)
        new_node.initial_energy = self.initial_energy
        new_node.type = self.type
        new_node.mode = self.mode
        new_node.cluster_id = self.cluster_id
        new_node.neighbors = self.neighbors.copy()
        new_node.send_queue = copy.deepcopy(self.send_queue)
        new_node.received_data = self.received_data.copy()
        new_node.last_data = self.last_data
        new_node.no_send_count = self.no_send_count
        new_node.last_active_time = self.last_active_time
        new_node.sleep_until = self.sleep_until
        new_node.transmit_count = self.transmit_count
        new_node.receive_count = self.receive_count
        new_node.data_history = self.data_history.copy() if self.data_history else []
        new_node.current_state = new_node.current_state.copy() if isinstance(new_node.current_state, np.ndarray) else new_node.current_state
        new_node.previous_state = new_node.previous_state.copy() if isinstance(new_node.previous_state, np.ndarray) else new_node.previous_state
        new_node.previous_action = self.previous_action
        
        # Copy DQL agent if it exists
        if hasattr(self, 'dql_agent') and self.dql_agent is not None:
            new_node.dql_agent = DQLAgent(self.state_size, self.action_size)
            # Only copy essential state, not the neural networks
            new_node.dql_agent.epsilon = self.dql_agent.epsilon
            new_node.dql_agent.steps_done = self.dql_agent.steps_done
            
        return new_node
        
    def distance_to(self, other_node):
        """Calculate distance to another node"""
        return calculate_distance(self.x, self.y, other_node.x, other_node.y)
    
    def is_alive(self):
        """Check if node has energy"""
        return self.energy > 0
    
    def reduce_energy(self, activity_type):
        """Reduce energy based on activity type with node-specific diversity"""
        if not self.is_alive():
            return False
        
        # Define energy consumption for different activities
        if activity_type == "send":
            energy_cost = POWER_SEND
        elif activity_type == "receive":
            energy_cost = POWER_RECEIVE
        elif activity_type == "active":
            energy_cost = POWER_ACTIVE
        elif activity_type == "sleep":
            energy_cost = POWER_SLEEP
        else:
            energy_cost = 0
        
        # Node-specific energy efficiency (creates diversity)
        node_efficiency = 1 + (self.node_id % 7) * 0.05  # 0% to 30% variation based on node ID
        energy_cost *= node_efficiency
        
        # Minimal base energy consumption (much reduced to make sleep effective)
        base_energy_loss = POWER_ACTIVE * 0.001  # Very small base consumption
        
        # Node-specific base energy loss (creates more diversity)
        node_base_factor = 1 + (self.node_id % 5) * 0.2  # 0% to 80% variation
        base_energy_loss *= node_base_factor
        
        # Add slight time-based degradation (much reduced)
        if hasattr(self, 'simulation_time_ratio'):
            time_degradation_factor = 1 + (self.simulation_time_ratio * 0.01)  # Max 1% increase
            base_energy_loss *= time_degradation_factor
        
        # Apply base energy loss based on activity type
        if activity_type == "sleep":
            # Sleeping nodes consume much less base energy
            energy_cost += base_energy_loss * 0.05  # Only 5% of base loss for sleeping
        else:
            # Active/send/receive get full base energy loss
            energy_cost += base_energy_loss
        
        # Reduce energy
        self.energy -= energy_cost
        
        # Ensure energy doesn't go below 0
        if self.energy < 0:
            self.energy = 0
        
        return self.energy > 0
    
    def update_mode(self, send_permission, sleep_threshold, current_time=None):
        """Update node mode based on activity with enhanced node-specific behavior"""
        if current_time is None:
            current_time = 0
        
        # Check if it's time to wake up
        if self.mode == NodeMode.SLEEP and current_time >= self.sleep_until:
            self.mode = NodeMode.ACTIVE
            self.no_send_count = 0
        
        # Update no-send counter
        if not send_permission:
            self.no_send_count += 1
        else:
            self.no_send_count = 0
        
        # Node-specific sleep threshold (more diverse)
        node_sleep_threshold = sleep_threshold + (self.node_id % 5)  # Add 0-4 to threshold
        
        # Check if should go to sleep
        if self.mode == NodeMode.ACTIVE and self.no_send_count >= node_sleep_threshold:
            self.mode = NodeMode.SLEEP
            
            # Node-specific sleep duration (more diverse)
            sleep_variation = 1 + (self.node_id % 7) * 0.15  # 0-90% variation
            sleep_duration = int(SLEEP_DURATION * sleep_variation)
            
            # Add some randomness to break synchronization
            sleep_duration += random.randint(-5, 5)
            sleep_duration = max(10, sleep_duration)  # Minimum 10 seconds
            
            self.sleep_until = current_time + sleep_duration
        
        return self.mode
    
    def should_transmit(self, sensor_data, change_threshold):
        """Decide if node should transmit based on data changes with enhanced node-specific sensitivity"""
        if self.last_data is None:
            self.last_data = sensor_data
            return True
        
        # Node-specific sensitivity (more diverse)
        node_sensitivity = 1 + (self.node_id % 6) * 0.3  # 0% to 150% variation
        node_threshold = change_threshold * node_sensitivity
        
        # Check if data has changed significantly
        data_diff = abs(sensor_data - self.last_data)
        should_send = data_diff > node_threshold
        
        # Add some probabilistic transmission to break perfect synchronization
        if not should_send and random.random() < 0.02:  # 2% chance of random transmission
            should_send = True
        
        if should_send:
            self.last_data = sensor_data
        
        return should_send

    def get_state(self, simulation):
        """Get enhanced state representation for DQL"""
        # Enhanced state with more features
        
        # 1. Normalized energy level
        energy_level = self.energy / self.initial_energy
        
        # 2. Normalized distance to cluster head
        ch_distance = 1.0  # Default max distance
        if self.cluster_id is not None and self.cluster_id < len(simulation.cluster_heads):
            ch = simulation.cluster_heads[self.cluster_id]
            ch_distance = min(1.0, self.distance_to(ch) / SEND_RANGE)
        
        # 3. Normalized distance to sink
        sink_distance = min(1.0, self.distance_to(simulation.sink) / (MAX_LONGITUDE + MAX_LATITUDE))
        
        # 4. Hop count to sink (normalized)
        hop_count = 1.0
        for neighbor_id, (hops, _) in self.neighbors.items():
            hop_count = min(hop_count, hops / 10.0)  # Normalize by assuming max 10 hops
        
        # 5. Data transmission urgency (based on data change)
        data_urgency = 0.0
        if self.last_data is not None and len(self.data_history) > 0:
            data_change = abs(self.last_data - self.data_history[-1]) / CHANGE_THRESHOLD
            data_urgency = min(1.0, data_change)
        
        # 6. Network congestion (approximated by queue length)
        congestion = min(1.0, len(self.send_queue) / 10.0)
        
        # 7. Sleep pressure (based on no_send_count)
        sleep_pressure = min(1.0, self.no_send_count / SLEEP_RESTRICT_THRESHOLD)
        
        # 8. Cluster health (based on CH energy)
        cluster_health = 0.5  # Default medium health
        if self.cluster_id is not None and self.cluster_id < len(simulation.cluster_heads):
            ch = simulation.cluster_heads[self.cluster_id]
            if ch.initial_energy > 0:
                cluster_health = ch.energy / ch.initial_energy
            else:
                cluster_health = 0.0  # Dead cluster head
        
        # 9. Time of day factor (diurnal pattern simulation)
        time_factor = (simulation.time % 300) / 300.0  # Cycles every 300 seconds
        
        return [energy_level, ch_distance, sink_distance, hop_count, 
                data_urgency, congestion, sleep_pressure, cluster_health, time_factor]

    def update_dql(self, simulation, reward, done):
        """Update DQL agent with experience"""
        if self.previous_state is not None and self.current_state is not None:
            self.dql_agent.memorize(
                self.previous_state,
                self.previous_action,
                reward,
                self.current_state,
                done
            )
            self.dql_agent.replay()
        
        # Periodically update target network
        if simulation.time % 50 == 0:
            self.dql_agent.update_target_network()

class ClusterHead(Node):
    """Class representing a cluster head node"""
    def __init__(self, node_id, x, y, initial_energy=NODE_INITIAL_ENERGY):
        super().__init__(node_id, x, y, initial_energy)
        self.type = NodeType.CLUSTER_HEAD
        self.cluster_members = []
        self.aggregated_data = {}
        self.routing_table = {}  # destination_id -> next_hop_id
        
    def copy(self):
        """Create a copy of this cluster head"""
        new_ch = super().copy()
        new_ch.__class__ = ClusterHead
        new_ch.type = NodeType.CLUSTER_HEAD
        new_ch.cluster_members = self.cluster_members.copy()
        new_ch.aggregated_data = self.aggregated_data.copy()
        new_ch.routing_table = self.routing_table.copy()
        return new_ch
    
    def aggregate_data(self, source_id, data):
        """Aggregate data from cluster members"""
        # Store data with source
        self.aggregated_data[source_id] = data
        
        # Calculate aggregate (e.g., average)
        if self.aggregated_data:
            return sum(self.aggregated_data.values()) / len(self.aggregated_data)
        return 0
    
    def update_neighbor_table(self, neighbor_id, hop_count, energy):
        """Update neighbor information"""
        self.neighbors[neighbor_id] = (hop_count, energy)
    
    def reduce_energy(self, activity_type):
        """Cluster heads have slightly higher energy consumption but not too much"""
        # Call parent method first
        success = super().reduce_energy(activity_type)
        
        # Reduced additional energy consumption for cluster head responsibilities
        if success and self.is_alive():
            # Cluster heads consume extra energy for coordination and data aggregation
            extra_consumption = POWER_ACTIVE * 0.02  # Reduced from 0.05 to 0.02
            
            # Scale with time to prevent long-term survival
            if hasattr(self, 'simulation_time_ratio'):
                time_factor = 1 + (self.simulation_time_ratio * 0.05)  # Reduced from 0.1
                extra_consumption *= time_factor
            
            self.energy -= extra_consumption
            if self.energy < 0:
                self.energy = 0
        
        return self.energy > 0

class SinkNode(Node):
    """Class representing the sink node"""
    def __init__(self, node_id, x, y):
        super().__init__(node_id, x, y, float('inf'))  # Sink has infinite energy
        self.type = NodeType.SINK
        self.received_packets = []
        self.total_data_received = 0
        self.data_by_node = {}  # node_id -> list of data points
        
    def copy(self):
        """Create a copy of this sink node"""
        new_sink = super().copy()
        new_sink.__class__ = SinkNode
        new_sink.type = NodeType.SINK
        new_sink.received_packets = self.received_packets.copy()
        new_sink.total_data_received = self.total_data_received
        new_sink.data_by_node = copy.deepcopy(self.data_by_node)
        return new_sink
    
    def receive_packet(self, packet):
        """Process received packet"""
        self.received_packets.append(packet)
        self.total_data_received += 1
        
        if packet.source_id not in self.data_by_node:
            self.data_by_node[packet.source_id] = []
        
        self.data_by_node[packet.source_id].append(packet.data)
        return True

class RLBEEPSimulation:
    """Main simulation class for RLBEEP"""
    def __init__(self, dataset_path, num_nodes=NUM_NODES, num_clusters=NUM_CLUSTERS, 
                 max_longitude=MAX_LONGITUDE, max_latitude=MAX_LATITUDE, 
                 send_range=SEND_RANGE, alpha=ALPHA,
                 dfr_min=DFR_MIN, dfr_max=DFR_MAX, 
                 total_time=TOTAL_SIMULATION_TIME,
                 sleep_threshold=SLEEP_RESTRICT_THRESHOLD,
                 change_threshold=CHANGE_THRESHOLD,
                 ch_rotation_interval=300,
                 debug=False):
        
        self.dataset_path = dataset_path
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.max_longitude = max_longitude
        self.max_latitude = max_latitude
        self.send_range = send_range
        self.alpha = alpha
        self.dfr_min = dfr_min
        self.dfr_max = dfr_max
        self.total_time = total_time
        self.sleep_threshold = sleep_threshold
        self.change_threshold = change_threshold
        self.ch_rotation_interval = ch_rotation_interval
        self.debug = debug
        
        # Simulation state
        self.time = 0
        self.nodes = []
        self.cluster_heads = []
        self.sink = None
        self.dataset_loader = None
        self.live_node_count = []
        self.live_node_percentage = []
        self.energy_levels = []
        self.transmissions = []
        self.first_death_time = -1  # -1 means no deaths
        
        # Enhanced tracking for individual nodes
        self.node_death_times = {}  # node_id -> death_time
        self.node_lifetime_data = []  # List of dicts with detailed node info per time step
        self.individual_node_energy = {}  # node_id -> [energy_over_time]
        
        # Initialize individual node tracking
        for i in range(1, self.num_nodes + 1):
            self.individual_node_energy[i] = []
        
        # Setup results directory
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        self._ensure_results_dir()
        
        # Initialize simulation components
        self.load_dataset()
        self.setup_network()
        self.initialize_hop_counts()
    
    def _ensure_results_dir(self):
        """Ensure the results directory exists"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"Created results directory at {self.results_dir}")
    
    def load_dataset(self):
        """Load dataset for simulation"""
        self.dataset_loader = DatasetLoader(self.dataset_path)
        self.dataset_loader.load_all_nodes(self.num_nodes)
    
    def setup_network(self):
        """Set up network with nodes, cluster heads, and sink"""
        # Create nodes with strategic positioning
        node_positions = self.generate_strategic_node_positions()
        
        # Create regular nodes
        for i in range(1, self.num_nodes + 1):
            x, y = node_positions[i-1]
            node = Node(i, x, y)
            self.nodes.append(node)
        
        # Select and create cluster heads (from regular nodes)
        ch_indices = self.select_optimal_cluster_heads()
        
        for i, idx in enumerate(ch_indices):
            # Convert the regular node to a cluster head
            node = self.nodes[idx]
            ch = ClusterHead(node.node_id, node.x, node.y, node.initial_energy)
            
            # Replace the regular node with cluster head
            self.nodes[idx] = ch
            self.cluster_heads.append(ch)
        
        # Create sink node (positioned optimally near center)
        sink_x = self.max_longitude / 2
        sink_y = self.max_latitude / 2
        self.sink = SinkNode(0, sink_x, sink_y)
        
        # Make sure sink is connected to at least one cluster head
        connected_to_sink = False
        for ch in self.cluster_heads:
            if ch.distance_to(self.sink) <= self.send_range:
                connected_to_sink = True
                break
        
        # If no cluster head is connected to sink, adjust the send range or sink position
        if not connected_to_sink and self.cluster_heads:
            # Find closest cluster head to sink
            closest_ch = min(self.cluster_heads, key=lambda ch: ch.distance_to(self.sink))
            min_distance = closest_ch.distance_to(self.sink)
            
            # Option 1: Increase send range slightly if it's close enough
            if min_distance <= self.send_range * 1.5:
                self.send_range = min_distance * 1.1  # Add 10% margin
                print(f"Adjusted send range to {self.send_range:.2f}m to ensure connectivity")
            # Option 2: Move sink closer to a cluster head
            else:
                # Move sink 80% of the way towards the closest cluster head
                sink_x = self.sink.x + 0.8 * (closest_ch.x - self.sink.x)
                sink_y = self.sink.y + 0.8 * (closest_ch.y - self.sink.y)
                self.sink = SinkNode(0, sink_x, sink_y)
                print(f"Moved sink to position ({sink_x:.2f}, {sink_y:.2f}) to ensure connectivity")
        
        # Assign nodes to clusters
        self.assign_nodes_to_clusters()
    
    def generate_strategic_node_positions(self):
        """Generate strategic positions for nodes for better coverage"""
        positions = []
        
        # Generate positions with some structure for better connectivity
        # Mix of grid-based positions with some randomness
        
        # Create a loose grid
        grid_spacing_x = self.max_longitude / (math.sqrt(self.num_nodes) + 1)
        grid_spacing_y = self.max_latitude / (math.sqrt(self.num_nodes) + 1)
        
        for i in range(self.num_nodes):
            # Calculate grid position
            grid_x = (i % int(math.sqrt(self.num_nodes))) * grid_spacing_x + grid_spacing_x
            grid_y = (i // int(math.sqrt(self.num_nodes))) * grid_spacing_y + grid_spacing_y
            
            # Add some randomness
            x = grid_x + random.uniform(-grid_spacing_x/3, grid_spacing_x/3)
            y = grid_y + random.uniform(-grid_spacing_y/3, grid_spacing_y/3)
            
            # Ensure within bounds
            x = max(0, min(x, self.max_longitude))
            y = max(0, min(y, self.max_latitude))
            
            positions.append((x, y))
        
        return positions
    
    def select_optimal_cluster_heads(self):
        """Select optimal cluster head positions based on coverage"""
        # Use K-means-like approach to find good CH positions
        # Here we'll simply select nodes that are well-distributed
        
        # First, divide the area into regions
        ch_indices = []
        region_width = self.max_longitude / math.sqrt(self.num_clusters)
        region_height = self.max_latitude / math.sqrt(self.num_clusters)
        
        for i in range(self.num_clusters):
            region_x = (i % int(math.sqrt(self.num_clusters))) * region_width
            region_y = (i // int(math.sqrt(self.num_clusters))) * region_height
            
            # Find node closest to center of this region
            center_x = region_x + region_width/2
            center_y = region_y + region_height/2
            
            closest_idx = -1
            closest_dist = float('inf')
            
            for j in range(len(self.nodes)):
                if j in ch_indices:
                    continue
                    
                node = self.nodes[j]
                dist = calculate_distance(node.x, node.y, center_x, center_y)
                
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = j
            
            if closest_idx != -1:
                ch_indices.append(closest_idx)
        
        # If we couldn't find enough, add random ones
        while len(ch_indices) < self.num_clusters:
            idx = random.randint(0, len(self.nodes)-1)
            if idx not in ch_indices:
                ch_indices.append(idx)
        
        return ch_indices
    
    def assign_nodes_to_clusters(self):
        """Assign each node to nearest cluster head"""
        for node in self.nodes:
            if node.type == NodeType.CLUSTER_HEAD:
                continue
            
            # Find closest cluster head
            closest_ch_idx = -1
            closest_dist = float('inf')
            
            for i, ch in enumerate(self.cluster_heads):
                dist = node.distance_to(ch)
                if dist < closest_dist and dist <= self.send_range:
                    closest_dist = dist
                    closest_ch_idx = i
            
            # Assign to cluster if within range
            if closest_ch_idx != -1:
                node.cluster_id = closest_ch_idx
                self.cluster_heads[closest_ch_idx].cluster_members.append(node.node_id)
            else:
                # Node is not within range of any CH, leave unassigned
                node.cluster_id = None
    
    def initialize_hop_counts(self):
        """Initialize hop counts to sink for all nodes"""
        # Set sink's hop count
        connected_chs = 0
        for ch in self.cluster_heads:
            dist_to_sink = ch.distance_to(self.sink)
            if dist_to_sink <= self.send_range:
                ch.neighbors[self.sink.node_id] = (1, float('inf'))  # 1 hop, infinite energy
                connected_chs += 1
        
        print(f"Connected cluster heads to sink: {connected_chs}/{len(self.cluster_heads)}")
        if connected_chs == 0 and self.cluster_heads:
            # Debug info: print distances to help diagnose
            for i, ch in enumerate(self.cluster_heads):
                dist = ch.distance_to(self.sink)
                print(f"  CH {ch.node_id} distance to sink: {dist:.2f}m (Send range: {self.send_range:.2f}m)")
        
        # For each cluster head, calculate hop counts for members
        for i, ch in enumerate(self.cluster_heads):
            # All members are 1 hop from their CH
            for member_id in ch.cluster_members:
                node = next((n for n in self.nodes if n.node_id == member_id), None)
                if node:
                    node.neighbors[ch.node_id] = (1, ch.energy)
                    
                    # If CH is 1 hop from sink, member is 2 hops
                    if self.sink.node_id in ch.neighbors:
                        node.neighbors[self.sink.node_id] = (2, float('inf'))
    
    def rotate_cluster_heads(self):
        """Rotate cluster heads to balance energy consumption while maintaining network connectivity"""
        # Skip rotation if too early
        if self.time < self.ch_rotation_interval:
            return
        
        if self.time % self.ch_rotation_interval != 0:
            return
            
        print(f"[Time {self.time}] Rotating cluster heads...")
        
        # Save original state in case we need to revert
        old_chs = self.cluster_heads.copy()
        old_nodes = [node.copy() if hasattr(node, 'copy') else copy.deepcopy(node) for node in self.nodes]
        
        # Find which cluster heads are currently connected to the sink
        connected_ch_indices = []
        for i, ch in enumerate(self.cluster_heads):
            if ch.distance_to(self.sink) <= self.send_range:
                connected_ch_indices.append(i)
                
        if self.debug:
            print(f"Currently {len(connected_ch_indices)} cluster heads connected to sink: {connected_ch_indices}")
        
        # Always keep at least one connected cluster head if possible
        preserve_connected_ch = None
        if connected_ch_indices:
            # Choose the connected CH with highest energy to preserve
            preserve_idx = connected_ch_indices[0]
            max_energy = self.cluster_heads[preserve_idx].energy
            
            for idx in connected_ch_indices:
                if self.cluster_heads[idx].energy > max_energy:
                    max_energy = self.cluster_heads[idx].energy
                    preserve_idx = idx
                    
            preserve_connected_ch = self.cluster_heads[preserve_idx]
            if self.debug:
                print(f"Preserving connected CH {preserve_connected_ch.node_id} with energy {preserve_connected_ch.energy:.2f}")
        
        # Temporary lists for new configuration
        new_ch_indices = []
        self.cluster_heads = []
        
        # Keep track of the closest CH to sink to guarantee connectivity
        min_distance_to_sink = float('inf')
        closest_ch_idx = -1
        
        # For each cluster, find a member with highest energy
        for i, old_ch in enumerate(old_chs):
            # If this is a preserved connected cluster head, keep it
            if preserve_connected_ch and old_ch.node_id == preserve_connected_ch.node_id:
                self.cluster_heads.append(old_ch)  # Keep the existing CH
                continue
                
            # Convert old CH back to regular node
            old_ch_idx = next((j for j, node in enumerate(self.nodes) 
                             if node.node_id == old_ch.node_id), None)
                             
            if old_ch_idx is not None:
                # Create a new regular node to replace CH
                regular_node = Node(old_ch.node_id, old_ch.x, old_ch.y, old_ch.energy)
                self.nodes[old_ch_idx] = regular_node
            
            # Find highest energy node in this cluster with preference for connectivity
            highest_energy = -1
            highest_energy_idx = -1
            
            # First pass: try to find nodes that can connect to sink
            for j, node in enumerate(self.nodes):
                if (node.cluster_id == i and node.energy > highest_energy and 
                    j not in new_ch_indices and node.type != NodeType.CLUSTER_HEAD):
                    dist_to_sink = calculate_distance(node.x, node.y, self.sink.x, self.sink.y)
                    if dist_to_sink <= self.send_range:
                        highest_energy = node.energy
                        highest_energy_idx = j
                        # Track closest to sink
                        if dist_to_sink < min_distance_to_sink:
                            min_distance_to_sink = dist_to_sink
                            closest_ch_idx = j
            
            # If no sink-connected node found, fall back to highest energy
            if highest_energy_idx == -1:
                for j, node in enumerate(self.nodes):
                    if (node.cluster_id == i and node.energy > highest_energy and 
                        j not in new_ch_indices and node.type != NodeType.CLUSTER_HEAD):
                        highest_energy = node.energy
                        highest_energy_idx = j
                        # Track distance to sink for this candidate
                        dist_to_sink = calculate_distance(node.x, node.y, self.sink.x, self.sink.y)
                        if dist_to_sink < min_distance_to_sink:
                            min_distance_to_sink = dist_to_sink
            
            # If found, convert to CH
            if highest_energy_idx != -1:
                node = self.nodes[highest_energy_idx]
                new_ch = ClusterHead(node.node_id, node.x, node.y, node.energy)
                self.nodes[highest_energy_idx] = new_ch
                self.cluster_heads.append(new_ch)
                new_ch_indices.append(highest_energy_idx)
            else:
                # If no suitable node found, reuse the original CH but reduce its energy
                # This keeps the node count consistent rather than creating new nodes
                old_ch.energy *= 0.7  # Reduce energy but don't create a new node
                self.cluster_heads.append(old_ch)
                print(f"Keeping CH {old_ch.node_id} as cluster head (no suitable replacement found)")
                
                # This CH is now the one we're checking distance for
                new_ch = old_ch
                
                # Check if this one is closest to sink
                dist_to_sink = new_ch.distance_to(self.sink)
                if dist_to_sink < min_distance_to_sink:
                    min_distance_to_sink = dist_to_sink
        
        # Reassign nodes to clusters
        self.assign_nodes_to_clusters()
        
        # Re-initialize hop counts
        self.initialize_hop_counts()
        
        # Check if we have at least one cluster head connected to sink
        connected_to_sink = any(ch.distance_to(self.sink) <= self.send_range for ch in self.cluster_heads)
        
        # If no CH is connected to sink but we identified the closest one, adjust the network
        if not connected_to_sink:
            print(f"Warning: No cluster heads connected to sink after rotation.")
            
            if preserve_connected_ch:
                print(f"ERROR: The preserved CH {preserve_connected_ch.node_id} should have been connected!")
            
            if closest_ch_idx != -1:
                print(f"Closest CH is at distance {min_distance_to_sink:.2f}m (send range: {self.send_range:.2f}m)")
                
                # Option 1: If close enough, adjust send range
                if min_distance_to_sink <= self.send_range * 1.5:
                    old_range = self.send_range
                    self.send_range = min_distance_to_sink * 1.1  # Add 10% margin
                    print(f"Increasing send range from {old_range:.2f}m to {self.send_range:.2f}m to maintain connectivity")
                    self.initialize_hop_counts()  # Update with new range
                    
                # Option 2: If too far, revert to old configuration
                else:
                    print(f"Reverting cluster head rotation - would cause network partition")
                    self.cluster_heads = old_chs
                    self.nodes = old_nodes
                    self.assign_nodes_to_clusters()
                    self.initialize_hop_counts()
            else:
                # Last resort: completely revert to old configuration
                print(f"No suitable cluster heads found. Reverting to previous configuration.")
                self.cluster_heads = old_chs
                self.nodes = old_nodes
                self.assign_nodes_to_clusters()
                self.initialize_hop_counts()
        
        # Final check - print connectivity status
        connected_chs = sum(1 for ch in self.cluster_heads if ch.distance_to(self.sink) <= self.send_range)
        print(f"After rotation: {connected_chs}/{len(self.cluster_heads)} cluster heads connected to sink")
    
    def ensure_dql_agent(self, node):
        """Ensure that a node has a properly initialized DQL agent"""
        if not hasattr(node, 'dql_agent') or node.dql_agent is None:
            # Create agent with reasonable defaults
            node.state_size = 9  # Enhanced state representation
            node.action_size = 3  # Forward to CH, Forward to Sink, Sleep
            node.dql_agent = DQLAgent(node.state_size, node.action_size)
            if self.debug:
                print(f"Created new DQL agent for node {node.node_id}")
    
    def select_next_hop(self, node, destination):
        """Select next hop using DQL"""
        # If direct connection to destination, send directly
        if destination.node_id in node.neighbors and node.neighbors[destination.node_id][0] == 1:
            next_hop_id = destination.node_id
        else:
            # Ensure DQL agent exists
            self.ensure_dql_agent(node)
            
            # Get state and use DQL to decide action
            node.current_state = node.get_state(self)
            action = node.dql_agent.select_action(node.current_state)
            
            # Map action to decision
            if action == 0 and node.cluster_id is not None:  # Forward to CH
                next_hop_id = self.cluster_heads[node.cluster_id].node_id
            elif action == 1:  # Try to forward to sink
                next_hop_id = self.sink.node_id if self.sink.node_id in node.neighbors else None
            else:  # Sleep/no send
                next_hop_id = None
                
            # Store for learning
            node.previous_state = node.current_state
            node.previous_action = action
        
        return next_hop_id
    
    def simulate_data_transmission(self):
        """Simulate data transmission for current time step"""
        # Track transmissions for this time step
        transmissions_this_step = 0
        
        # Calculate time ratio for energy degradation (prevents stuck patterns)
        time_ratio = self.time / self.total_time if self.total_time > 0 else 0
        
        # Update node modes and check for data transmission
        for node in self.nodes:
            # Skip only if node is completely dead
            if not node.is_alive():
                continue
            
            # Set simulation time ratio for energy degradation
            node.simulation_time_ratio = time_ratio
            
            # All alive nodes consume some energy every time step
            if node.mode == NodeMode.SLEEP:
                # Sleeping nodes consume sleep energy
                node.reduce_energy("sleep")
                continue  # Skip transmission logic for sleeping nodes
            
            # Active nodes consume active energy
            node.reduce_energy("active")
            
            # Get sensor reading for this time
            sensor_data = self.dataset_loader.get_sensor_reading(
                node.node_id, self.time % TRANSMISSION_PERIOD)
            
            # Check if node should transmit based on data change
            send_permission = node.should_transmit(sensor_data, self.change_threshold)
            
            # Update mode based on send permission
            node.update_mode(send_permission, self.sleep_threshold, self.time)
            
            # Record data point
            node.data_history.append(sensor_data)
            
            if send_permission:
                # Create packet
                packet = Packet(node.node_id, sensor_data)
                packet.created_time = self.time
                
                # For regular nodes, send to cluster head
                if node.type == NodeType.REGULAR:
                    if node.cluster_id is not None:
                        destination = self.sink  # Ultimate destination
                        next_hop_id = self.select_next_hop(node, destination)
                        
                        if next_hop_id is not None:
                            next_hop = None
                            if next_hop_id == self.sink.node_id:
                                next_hop = self.sink
                            else:
                                next_hop = next((n for n in self.nodes if n.node_id == next_hop_id), None)
                            
                            if next_hop and next_hop.is_alive():
                                # Send packet - energy consumed regardless of success
                                packet.sender_id = node.node_id
                                packet.destination_id = next_hop.node_id
                                
                                # Node consumes energy for transmission attempt
                                if node.reduce_energy("send"):
                                    # Check if receiver can receive
                                    if next_hop.type == NodeType.SINK:
                                        # Sink always receives
                                        next_hop.receive_packet(packet)
                                        node.transmit_count += 1
                                        transmissions_this_step += 1
                                    else:
                                        # Regular node/CH receives if has energy
                                        if next_hop.reduce_energy("receive"):
                                            next_hop.send_queue.append(packet)
                                            next_hop.receive_count += 1
                                            node.transmit_count += 1
                                            transmissions_this_step += 1
                                        else:
                                            # Receiver dead - packet dropped, but sender energy already consumed
                                            pass
                                else:
                                    # Sender dead - packet dropped
                                    pass
                            else:
                                # No valid next hop - packet dropped but node still consumes energy
                                node.reduce_energy("send")
                        else:
                            # No route available - packet dropped but node still consumes energy
                            node.reduce_energy("send")
                    else:
                        # No cluster assigned - packet dropped but node still consumes energy
                        node.reduce_energy("send")
                
                # For cluster heads, aggregate and forward to sink
                elif node.type == NodeType.CLUSTER_HEAD:
                    # Process packets in queue
                    while node.send_queue:
                        queued_packet = node.send_queue.popleft()
                        
                        # Aggregate data
                        agg_data = node.aggregate_data(queued_packet.source_id, queued_packet.data)
                        
                        # Create new packet with aggregated data
                        new_packet = Packet(node.node_id, agg_data)
                        new_packet.created_time = self.time
                        new_packet.sender_id = node.node_id
                        new_packet.destination_id = self.sink.node_id
                        
                        # Always consume energy for transmission attempt
                        if node.reduce_energy("send"):
                            # Check if sink is within range
                            if node.distance_to(self.sink) <= self.send_range:
                                # Send directly to sink
                                self.sink.receive_packet(new_packet)
                                node.transmit_count += 1
                                transmissions_this_step += 1
                            else:
                                # Try to route through another CH
                                routed = False
                                for other_ch in self.cluster_heads:
                                    if (other_ch.node_id != node.node_id and 
                                        other_ch.is_alive() and
                                        node.distance_to(other_ch) <= self.send_range and
                                        other_ch.distance_to(self.sink) < node.distance_to(self.sink)):
                                        
                                        if other_ch.reduce_energy("receive"):
                                            other_ch.send_queue.append(new_packet)
                                            other_ch.receive_count += 1
                                            node.transmit_count += 1
                                            transmissions_this_step += 1
                                            routed = True
                                            break
                                
                                # If couldn't route through any CH, packet is dropped
                                # Energy was already consumed for the transmission attempt
                                if not routed:
                                    pass  # Packet dropped
                        else:
                            # Node died during transmission - packet dropped
                            break
        
        # Record transmissions
        self.transmissions.append(transmissions_this_step)
    
    def calculate_reward(self, node, action, success):
        """Calculate reward for RL-based routing decisions"""
        # Base reward
        if not success:
            return -1.0
            
        reward = 0.5
        
        # Energy efficiency reward
        energy_ratio = node.energy / node.initial_energy
        energy_reward = 0.5 * energy_ratio
        
        # Network lifetime reward
        alive_ratio = sum(1 for n in self.nodes if n.is_alive()) / len(self.nodes)
        lifetime_reward = 0.5 * alive_ratio
        
        # Distance-based reward
        if node.cluster_id is not None and action == 0:  # Forward to CH
            ch = self.cluster_heads[node.cluster_id]
            dist = node.distance_to(ch)
            max_dist = self.send_range
            dist_reward = 0.5 * (1 - min(1.0, dist / max_dist))
        elif action == 1:  # Forward to sink
            dist = node.distance_to(self.sink)
            max_dist = self.max_longitude + self.max_latitude
            dist_reward = 1.0 * (1 - min(1.0, dist / max_dist))
        else:  # Sleep/no send
            dist_reward = 0.1
        
        total_reward = reward + energy_reward + lifetime_reward + dist_reward
        return total_reward
    
    def update_statistics(self):
        """Update simulation statistics with enhanced individual node tracking"""
        # Count live nodes
        live_count = sum(1 for node in self.nodes if node.is_alive())
        self.live_node_count.append(live_count)
        
        # Calculate percentage
        live_percentage = (live_count / len(self.nodes)) * 100
        self.live_node_percentage.append(live_percentage)
        
        # Track individual node status and energy
        node_status_data = {
            'time': self.time,
            'live_nodes': live_count,
            'live_percentage': live_percentage,
            'total_transmissions': sum(self.transmissions) if self.transmissions else 0
        }
        
        # Add individual node information
        for node in self.nodes:
            node_id = node.node_id
            is_alive = node.is_alive()
            
            # Track energy over time
            self.individual_node_energy[node_id].append(node.energy)
            
            # Add node-specific data to this time step
            node_status_data[f'node_{node_id}_energy'] = node.energy
            node_status_data[f'node_{node_id}_alive'] = 1 if is_alive else 0
            node_status_data[f'node_{node_id}_type'] = node.type.name if hasattr(node.type, 'name') else str(node.type)
            
            # Check for node death and record death time
            if not is_alive and node_id not in self.node_death_times:
                self.node_death_times[node_id] = self.time
                print(f"Node {node_id} died at time {self.time}")
        
        # Store this time step's data
        self.node_lifetime_data.append(node_status_data)
        
        # Check for stuck percentage pattern (added to prevent stuck issues)
        if len(self.live_node_percentage) > 100:  # Check last 100 time steps
            recent_percentages = self.live_node_percentage[-100:]
            if all(abs(p - recent_percentages[0]) < 0.1 for p in recent_percentages):
                if self.debug:
                    print(f"WARNING: Stuck percentage detected at {live_percentage:.1f}% for 100+ steps")
        
        # Track energy levels (keep original for compatibility)
        energy_snapshot = [node.energy for node in self.nodes]
        self.energy_levels.append(energy_snapshot)
        
        # Check for first node death
        if live_count < len(self.nodes) and self.first_death_time == -1:
            self.first_death_time = self.time
    
    def emergency_connectivity_restoration(self):
        """Emergency connectivity restoration method.
        Attempts to create a new cluster head from the closest node to the sink.
        Returns:
            bool: True if connectivity was restored, False otherwise
        """
        print(f"[Time {self.time}] Attempting emergency connectivity restoration...")
        
        # Find all alive non-cluster-head nodes
        alive_nodes = [node for node in self.nodes if node.is_alive()]
        alive_chs = [ch for ch in self.cluster_heads if ch.is_alive()]
        
        if not alive_nodes:
            print("No alive nodes available for emergency restoration.")
            return False
            
        # Find the closest alive node to the sink
        closest_node = min(alive_nodes, key=lambda n: n.distance_to(self.sink))
        distance_to_sink = closest_node.distance_to(self.sink)
        
        # Check if this node can potentially reach the sink
        if distance_to_sink <= self.send_range * 2.5:  # Allow up to 2.5x range for emergency
            # If the closest node is already a cluster head, increase range
            if closest_node in self.cluster_heads:
                old_range = self.send_range
                self.send_range = distance_to_sink * 1.1  # Add 10% margin
                print(f"EMERGENCY RESTORATION: Increased send range from {old_range:.2f}m to {self.send_range:.2f}m")
                print(f"  -> Existing CH {closest_node.node_id} now reachable at {distance_to_sink:.2f}m")
            else:
                # Make the closest node a cluster head
                if closest_node not in self.cluster_heads:
                    # Create a new ClusterHead object from the regular node
                    new_ch = ClusterHead(closest_node.node_id, closest_node.x, closest_node.y, closest_node.energy)
                    # Copy important attributes from the original node
                    new_ch.energy = closest_node.energy
                    new_ch.initial_energy = closest_node.initial_energy
                    new_ch.mode = closest_node.mode
                    new_ch.neighbors = closest_node.neighbors.copy()
                    new_ch.data_history = closest_node.data_history.copy() if closest_node.data_history else []
                    
                    # Replace the node in the nodes list
                    node_index = self.nodes.index(closest_node)
                    self.nodes[node_index] = new_ch
                    
                    # Add to cluster heads list
                    self.cluster_heads.append(new_ch)
                    print(f"EMERGENCY RESTORATION: Created new cluster head {new_ch.node_id}")
                    print(f"  -> Distance to sink: {distance_to_sink:.2f}m")
            
            # Re-initialize hop counts with new configuration
            self.initialize_hop_counts()
            
            # Verify connectivity was restored
            if closest_node.distance_to(self.sink) <= self.send_range:
                print(f"EMERGENCY RESTORATION SUCCESSFUL: CH {closest_node.node_id} connected to sink")
                return True
            else:
                print(f"EMERGENCY RESTORATION FAILED: CH {closest_node.node_id} still too far from sink")
                return False
        else:
            print(f"EMERGENCY RESTORATION FAILED: Closest node {closest_node.node_id} is {distance_to_sink:.2f}m from sink (max emergency range: {self.send_range * 2.5:.2f}m)")
            return False

    def should_continue_simulation(self):
        """Check if the simulation should continue based on network state
        Returns:
            bool: True if the simulation should continue, False otherwise
        """
        # Check if we've reached the maximum simulation time
        if self.time >= self.total_time:
            return False
        
        # Check if all nodes are dead
        if all(not node.is_alive() for node in self.nodes):
            print(f"All nodes are dead at time {self.time}. Stopping simulation.")
            return False
        
        # Skip network partition check on the first few time steps to allow for setup
        if self.time < 5:  # Give the network a few time steps to establish connections
            return True
            
        # Check if network is partitioned (no viable path to sink)
        live_chs = [ch for ch in self.cluster_heads if ch.is_alive()]
        
        if not live_chs:
            if self.cluster_heads:  # If we had cluster heads but none are alive
                print(f"All cluster heads are dead at time {self.time}. Network is non-functional.")
                return False
            return True  # If we never had cluster heads, let simulation continue
            
        # Check for direct connections to sink
        directly_connected_chs = [ch for ch in live_chs if ch.distance_to(self.sink) <= self.send_range]
        
        # Check for indirect paths (CH to CH to Sink)
        indirectly_connected_chs = []
        for ch in live_chs:
            if ch not in directly_connected_chs:
                if any(ch.distance_to(other_ch) <= self.send_range for other_ch in directly_connected_chs):
                    indirectly_connected_chs.append(ch)
        
        # If no direct connections, try to restore connectivity
        if not directly_connected_chs:
            # Find the closest cluster head to sink
            if live_chs:
                closest_ch = min(live_chs, key=lambda ch: ch.distance_to(self.sink))
                min_distance = closest_ch.distance_to(self.sink)
                
                # Try to restore connectivity by temporarily increasing send range
                if min_distance <= self.send_range * 2.0:  # Allow up to 2x range increase
                    old_range = self.send_range
                    self.send_range = min_distance * 1.1  # Add 10% margin
                    print(f"CONNECTIVITY RESTORED: Increased send range from {old_range:.2f}m to {self.send_range:.2f}m")
                    print(f"  -> Closest CH {closest_ch.node_id} now reachable at {min_distance:.2f}m")
                    
                    # Re-initialize hop counts with new range
                    self.initialize_hop_counts()
                    return True
            
            # If range increase didn't work, try emergency restoration
            if self.emergency_connectivity_restoration():
                return True
                
            # Last resort - network truly partitioned
            print(f"NETWORK PARTITION DETECTED at time {self.time}:")
            print(f"- Live cluster heads: {len(live_chs)}/{len(self.cluster_heads)}")
            print(f"- No cluster heads can reach sink (range: {self.send_range:.2f}m)")
            if live_chs:
                min_distance = min(ch.distance_to(self.sink) for ch in live_chs)
                print(f"- Closest CH distance: {min_distance:.2f}m (too far for recovery)")
            
            # Show distances to help diagnose
            for ch in live_chs:
                dist = ch.distance_to(self.sink)
                print(f"  CH {ch.node_id} distance to sink: {dist:.2f}m (energy: {ch.energy:.2f})")
            
            # Allow simulation to continue for a while in case connectivity is restored
            if hasattr(self, 'partition_start_time'):
                partition_duration = self.time - self.partition_start_time
                if partition_duration > 1000:  # Stop after 1000 seconds of partition
                    print(f"Network has been partitioned for {partition_duration} seconds. Stopping simulation.")
                    return False
                elif partition_duration % 200 == 0:  # Try emergency restoration every 200 seconds
                    print(f"Partition duration: {partition_duration}s. Retrying emergency restoration...")
                    if self.emergency_connectivity_restoration():
                        return True
            else:
                self.partition_start_time = self.time
                print(f"Network partitioned. Continuing simulation for potential recovery...")
            
            return True
        else:
            # Connectivity restored - reset partition timer
            if hasattr(self, 'partition_start_time'):
                delattr(self, 'partition_start_time')
                print(f"Network connectivity restored at time {self.time}")
        
        # Continue if we have paths
        if self.debug and self.time % 100 == 0:
            print(f"[Time {self.time}] Network connectivity: {len(directly_connected_chs)} CHs directly connected, "
                  f"{len(indirectly_connected_chs)} CHs indirectly connected")
            
        return True
    
    def run_simulation(self):
        """Run the complete simulation"""
        print(f"Starting RLBEEP simulation with {self.num_nodes} nodes, {self.num_clusters} clusters")
        print(f"Simulation will run for {self.total_time} seconds or until network failure")
        
        t = 0
        while t < self.total_time:
            self.time = t
            
            # Check if simulation should continue
            if not self.should_continue_simulation():
                print(f"Simulation ending early at time {t}")
                break
                
            # Check for cluster head rotation
            self.rotate_cluster_heads()
            
            # Simulate data transmission
            self.simulate_data_transmission()
            
            # Update statistics
            self.update_statistics()
            
            # Print progress with stuck percentage detection
            if t % 100 == 0:
                live_pct = self.live_node_percentage[-1]
                
                # Check for stuck pattern in recent history (more intelligent detection)
                stuck_warning = ""
                if len(self.live_node_percentage) >= 200:  # Check last 200 time steps (longer window)
                    recent_percentages = self.live_node_percentage[-200:]
                    # Only warn if percentage is stuck AND not at 100% or 0% (which are normal states)
                    if (all(abs(p - recent_percentages[0]) < 0.1 for p in recent_percentages) and 
                        recent_percentages[0] != 100.0 and recent_percentages[0] != 0.0):
                        stuck_warning = " [POTENTIALLY STUCK]"
                
                print(f"Time {t}/{self.total_time}: {live_pct:.1f}% nodes alive{stuck_warning}")
                
                # Additional debug info every 1000 steps
                if t % 1000 == 0 and self.debug:
                    alive_chs = sum(1 for ch in self.cluster_heads if ch.is_alive())
                    alive_regs = sum(1 for node in self.nodes if node.is_alive() and node.type != NodeType.CLUSTER_HEAD)
                    print(f"  -> CH alive: {alive_chs}/{len(self.cluster_heads)}, Regular alive: {alive_regs}")
                    
                    # Show energy distribution
                    alive_energies = [node.energy for node in self.nodes if node.is_alive()]
                    if alive_energies:
                        min_energy = min(alive_energies)
                        max_energy = max(alive_energies)
                        avg_energy = sum(alive_energies) / len(alive_energies)
                        print(f"  -> Energy - Min: {min_energy:.2f}, Max: {max_energy:.2f}, Avg: {avg_energy:.2f}")
            
            # Increment time counter
            t += 1
        
        # Calculate final statistics
        final_live_percentage = self.live_node_percentage[-1] if self.live_node_percentage else 0
        
        print(f"\nSimulation complete.")
        print(f"First node death time: {self.first_death_time if self.first_death_time != -1 else 'No deaths'}")
        print(f"Final live node percentage: {final_live_percentage:.1f}%")
        
        results = {
            'first_death_time': self.first_death_time,
            'final_live_percentage': final_live_percentage,
            'live_node_count': self.live_node_count,
            'live_node_percentage': self.live_node_percentage,
            'energy_levels': self.energy_levels,
            'transmissions': self.transmissions,
            'total_time': self.total_time
        }
        
        return results
    
    def save_epoch_results(self, epoch_num, csv_filename="epoch_results.csv"):
        """Save epoch results to CSV file for multi-epoch analysis"""
        import csv
        
        # Create CSV file path
        csv_path = os.path.join(self.results_dir, csv_filename)
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(csv_path)
        
        # Calculate final statistics
        final_live_percentage = self.live_node_percentage[-1] if self.live_node_percentage else 0
        total_transmissions = sum(self.transmissions) if self.transmissions else 0
        
        # Calculate energy statistics
        if self.energy_levels and self.energy_levels[-1]:
            avg_energy = sum(self.energy_levels[-1]) / len(self.energy_levels[-1])
            energy_efficiency = 1 - (avg_energy / NODE_INITIAL_ENERGY)
        else:
            avg_energy = NODE_INITIAL_ENERGY
            energy_efficiency = 0.0
        
        # Write to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = [
                'epoch', 'first_death_time', 'final_live_percentage', 
                'simulation_duration', 'total_transmissions', 'avg_remaining_energy',
                'energy_efficiency', 'num_nodes', 'num_clusters'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write epoch data
            writer.writerow({
                'epoch': epoch_num,
                'first_death_time': self.first_death_time if self.first_death_time != -1 else self.total_time,
                'final_live_percentage': final_live_percentage,
                'simulation_duration': self.time,
                'total_transmissions': total_transmissions,
                'avg_remaining_energy': avg_energy,
                'energy_efficiency': energy_efficiency,
                'num_nodes': self.num_nodes,
                'num_clusters': self.num_clusters
            })
        
        print(f"Epoch {epoch_num} results saved to {csv_path}")
        return csv_path
    
    def save_results(self, filename=None):
        """Save simulation results to a file"""
        if filename is None:
            filename = "rlbeep_results.txt"
        
        results_path = os.path.join(self.results_dir, filename)
        
        with open(results_path, 'w') as f:
            f.write("RLBEEP Simulation Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Number of Nodes: {self.num_nodes}\n")
            f.write(f"Number of Clusters: {self.num_clusters}\n")
            f.write(f"Simulation Duration: {self.time}/{self.total_time} seconds\n\n")
            
            f.write(f"First Node Death Time: {self.first_death_time if self.first_death_time != -1 else 'No deaths'}\n")
            
            # Check if there are any live node statistics before trying to access them
            if self.live_node_percentage:
                f.write(f"Final Live Node Percentage: {self.live_node_percentage[-1]:.1f}%\n\n")
            else:
                f.write(f"Final Live Node Percentage: 0.0%\n\n")
            
            f.write("Network Statistics:\n")
            f.write("-" * 30 + "\n")
            
            # Check if there are any transmission statistics
            if self.transmissions:
                total_tx = sum(self.transmissions)
                f.write(f"Total Data Transmissions: {total_tx}\n")
                
                avg_tx = total_tx / len(self.transmissions)
                f.write(f"Average Transmissions per Second: {avg_tx:.2f}\n")
            else:
                f.write(f"Total Data Transmissions: 0\n")
                f.write(f"Average Transmissions per Second: 0.00\n")
            
            # Calculate average remaining energy
            if self.energy_levels and self.energy_levels[-1]:
                avg_energy = sum(self.energy_levels[-1]) / len(self.energy_levels[-1])
                f.write(f"Average Remaining Energy: {avg_energy:.2f} J\n")
                
                energy_efficiency = 1 - (avg_energy / NODE_INITIAL_ENERGY)
                f.write(f"Energy Efficiency: {energy_efficiency:.2f}\n")
            else:
                f.write(f"Average Remaining Energy: {NODE_INITIAL_ENERGY:.2f} J\n")
                f.write(f"Energy Efficiency: 0.00\n")
            
            print(f"Results saved to {results_path}")
        
        # Also save as CSV for easy analysis
        self.save_csv_results()
        
        return results_path

    def save_csv_results(self):
        """Save simulation results as CSV files for analysis"""
        
        # Create the main network lifetime CSV with detailed node information
        if self.node_lifetime_data:
            lifetime_df = pd.DataFrame(self.node_lifetime_data)
            lifetime_csv_path = os.path.join(self.results_dir, "network_lifetime_data.csv")
            lifetime_df.to_csv(lifetime_csv_path, index=False)
            print(f"Network lifetime data saved to {lifetime_csv_path}")
        
        # Create individual node death times CSV
        death_times_data = {
            'Node_ID': [],
            'Death_Time': [],
            'Node_Type': [],
            'Initial_Energy': [],
            'Final_Energy': []
        }
        
        for node in self.nodes:
            death_times_data['Node_ID'].append(node.node_id)
            if node.node_id in self.node_death_times:
                death_times_data['Death_Time'].append(self.node_death_times[node.node_id])
            else:
                death_times_data['Death_Time'].append('Survived' if node.is_alive() else self.time)
            death_times_data['Node_Type'].append(node.type.name if hasattr(node.type, 'name') else str(node.type))
            death_times_data['Initial_Energy'].append(node.initial_energy)
            death_times_data['Final_Energy'].append(node.energy)
        
        death_times_df = pd.DataFrame(death_times_data)
        death_times_csv_path = os.path.join(self.results_dir, "node_death_times.csv")
        death_times_df.to_csv(death_times_csv_path, index=False)
        print(f"Node death times saved to {death_times_csv_path}")
        
        # Create simplified network lifetime CSV (for easy plotting)
        simple_lifetime_data = {
            'Time': list(range(len(self.live_node_percentage))),
            'Live_Nodes': self.live_node_count if self.live_node_count else [],
            'Live_Percentage': self.live_node_percentage if self.live_node_percentage else []
        }
        
        # Ensure all lists are the same length
        max_len = max(len(v) for v in simple_lifetime_data.values() if v)
        for key in simple_lifetime_data:
            if len(simple_lifetime_data[key]) < max_len:
                simple_lifetime_data[key].extend([0] * (max_len - len(simple_lifetime_data[key])))
        
        simple_df = pd.DataFrame(simple_lifetime_data)
        simple_csv_path = os.path.join(self.results_dir, "node_lifetime_data.csv")
        simple_df.to_csv(simple_csv_path, index=False)
        print(f"Simple network lifetime data saved to {simple_csv_path}")
        
        # Create simulation summary CSV
        summary_data = {
            'Metric': [
                'Number_of_Nodes',
                'Number_of_Clusters', 
                'First_Death_Time',
                'Last_Death_Time',
                'Network_Lifetime',
                'Final_Live_Percentage',
                'Total_Transmissions',
                'Average_Transmissions_per_Second',
                'Simulation_Duration'
            ],
            'Value': [
                self.num_nodes,
                self.num_clusters,
                self.first_death_time if self.first_death_time != -1 else 'No_deaths',
                max(self.node_death_times.values()) if self.node_death_times else 'No_deaths',
                max(self.node_death_times.values()) if self.node_death_times else self.time,
                self.live_node_percentage[-1] if self.live_node_percentage else 0,
                sum(self.transmissions) if self.transmissions else 0,
                sum(self.transmissions) / len(self.transmissions) if self.transmissions else 0,
                self.time
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(self.results_dir, "simulation_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Simulation summary saved to {summary_csv_path}")
        
        # Print key statistics
        print(f"\n=== SIMULATION STATISTICS ===")
        print(f"First node death time: {self.first_death_time if self.first_death_time != -1 else 'No deaths'}")
        if self.node_death_times:
            print(f"Last node death time: {max(self.node_death_times.values())}")
            print(f"Total nodes died: {len(self.node_death_times)}")
            print(f"Nodes survived: {self.num_nodes - len(self.node_death_times)}")
        print(f"Final live percentage: {self.live_node_percentage[-1] if self.live_node_percentage else 0:.1f}%")
        print(f"=============================\n")

    def plot_results(self):
        """Plot simulation results"""
        visualizer = SimulationVisualizer(self)
        visualizer.plot_all_results()

#------------------------------------------------------------------------------
# VISUALIZATION COMPONENTS
#------------------------------------------------------------------------------

class SimulationVisualizer:
    """Class for visualizing simulation results"""
    
    def __init__(self, simulation):
        """Initialize with reference to the simulation instance"""
        self.simulation = simulation
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        self._ensure_results_dir()
    
    def _ensure_results_dir(self):
        """Ensure the results directory exists"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def plot_network_topology(self, filename="network_topology.png"):
        """Plot the network topology showing nodes, cluster heads, and sink"""
        plt.figure(figsize=(10, 8))
        
        # Plot regular nodes
        for node in self.simulation.nodes:
            if node.type == NodeType.REGULAR:
                plt.scatter(node.x, node.y, color='blue', s=50)
                
                # Draw lines to cluster head if assigned
                if node.cluster_id is not None:
                    ch = self.simulation.cluster_heads[node.cluster_id]
                    plt.plot([node.x, ch.x], [node.y, ch.y], 'b-', alpha=0.3)
        
        # Plot cluster heads
        for ch in self.simulation.cluster_heads:
            plt.scatter(ch.x, ch.y, color='red', s=100, marker='s')
            
            # Draw lines to sink if within range
            if ch.distance_to(self.simulation.sink) <= self.simulation.send_range:
                plt.plot([ch.x, self.simulation.sink.x], 
                         [ch.y, self.simulation.sink.y], 'g-', alpha=0.5)
        
        # Plot sink node
        plt.scatter(self.simulation.sink.x, self.simulation.sink.y, 
                   color='green', s=200, marker='*', label='Sink')
        
        # Add node IDs as labels
        for node in self.simulation.nodes + [self.simulation.sink]:
            plt.annotate(f"{node.node_id}", (node.x, node.y), 
                        textcoords="offset points", xytext=(0,5), ha='center')
        
        plt.title("Network Topology")
        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        plt.grid(True)
        plt.legend(['Regular Node', 'Cluster Head', 'Sink'])
        
        plt.savefig(os.path.join(self.results_dir, filename))
        plt.close()
    
    def plot_node_lifetime(self, filename="node_lifetime.png"):
        """Plot node lifetime (percentage of alive nodes)"""
        plt.figure(figsize=(10, 6))
        
        # Create time array
        times = list(range(len(self.simulation.live_node_percentage)))
        
        # Plot percentage of alive nodes
        plt.plot(times, self.simulation.live_node_percentage, 'g-', linewidth=2)
        
        # Add first death time marker if applicable
        if self.simulation.first_death_time != -1:
            plt.axvline(x=self.simulation.first_death_time, color='r', linestyle='--', 
                       label=f'First Death: {self.simulation.first_death_time}s')
        
        plt.title("Node Lifetime Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Alive Nodes (%)")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)  # Add some margin above 100%
        
        if self.simulation.first_death_time != -1:
            plt.legend()
        
        plt.savefig(os.path.join(self.results_dir, filename))
        plt.close()
    
    def plot_all_results(self):
        """Plot all result visualizations"""
        print("Generating visualizations...")
        
        # Ensure the results directory exists
        if not os.path.exists(self.results_dir):
            try:
                os.makedirs(self.results_dir)
                print(f"Created results directory at {self.results_dir}")
            except Exception as e:
                print(f"Error creating results directory: {str(e)}")
                return
        
        try:
            self.plot_network_topology()
            self.plot_node_lifetime()
            print("Visualizations saved to results directory")
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")

#------------------------------------------------------------------------------
# MAIN ENTRY POINT
#------------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='RLBEEP Simulation')
    
    parser.add_argument('--nodes', type=int, default=NUM_NODES,
                        help=f'Number of nodes in the network (default: {NUM_NODES})')
    
    parser.add_argument('--clusters', type=int, default=NUM_CLUSTERS,
                        help=f'Number of clusters (default: {NUM_CLUSTERS})')
    
    parser.add_argument('--time', type=int, default=TOTAL_SIMULATION_TIME,
                        help=f'Total simulation time in seconds (default: {TOTAL_SIMULATION_TIME})')
    
    parser.add_argument('--dataset', type=str, default=None,
                        help=f'Path to the dataset directory')
    
    parser.add_argument('--energy', type=float, default=NODE_INITIAL_ENERGY,
                        help=f'Initial energy of each node in Joules (default: {NODE_INITIAL_ENERGY})')
    
    parser.add_argument('--alpha', type=float, default=ALPHA,
                        help=f'Learning rate for Q-value updates (default: {ALPHA})')
    
    parser.add_argument('--threshold', type=float, default=CHANGE_THRESHOLD,
                        help=f'Threshold for data transmission (default: {CHANGE_THRESHOLD})')
    
    parser.add_argument('--send-range', type=float, default=SEND_RANGE,
                        help=f'Send range in meters (default: {SEND_RANGE})')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of simulation results')
                        
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename for simulation results')
                        
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    
    return parser.parse_args()

def main():
    """Main function to run the RLBEEP simulation"""
    args = parse_args()
    
    # Print a nice banner with simulation information
    print("\n" + "=" * 70)
    print("  RLBEEP: Reinforcement Learning-Based Energy Efficient Protocol for WSNs  ")
    print("=" * 70)
    print(f"Number of Nodes: {args.nodes}")
    print(f"Number of Clusters: {args.clusters}")
    print(f"Simulation Time: {args.time} seconds")
    print(f"Initial Node Energy: {args.energy} J")
    print(f"Visualization: Enabled (results will be saved in the results folder)")
    
    # Get dataset path
    if args.dataset:
        dataset_path = args.dataset
    else:
        # Use current directory + Dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, "Dataset")
    
    print(f"Dataset Path: {dataset_path}")
    print("=" * 70)
    
    # Check how many nodes are actually available in the dataset
    temp_loader = DatasetLoader(dataset_path)
    available_nodes = temp_loader.dataset_info.get('available_nodes', 0)
    if available_nodes == 0:
        print("ERROR: No valid node data files found in the dataset directory.")
        sys.exit(1)
    
    # Adjust number of nodes if necessary - allow more nodes than dataset files since we have enhanced data generation
    actual_nodes = args.nodes  # Use requested nodes regardless of dataset availability
    if available_nodes < args.nodes:
        print(f"INFO: Using enhanced data generation for {args.nodes} nodes with {available_nodes} dataset files.")
    
    # Adjust clusters if necessary (ensure we don't have more clusters than nodes)
    actual_clusters = min(args.clusters, actual_nodes)
    if actual_clusters < args.clusters:
        print(f"WARNING: Reducing clusters to {actual_clusters} due to node count.")
    
    # Create and run simulation
    simulation = RLBEEPSimulation(
        dataset_path=dataset_path,
        num_nodes=actual_nodes,
        num_clusters=actual_clusters,
        total_time=args.time,
        send_range=args.send_range,
        alpha=args.alpha,
        change_threshold=args.threshold,
        debug=args.debug
    )
    
    # Enable debug mode if requested
    if args.debug:
        print("Debug mode enabled")
        # Print network layout
        print(f"Sink position: ({simulation.sink.x:.2f}, {simulation.sink.y:.2f})")
        for i, ch in enumerate(simulation.cluster_heads):
            print(f"Cluster Head {ch.node_id} position: ({ch.x:.2f}, {ch.y:.2f})")
            print(f"  Distance to sink: {ch.distance_to(simulation.sink):.2f}m")
            print(f"  Connected to sink: {simulation.sink.node_id in ch.neighbors}")
            print(f"  Members: {len(ch.cluster_members)}")
    
    results = simulation.run_simulation()
    
    # Save results to file
    simulation.save_results(args.output)
    
    # Generate visualizations by default (no need for --visualize flag)
    simulation.plot_results()
    print("All results have been saved in the results directory")

if __name__ == "__main__":
    main()
