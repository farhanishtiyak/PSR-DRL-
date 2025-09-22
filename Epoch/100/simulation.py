#!/usr/bin/env python3
"""
RLBEEP: Reinforcement-Learning-Based Energy Efficient Protocol for Wireless Sensor Networks
This simulation implements the RLBEEP protocol using the provided dataset and the algorithmic steps
described in the documentation.
"""

import os
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from enum import Enum
import time


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
        self.source_id = source_id  # Original creator of the packet
        self.sender_id = sender_id if sender_id is not None else source_id  # Last sender
        self.destination_id = destination_id  # Intended recipient (None for broadcast)
        self.data = data  # Sensor data or aggregated data
        self.hops = 0  # Number of hops taken so far


class Node:
    """Base class representing a node in the WSN"""
    def __init__(self, node_id, x, y, initial_energy=100):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.energy = initial_energy
        self.mode = NodeMode.ACTIVE
        self.type = NodeType.REGULAR
        self.cluster_head_id = None
        self.neighbors = []
        self.q_values = {}  # For RL-based routing
        self.no_send_counter = 0
        self.last_sensor_data = None
        self.min_value = float('inf')
        self.max_value = float('-inf')
        self.hop_count_to_sink = float('inf')
        
        # For data transmission restriction
        self.data_history = []
        
    def distance_to(self, other_node):
        """Calculate Euclidean distance to another node"""
        return math.sqrt((self.x - other_node.x) ** 2 + (self.y - other_node.y) ** 2)
    
    def is_alive(self):
        """Check if the node still has energy"""
        return self.energy > 0
    
    def reduce_energy(self, activity_type):
        """Reduce node energy based on activity type with random variation"""
        base_energy_consumption = {
            'send': 0.5,      # Increased for 100-node network
            'receive': 0.3,   # Increased for 100-node network
            'active': 0.15,   # Increased for 100-node network
            'sleep': 0.08     # Increased for 100-node network
        }
        
        if activity_type in base_energy_consumption:
            # Add random variation (±15%) to energy consumption
            # This creates more realistic and varied energy depletion patterns
            variation = random.uniform(0.85, 1.15)
            actual_consumption = base_energy_consumption[activity_type] * variation
            
            self.energy -= actual_consumption
            self.energy = max(0, self.energy)  # Ensure energy doesn't go negative
    
    def update_mode(self, send_permission, sleep_threshold):
        """Update node mode based on sleep scheduling algorithm"""
        if self.type == NodeType.CLUSTER_HEAD:
            # Cluster heads stay active
            self.mode = NodeMode.ACTIVE
            return
        
        if self.mode == NodeMode.ACTIVE:
            if not send_permission:
                self.no_send_counter += 1
                if self.no_send_counter >= sleep_threshold:
                    self.mode = NodeMode.SLEEP
                    self.no_send_counter = 0
            else:
                self.no_send_counter = 0
        # Node in SLEEP mode will be woken up by the simulation at appropriate intervals
    
    def should_transmit(self, sensor_data, change_threshold):
        """Determine if data should be transmitted based on change threshold"""
        if self.min_value > sensor_data:
            self.min_value = sensor_data
        
        if self.max_value < sensor_data:
            self.max_value = sensor_data
        
        if (self.max_value - sensor_data) > change_threshold or (sensor_data - self.min_value) > change_threshold:
            return True
        
        return False


class ClusterHead(Node):
    """Class representing a cluster head node"""
    def __init__(self, node_id, x, y, initial_energy=100):
        super().__init__(node_id, x, y, initial_energy)
        self.type = NodeType.CLUSTER_HEAD
        self.cluster_members = []
        self.neighbor_clusters = {}
        self.aggregated_data = {}
    
    def aggregate_data(self, source_id, data):
        """Aggregate data from cluster members"""
        self.aggregated_data[source_id] = data
        # Simple averaging for aggregation
        return sum(self.aggregated_data.values()) / len(self.aggregated_data)
    
    def update_neighbor_table(self, neighbor_id, hop_count, energy):
        """Update neighbor cluster information"""
        self.neighbor_clusters[neighbor_id] = {
            'hop_count': hop_count,
            'energy': energy
        }


class SinkNode(Node):
    """Class representing the sink node"""
    def __init__(self, node_id, x, y):
        super().__init__(node_id, x, y, float('inf'))  # Sink has unlimited energy
        self.type = NodeType.SINK
        self.hop_count_to_sink = 0
        self.received_packets = 0
        self.received_data = {}
    
    def receive_packet(self, packet):
        """Process received packet at sink"""
        self.received_packets += 1
        self.received_data[packet.source_id] = packet.data
        return True


class RLBEEPSimulation:
    """Main simulation class for RLBEEP"""
    def __init__(self, dataset_path, num_nodes=100, num_clusters=40, 
                 max_longitude=200.0, max_latitude=200.0, 
                 send_range=30, alpha=0.5,  # Increased parameters for 100 nodes
                 dfr_min=5.0, dfr_max=55.0, 
                 total_time=3000,
                 sleep_threshold=5,
                 change_threshold=1.0,
                 ch_rotation_interval=300):  # Default rotation interval: 300 seconds
        
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
        
        self.nodes = []
        self.sink = None
        self.cluster_heads = []
        self.time = 0
        self.first_death_time = -1
        self.live_node_percentage = []
        self.cluster_to_members = {}  # Maps cluster head IDs to their member node IDs
        
        # Additional metrics for CSV output
        self.time_data = []
        self.energy_data = []  # Track total network energy over time
        self.packets_received = []  # Track packets received at sink
        self.active_nodes_data = []  # Track number of active nodes
        
        # Load dataset
        self.dataset = self.load_dataset()
        
        # Initialize simulation
        self.setup_network()
        
    def load_dataset(self):
        """Load the dataset for all nodes (only for nodes 1-10, others will reuse this data)"""
        dataset = {}
        for i in range(1, 11):  # Only load data for nodes 1-10
            file_path = os.path.join(self.dataset_path, f"node{i}.csv")
            if os.path.exists(file_path):
                node_data = pd.read_csv(file_path, header=None)
                dataset[i] = node_data
        return dataset
    
    def setup_network(self):
        """Initialize the network with nodes, cluster heads, and sink"""
        # Create sink at the center of the network
        self.sink = SinkNode(0, self.max_longitude / 2, self.max_latitude / 2)
        
        # Create regular nodes with random positions and slightly random initial energy
        for i in range(1, self.num_nodes + 1):
            x = random.uniform(0, self.max_longitude)
            y = random.uniform(0, self.max_latitude)
            # Add randomness to initial energy (increased for 100-node network)
            # With 100 nodes, increase base energy to handle much more traffic
            initial_energy = 300 + random.uniform(-20, 20)  # Range: 280-320
            self.nodes.append(Node(i, x, y, initial_energy))
        
        # Select cluster heads (distribute them more evenly across the network)
        # First, sort nodes by their distance from center to get a good distribution
        center_x, center_y = self.max_longitude / 2, self.max_latitude / 2
        nodes_with_distance = [(node, math.sqrt((node.x - center_x)**2 + (node.y - center_y)**2)) 
                              for node in self.nodes]
        nodes_with_distance.sort(key=lambda x: x[1])  # Sort by distance from center
        
        # Select cluster heads with some spacing to avoid clustering
        selected_indices = []
        nodes_per_cluster = len(self.nodes) // self.num_clusters
        
        for i in range(self.num_clusters):
            # Try to select nodes that are reasonably spaced
            base_index = i * nodes_per_cluster
            if base_index < len(self.nodes):
                selected_indices.append(base_index)
        
        # Ensure we have exactly num_clusters cluster heads
        while len(selected_indices) < self.num_clusters and len(selected_indices) < len(self.nodes):
            for i in range(len(self.nodes)):
                if i not in selected_indices:
                    selected_indices.append(i)
                    break
        
        for i in selected_indices[:self.num_clusters]:
            if i < len(self.nodes):
                node = self.nodes[i]
                # Replace regular node with cluster head
                cluster_head = ClusterHead(node.node_id, node.x, node.y, node.energy)
                self.nodes[i] = cluster_head
                self.cluster_heads.append(cluster_head)
                
                # Initialize cluster member tracking
                self.cluster_to_members[cluster_head.node_id] = []
        
        # Assign cluster heads to nodes (nearest cluster head)
        for node in self.nodes:
            if node.type != NodeType.CLUSTER_HEAD:
                nearest_ch = min(self.cluster_heads, key=lambda ch: node.distance_to(ch))
                node.cluster_head_id = nearest_ch.node_id
                nearest_ch.cluster_members.append(node.node_id)
                
                # Update cluster member tracking
                if nearest_ch.node_id in self.cluster_to_members:
                    self.cluster_to_members[nearest_ch.node_id].append(node.node_id)
        
        # Identify neighbors for all nodes (within send range)
        all_nodes = self.nodes + [self.sink]
        for node in all_nodes:
            for potential_neighbor in all_nodes:
                if node.node_id != potential_neighbor.node_id and node.distance_to(potential_neighbor) <= self.send_range:
                    node.neighbors.append(potential_neighbor.node_id)
        
        # Initialize Q-values for routing
        for node in all_nodes:
            for neighbor_id in node.neighbors:
                neighbor = self.get_node_by_id(neighbor_id)
                node.q_values[neighbor_id] = 0.0
        
        # Initialize hop counts (using simple shortest path for initialization)
        self.initialize_hop_counts()
    
    def initialize_hop_counts(self):
        """Initialize hop counts to sink using breadth-first search"""
        queue = [(self.sink.node_id, 0)]
        visited = set([self.sink.node_id])
        
        while queue:
            node_id, hops = queue.pop(0)
            node = self.get_node_by_id(node_id)
            node.hop_count_to_sink = hops
            
            for neighbor_id in node.neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, hops + 1))
    
    def get_node_by_id(self, node_id):
        """Get node object by ID"""
        if node_id == 0:  # Sink node ID
            return self.sink
        
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None
    
    def calculate_reward(self, current_node, neighbor_node):
        """Calculate reward for RL-based routing"""
        distance = current_node.distance_to(neighbor_node)
        normalized_distance = distance / max(self.max_longitude, self.max_latitude)
        n_param = normalized_distance * (self.dfr_max - self.dfr_min) + self.dfr_min
        
        # Ensure hop count is at least 1 to avoid division by zero
        hop_count = max(1, neighbor_node.hop_count_to_sink)
        
        # Avoid division by zero for distance
        if distance == 0:
            distance = 0.1
        
        reward = neighbor_node.energy / ((distance ** n_param) * hop_count)
        return reward
    
    def update_q_value(self, current_node, neighbor_id):
        """Update Q-value for RL-based routing"""
        neighbor_node = self.get_node_by_id(neighbor_id)
        
        if neighbor_node is None:
            return
        
        reward = self.calculate_reward(current_node, neighbor_node)
        
        # Get the best Q-value from the neighbor
        best_q_from_neighbor = 0
        if neighbor_node.neighbors:
            best_q_from_neighbor = max([neighbor_node.q_values.get(nid, 0) for nid in neighbor_node.neighbors])
        
        # Update Q-value using the formula
        old_q = current_node.q_values.get(neighbor_id, 0)
        new_q = (1 - self.alpha) * old_q + self.alpha * (reward + best_q_from_neighbor)
        current_node.q_values[neighbor_id] = new_q
    
    def find_next_forwarder(self, current_node):
        """Find the best next forwarder based on Q-values"""
        if not current_node.neighbors:
            return None
        
        # Filter out neighbors that are dead or not useful for forwarding
        valid_neighbors = []
        for neighbor_id in current_node.neighbors:
            neighbor = self.get_node_by_id(neighbor_id)
            if neighbor and neighbor.is_alive() and neighbor.node_id != current_node.node_id:
                # Prefer neighbors that are closer to sink
                if neighbor.hop_count_to_sink < current_node.hop_count_to_sink:
                    valid_neighbors.append(neighbor_id)
        
        # If no closer neighbors, use all alive neighbors
        if not valid_neighbors:
            valid_neighbors = [nid for nid in current_node.neighbors 
                             if self.get_node_by_id(nid) and self.get_node_by_id(nid).is_alive()]
        
        if not valid_neighbors:
            return None
        
        # Update Q-values for valid neighbors only
        for neighbor_id in valid_neighbors:
            self.update_q_value(current_node, neighbor_id)
        
        # Select neighbor with highest Q-value among valid neighbors
        best_neighbor_id = None
        best_q_value = float('-inf')
        
        for neighbor_id in valid_neighbors:
            q_value = current_node.q_values.get(neighbor_id, 0)
            if q_value > best_q_value:
                best_q_value = q_value
                best_neighbor_id = neighbor_id
        
        return self.get_node_by_id(best_neighbor_id) if best_neighbor_id else None
    
    def get_sensor_data(self, node_id, time_index):
        """Get sensor data from dataset for a specific node at a specific time"""
        # Map nodes 1-10 to their actual dataset, nodes 11-100 to reused data with variations
        if node_id <= 10:
            # Use original dataset for nodes 1-10
            dataset_node_id = node_id
        else:
            # For nodes 11-100, map to nodes 1-10 cyclically
            dataset_node_id = ((node_id - 1) % 10) + 1
        
        if dataset_node_id not in self.dataset:
            # Generate random data if no dataset available
            return random.uniform(20, 25)  # Random temperature
        
        # Cycle through the dataset if time exceeds dataset size
        dataset_size = len(self.dataset[dataset_node_id])
        if dataset_size == 0:
            return random.uniform(20, 25)
        
        # Use temperature data (column 11 in the dataset, 0-indexed)
        temperature_column = 11
        
        # Calculate the index based on time and transmission period (6 sec)
        actual_index = (time_index // 6) % dataset_size
        
        # Get base value from dataset
        try:
            base_value = float(self.dataset[dataset_node_id].iloc[actual_index, temperature_column])
            # Handle NaN values
            if math.isnan(base_value):
                base_value = random.uniform(21, 23)  # Default temperature range
        except (ValueError, IndexError):
            base_value = random.uniform(21, 23)  # Default temperature range
        
        # Add variation based on node_id for nodes 11-100 to create diversity
        if node_id > 10:
            # Create different temperature patterns for extended nodes
            node_offset = (node_id - 11) * 0.15  # Smaller offset for many more nodes
            seasonal_variation = math.sin((time_index / 150) + node_offset) * 2.5  # Enhanced seasonal pattern
            daily_variation = math.cos((time_index / 24) + node_offset * 0.3) * 1.2  # Daily temperature cycle
            regional_variation = math.sin((node_id / 20) + (time_index / 200)) * 1.0  # Regional climate differences
            base_value += seasonal_variation + daily_variation + regional_variation
        
        # Add small noise to sensor readings (±0.5°C) to create more varied patterns
        noise = random.uniform(-0.5, 0.5)
        return base_value + noise
    
    def sleep_scheduling(self, node, send_permission):
        """Execute sleep scheduling algorithm"""
        if node.type == NodeType.CLUSTER_HEAD or node.type == NodeType.SINK:
            # Cluster heads and sink always stay active
            return NodeMode.ACTIVE
        
        if node.mode == NodeMode.ACTIVE:
            if not send_permission:
                node.no_send_counter += 1
                if node.no_send_counter >= self.sleep_threshold:
                    node.mode = NodeMode.SLEEP
                    node.no_send_counter = 0
                    return NodeMode.SLEEP
            else:
                node.no_send_counter = 0
        else:  # Node is in SLEEP mode
            # Wake up nodes every 30 seconds (adjustable parameter)
            if self.time % 30 == 0:
                node.mode = NodeMode.ACTIVE
                return NodeMode.ACTIVE
        
        return node.mode
    
    def restrict_data_transmission(self, node, sensor_data):
        """Execute data transmission restriction algorithm"""
        if node.min_value > sensor_data:
            node.min_value = sensor_data
        
        if node.max_value < sensor_data:
            node.max_value = sensor_data
        
        # Check if change is significant
        if ((node.max_value - sensor_data) > self.change_threshold or 
            (sensor_data - node.min_value) > self.change_threshold):
            return True
        
        return False
    
    def process_regular_node(self, node, time_index):
        """Process regular node behavior"""
        # Apply sleep scheduling
        if node.mode == NodeMode.SLEEP:
            node.reduce_energy('sleep')
            # Check if it's time to wake up
            if self.time % 30 == 0:  # Wake up check every 30 seconds
                node.mode = NodeMode.ACTIVE
            return
        
        # Node is active, consume energy
        node.reduce_energy('active')
        
        # Read sensor data
        sensor_data = self.get_sensor_data(node.node_id, time_index)
        node.last_sensor_data = sensor_data
        
        # Check if should send data
        send_permission = self.restrict_data_transmission(node, sensor_data)
        
        # Update node mode based on send permission
        self.sleep_scheduling(node, send_permission)
        
        # If allowed to send and active, send data to cluster head
        if send_permission and node.mode == NodeMode.ACTIVE:
            packet = Packet(node.node_id, sensor_data, node.node_id, node.cluster_head_id)
            cluster_head = self.get_node_by_id(node.cluster_head_id)
            
            if cluster_head and cluster_head.is_alive():
                # Sending consumes energy
                node.reduce_energy('send')
                self.process_packet_reception(cluster_head, packet)
    
    def process_cluster_head(self, node, time_index):
        """Process cluster head behavior"""
        # Cluster heads are always active
        node.reduce_energy('active')
        
        # If this is a forwarding operation from another CH, it's handled in process_packet_reception
        
        # For own sensor readings (cluster heads also sense)
        sensor_data = self.get_sensor_data(node.node_id, time_index)
        node.last_sensor_data = sensor_data
        
        # Cluster heads always aggregate and forward data
        # But we don't need to do anything here as it's handled in process_packet_reception
    
    def process_packet_reception(self, node, packet):
        """Process packet reception at a node"""
        # Energy consumption for receiving
        node.reduce_energy('receive')
        
        # Add hop limit to prevent infinite loops
        if packet.hops > 15:  # Maximum 15 hops allowed for 100-node network
            return False
        
        if node.type == NodeType.SINK:
            # Packet reached the sink
            self.sink.receive_packet(packet)
            return True
        
        if node.type == NodeType.CLUSTER_HEAD:
            # Extract information and update tables
            source_node = self.get_node_by_id(packet.source_id)
            if source_node and source_node.type == NodeType.CLUSTER_HEAD:
                # Update neighbor cluster info
                node.update_neighbor_table(packet.source_id, packet.hops, source_node.energy)
            
            # Aggregate data if from cluster member
            if packet.source_id in node.cluster_members:
                aggregated_data = node.aggregate_data(packet.source_id, packet.data)
                packet.data = aggregated_data
            
            # Find next forwarder
            next_node = self.find_next_forwarder(node)
            
            if next_node and next_node.node_id != node.node_id:  # Avoid self-forwarding
                # Forward packet
                packet.sender_id = node.node_id
                packet.hops += 1
                node.reduce_energy('send')
                return self.process_packet_reception(next_node, packet)
            else:
                # No valid next hop, drop the packet
                return False
        
        return True
    
    def save_simulation_data(self):
        """Save simulation data to CSV file"""
        # Create a DataFrame with simulation metrics
        simulation_data = {
            'Time': self.time_data,
            'Live_Node_Percentage': self.live_node_percentage,
            'Total_Energy': self.energy_data,
            'Packets_Received': self.packets_received,
            'Active_Nodes': self.active_nodes_data
        }
        
        df = pd.DataFrame(simulation_data)
        
        # Save to CSV file
        csv_filename = 'rlbeep_simulation_data.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Simulation data saved to {csv_filename}")
        
        # Also save network configuration and results summary
        summary_data = {
            'Parameter': ['Num_Nodes', 'Num_Clusters', 'Network_Area', 'Send_Range', 
                         'Total_Time', 'First_Death_Time', 'Final_Live_Percentage'],
            'Value': [self.num_nodes, self.num_clusters, 
                     f"{self.max_longitude}x{self.max_latitude}", self.send_range,
                     self.total_time, self.first_death_time, 
                     self.live_node_percentage[-1]]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = 'rlbeep_simulation_summary.csv'
        summary_df.to_csv(summary_filename, index=False)
        print(f"Simulation summary saved to {summary_filename}")
    
    def run_simulation(self):
        """Run the complete simulation"""
        print(f"Starting RLBEEP simulation for {self.total_time} seconds...")
        
        # Initialize metrics
        self.first_death_time = -1
        self.live_node_percentage = []
        
        # Track cluster head rotation timing
        last_rotation_time = 0
        
        # Main simulation loop
        for t in range(self.total_time):
            self.time = t
            
            # Track the percentage of live nodes
            live_nodes = sum(1 for node in self.nodes if node.is_alive())
            live_percentage = (live_nodes / len(self.nodes)) * 100
            self.live_node_percentage.append(live_percentage)
            
            # Track additional metrics
            total_energy = sum(node.energy for node in self.nodes if node.is_alive())
            active_nodes = sum(1 for node in self.nodes if node.is_alive() and node.mode == NodeMode.ACTIVE)
            
            self.time_data.append(t)
            self.energy_data.append(total_energy)
            self.packets_received.append(self.sink.received_packets)
            self.active_nodes_data.append(active_nodes)
            
            # Introduce small random chance of node failure (independent of energy)
            # This adds randomness to node death time beyond just energy depletion
            if live_nodes == len(self.nodes) and random.random() < 0.0001:  # 0.01% chance per node per timestep
                # Randomly select a node that might fail due to environmental factors
                alive_nodes = [node for node in self.nodes if node.is_alive()]
                if alive_nodes and random.random() < 0.1:  # 10% chance to actually trigger the failure
                    unlucky_node = random.choice(alive_nodes)
                    # Drop energy to near zero to simulate imminent failure
                    unlucky_node.energy = random.uniform(0.01, 0.1)
            
            # Check for first node death
            if live_nodes < len(self.nodes) and self.first_death_time == -1:
                # Add small random offset to death time (±5 seconds) to avoid discrete values
                offset = random.randint(-5, 5)
                self.first_death_time = max(1, t + offset)  # Ensure time is at least 1
                print(f"First node died at time: {self.first_death_time} seconds")
            
            # Perform cluster head rotation at the specified interval
            if (t - last_rotation_time) >= self.ch_rotation_interval and t > 0:
                print(f"\nPerforming cluster head rotation at time: {t} seconds")
                self.perform_cluster_head_rotation()
                last_rotation_time = t
                print(f"Cluster head rotation complete at time: {t} seconds\n")
            
            # Process each node
            for node in self.nodes:
                if not node.is_alive():
                    continue
                
                if node.type == NodeType.REGULAR:
                    self.process_regular_node(node, t)
                elif node.type == NodeType.CLUSTER_HEAD:
                    self.process_cluster_head(node, t)
            
            # Optional: Print progress every 100 seconds
            if t % 100 == 0:
                print(f"Simulation time: {t}/{self.total_time} seconds, Live nodes: {live_nodes}/{len(self.nodes)}")
        
        print("Simulation complete!")
        print(f"First node death time: {self.first_death_time if self.first_death_time != -1 else 'No nodes died'}")
        print(f"Final live node percentage: {self.live_node_percentage[-1]:.2f}%")
        
        # Save simulation data to CSV
        self.save_simulation_data()
        
        # Return simulation results
        return {
            'first_death_time': self.first_death_time,
            'live_node_percentage': self.live_node_percentage,
            'final_live_percentage': self.live_node_percentage[-1]
        }
    
    def plot_results(self):
        """Plot simulation results"""
        plt.figure(figsize=(14, 6))
        
        # Plot 1: Network Topology
        plt.subplot(1, 2, 1)
        for node in self.nodes:
            if node.type == NodeType.CLUSTER_HEAD:
                plt.scatter(node.x, node.y, color='red', s=100, marker='^', label='Cluster Head' if node == self.nodes[0] else "")
            else:
                plt.scatter(node.x, node.y, color='blue', s=50, label='Regular Node' if node == self.nodes[self.num_clusters] else "")
                
                # Draw line to cluster head
                cluster_head = self.get_node_by_id(node.cluster_head_id)
                if cluster_head:
                    plt.plot([node.x, cluster_head.x], [node.y, cluster_head.y], 'k--', alpha=0.3)
        
        # Plot sink
        plt.scatter(self.sink.x, self.sink.y, color='green', s=200, marker='*', label='Sink')
        
        plt.title('Network Topology')
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Live Node Percentage Over Time (previously Plot 3)
        plt.subplot(1, 2, 2)
        plt.plot(range(self.total_time), self.live_node_percentage, 'g-')
        plt.title('Live Node Percentage Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Live Node Percentage (%)')
        plt.grid(True)
        
        # Mark first death time
        if self.first_death_time != -1:
            plt.axvline(x=self.first_death_time, color='r', linestyle='--', 
                        label=f'First Death: {self.first_death_time}s')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('rlbeep_simulation_results.png')
        plt.show()
    
    def select_new_cluster_head(self, old_ch):
        """
        Select a new cluster head from the members of the current cluster.
        The selection is based on remaining energy and position (centrality).
        """
        if not old_ch.cluster_members:
            return None  # No members to choose from
            
        # Get all potential candidates (alive nodes in the cluster)
        candidates = []
        for member_id in old_ch.cluster_members:
            member = self.get_node_by_id(member_id)
            if member and member.is_alive():
                candidates.append(member)
                
        if not candidates:
            return None  # No viable candidates
            
        # Calculate distances from each node to all other nodes in the cluster
        centrality_scores = {}
        for node in candidates:
            total_distance = 0
            for other_node in candidates:
                if node.node_id != other_node.node_id:
                    total_distance += node.distance_to(other_node)
            
            # Avoid division by zero
            if len(candidates) > 1:
                centrality_scores[node.node_id] = total_distance / (len(candidates) - 1)
            else:
                centrality_scores[node.node_id] = 0
        
        # Select new CH based on weighted score of energy and centrality
        best_score = float('-inf')
        best_node = None
        
        for node in candidates:
            # Normalize centrality (lower is better)
            if max(centrality_scores.values()) - min(centrality_scores.values()) > 0:
                normalized_centrality = 1 - ((centrality_scores[node.node_id] - min(centrality_scores.values())) / 
                                          (max(centrality_scores.values()) - min(centrality_scores.values())))
            else:
                normalized_centrality = 1
                
            # Energy score (higher is better)
            energy_score = node.energy / 300  # Normalize by updated initial energy for 100 nodes
            
            # Combined score (70% energy, 30% centrality)
            combined_score = 0.7 * energy_score + 0.3 * normalized_centrality
            
            if combined_score > best_score:
                best_score = combined_score
                best_node = node
        
        return best_node
        
    def rotate_cluster_head(self, old_ch):
        """
        Rotate the cluster head role from the current CH to a newly selected node.
        Returns the new cluster head node.
        """
        # Select new cluster head
        new_ch_node = self.select_new_cluster_head(old_ch)
        
        if not new_ch_node:
            # If no suitable replacement found, keep the old one
            return old_ch
            
        # Create a new cluster head node based on the selected regular node
        new_ch = ClusterHead(new_ch_node.node_id, new_ch_node.x, new_ch_node.y, new_ch_node.energy)
        
        # Preserve relevant data from the old node
        new_ch.hop_count_to_sink = new_ch_node.hop_count_to_sink
        new_ch.neighbors = new_ch_node.neighbors.copy()
        new_ch.q_values = new_ch_node.q_values.copy()
        new_ch.mode = NodeMode.ACTIVE  # Ensure new CH is active
        
        # Copy old CH's cluster members and remove the new CH from the list if it was a member
        new_ch.cluster_members = old_ch.cluster_members.copy()
        if new_ch.node_id in new_ch.cluster_members:
            new_ch.cluster_members.remove(new_ch.node_id)
            
        # Add the old CH as a cluster member
        new_ch.cluster_members.append(old_ch.node_id)
        
        # Update node references in the simulation
        for i, node in enumerate(self.nodes):
            if node.node_id == new_ch.node_id:
                self.nodes[i] = new_ch
                
        # Update cluster head list
        for i, ch in enumerate(self.cluster_heads):
            if ch.node_id == old_ch.node_id:
                self.cluster_heads[i] = new_ch
                
        # Create a regular node from the old CH
        old_regular_node = Node(old_ch.node_id, old_ch.x, old_ch.y, old_ch.energy)
        old_regular_node.hop_count_to_sink = old_ch.hop_count_to_sink
        old_regular_node.neighbors = old_ch.neighbors.copy()
        old_regular_node.q_values = old_ch.q_values.copy()
        old_regular_node.cluster_head_id = new_ch.node_id
        old_regular_node.mode = NodeMode.ACTIVE
        
        # Replace old CH with regular node in nodes list
        for i, node in enumerate(self.nodes):
            if node.node_id == old_ch.node_id:
                self.nodes[i] = old_regular_node
                
        # Update cluster head ID for all members
        for member_id in new_ch.cluster_members:
            member = self.get_node_by_id(member_id)
            if member:
                member.cluster_head_id = new_ch.node_id
                
        print(f"Cluster head rotated: Node {old_ch.node_id} → Node {new_ch.node_id}")
        
        return new_ch
        
    def perform_cluster_head_rotation(self):
        """Rotate cluster heads in all clusters to balance energy consumption"""
        if not self.cluster_heads:
            return
            
        # Make a copy since we'll be modifying the list
        current_chs = self.cluster_heads.copy()
        
        for old_ch in current_chs:
            self.rotate_cluster_head(old_ch)
            
        # Recalculate hop counts after rotation
        self.initialize_hop_counts()
def main():
    """Main function to run the simulation"""
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "Dataset")
    
    # Create and run simulation with 100 nodes and 40 cluster heads
    simulation = RLBEEPSimulation(
        dataset_path=dataset_path,
        num_nodes=100,  # Increased from 50 to 100
        num_clusters=40,  # Increased from 20 to 40
        max_longitude=200.0,  # Increased area to accommodate more nodes
        max_latitude=200.0,   # Increased area to accommodate more nodes
        send_range=30,        # Increased range for larger network (was 25)
        alpha=0.5,
        dfr_min=5.0,
        dfr_max=55.0,
        total_time=3000,
        sleep_threshold=10,
        change_threshold=10.0,
        ch_rotation_interval=300  # Standard rotation interval for large network
    )
    
    results = simulation.run_simulation()
    simulation.plot_results()
    
    print("\nSimulation Results Summary:")
    print(f"First Node Death Time: {results['first_death_time']} seconds")
    print(f"Final Live Node Percentage: {results['final_live_percentage']:.2f}%")


if __name__ == "__main__":
    main()
