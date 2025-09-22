import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np

# Create plot and axis for PSR-DRL Architecture
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors for PSR-DRL
colors = {
    'experience': '#E6F3FF',    # Light blue for experience replay
    'network': '#E8E8E8',       # Light gray for networks
    'environment': '#D4D4D4',   # Gray for WSN environment
    'loss': '#FFFFFF',          # White for loss function
    'state': '#E8F5E8',         # Light green for state
    'action': '#FFE8E8',        # Light red for actions
    'reward': '#FFF8E8',        # Light yellow for reward
    'border': '#000000',        # Black borders
    # Different colors for each arrow type
    'arrow_state': '#2E7D32',      # Dark green for state
    'arrow_action': '#D32F2F',     # Dark red for actions  
    'arrow_reward': '#F57C00',     # Orange for rewards
    'arrow_experience': '#7B1FA2', # Purple for experience
    'arrow_gradient': '#1976D2',   # Blue for gradients
    'arrow_update': '#388E3C',     # Green for updates
    'arrow_qvalue': '#E91E63',     # Pink for Q-values
    'arrow_sample': '#00796B'      # Teal for sampling
}

def create_box(ax, x, y, width, height, text, color, fontsize=14, bold=True):
    """Create a rectangular box with text"""
    box = Rectangle((x, y), width, height, facecolor=color, 
                   edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text, fontsize=fontsize+4, 
            fontweight=weight, ha='center', va='center')

def create_neural_network(ax, x, y, width, height, title, layer_config):
    """Create a neural network visualization with specific layer configuration"""
    # Main box
    box = Rectangle((x, y), width, height, facecolor=colors['network'], 
                   edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(box)
    
    # Title
    ax.text(x + width/2, y + height - 0.3, title, fontsize=14+4, 
            fontweight='bold', ha='center', va='center')
    
    # Neural network layers based on PSR-DRL architecture: 9-64-64-3
    layers = layer_config
    layer_x_positions = np.linspace(x + 0.3, x + width - 0.3, len(layers))
    
    for i, (layer_x, nodes) in enumerate(zip(layer_x_positions, layers)):
        node_y_positions = np.linspace(y + 0.4, y + height - 0.6, nodes)
        
        # Draw nodes
        for j, node_y in enumerate(node_y_positions):
            if i == 0:  # Input layer (9 state features) - green nodes
                circle = Circle((layer_x, node_y), 0.03, facecolor='#4CAF50', 
                              edgecolor='black', linewidth=0.5)
            elif i == len(layers) - 1:  # Output layer (3 actions) - red nodes
                circle = Circle((layer_x, node_y), 0.04, facecolor='#F44336', 
                              edgecolor='black', linewidth=0.5)
            else:  # Hidden layers (64 neurons each) - white nodes
                circle = Circle((layer_x, node_y), 0.03, facecolor='white', 
                              edgecolor='black', linewidth=0.5)
            ax.add_patch(circle)
        
        # Add layer size numbers below each layer
        ax.text(layer_x, y + 0.1, str(nodes), fontsize=11, fontweight='bold', 
                ha='center', va='center', color='darkred',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.8))
        
        # Draw connections to next layer
        if i < len(layers) - 1:
            next_layer_x = layer_x_positions[i + 1]
            next_node_positions = np.linspace(y + 0.4, y + height - 0.6, layers[i + 1])
            
            # Sample connections to avoid clutter
            for j in range(0, len(node_y_positions), max(1, len(node_y_positions)//3)):
                for k in range(0, len(next_node_positions), max(1, len(next_node_positions)//3)):
                    ax.plot([layer_x + 0.03, next_layer_x - 0.03], 
                           [node_y_positions[j], next_node_positions[k]], 
                           'k-', linewidth=0.2, alpha=0.4)

def create_arrow(ax, start, end, color, label='', label_pos='mid', offsetx=0, offsety=0.2, curved=False):
    """Create an arrow with optional label and curve"""
    if curved:
        # Create curved arrow
        connectionstyle = "arc3,rad=0.3"
        arrow = FancyArrowPatch(start, end, arrowstyle='->', 
                               color=color, linewidth=2, mutation_scale=15,
                               connectionstyle=connectionstyle)
    else:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', 
                               color=color, linewidth=2, mutation_scale=15)
    ax.add_patch(arrow)
    
    if label:
        if label_pos == 'mid':
            mid_x = (start[0] + end[0]) / 2 + offsetx
            mid_y = (start[1] + end[1]) / 2 + offsety
        elif label_pos == 'start':
            mid_x = start[0] + 0.3 + offsetx
            mid_y = start[1] + offsety
        elif label_pos == 'end':
            mid_x = end[0] - 0.3 + offsetx
            mid_y = end[1] + offsety
        
        ax.text(mid_x, mid_y, label, fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, 
                         edgecolor='gray', linewidth=1))

# Experience Replay Buffer (left side)
create_box(ax, 0.5, 2.5, 2.2, 5, 'Experience\nReplay Buffer\n\n< s, a, r, s\' >\n\nCapacity:\n10,000', 
          colors['experience'], fontsize=14)

# Target Network (top center) - PSR-DRL specific: 9-64-64-3
create_neural_network(ax, 4, 7, 3.5, 2.2, 'Target Network Q(s,a;θ⁻)', [9, 64, 64, 3])

# Policy Network (middle center) - PSR-DRL specific: 9-64-64-3  
create_neural_network(ax, 4, 4.2, 3.5, 2.2, 'Policy Network Q(s,a;θ)', [9, 64, 64, 3])

# WSN Environment (moved back left with reduced width)
create_box(ax, 10.5, 4, 2.5, 3, 'WSN Environment\n\n• Sensor Nodes\n• Cluster Heads\n• Energy Levels\n• Network Topology\n\n9D State Formation', 
          colors['environment'], fontsize=12)

# Loss Function (moved to top right, replacing state vector)
create_box(ax, 8.5, 7.5, 4.5, 1.8, 'Loss Function\nL(θ) = E[(r + γmax Q(s\',a\';θ⁻) - Q(s,a;θ))²]\nSmoothL1 Loss (Huber Loss)', 
          colors['loss'], fontsize=12)

# Action Space (moved lower)
create_box(ax, 8.5, 3.2, 1.8, 1.8, 'Action Space\n\na₀: To CH\na₁: To Sink\na₂: Sleep', 
          colors['action'], fontsize=15)

# Reward Function (moved down and made wider)
create_box(ax, 8.5, 1.0, 5.3, 1.5, 'Multi-Objective Reward\nR = R_base + R_energy + R_lifetime + R_distance', 
          colors['reward'], fontsize=13)

# PSR-DRL specific arrows and flow

# 1. State observation from WSN to policy network (updated coordinates)
create_arrow(ax, (10.5, 5.5), (7.5, 5.2), colors['arrow_state'], '9D State s', offsetx=0.1, offsety=0.3)

# 2. Experience to policy network (state input)
create_arrow(ax, (2.7, 4.5), (4, 5.2), colors['arrow_sample'], 'Batch\nSampling', offsetx=0, offsety=0.6)

# 3. Experience to target network 
create_arrow(ax, (2.7, 7), (4, 8), colors['arrow_sample'], 'Next State s\'', offsetx=-0.1, offsety=0.6)

# 4. Action selection (ε-greedy) - updated coordinates for moved action space
create_arrow(ax, (7.5, 5.2), (8.5, 4.1), colors['arrow_action'], 'ε-greedy\nAction', offsetx=-0.1, offsety=-0.6)

# 5. Action execution to environment - updated coordinates  
create_arrow(ax, (10.3, 4.1), (10.5, 4.5), colors['arrow_action'], 'Execute', offsetx=0.3, offsety=-0.6)

# 6. Reward feedback - updated coordinates for moved reward function
create_arrow(ax, (11.5, 4), (11.5, 2.5), colors['arrow_reward'], 'Reward', offsetx=0.4, offsety=0)

# 7. Experience storage - updated coordinates
create_arrow(ax, (8.5, 1.7), (2.7, 3), colors['arrow_experience'], 'Store Experience', offsetx=-0.3, offsety=0.8, curved=True)

# 8. Target Q-values to loss
create_arrow(ax, (6.5, 7.8), (8.5, 7.8), colors['arrow_qvalue'], 'Target Q', offsetx=0.4, offsety=0.3)

# 9. Policy Q-values to loss
create_arrow(ax, (7.5, 6), (8.5, 7.5), colors['arrow_gradient'], 'Current Q', offsetx=-0.5, offsety=0.0)


# 10. Gradient update
create_arrow(ax, (9, 7.5), (7.5, 5.5), colors['arrow_update'], 'Gradient ∂L/∂θ', offsetx=0.7, offsety=0)

# 11. Target network update
create_arrow(ax, (5.7, 6.4), (5.7, 7), colors['arrow_update'], 'τ-update', offsetx=-0.50, offsety=0)

# Add title
# ax.text(7, 9.7, 'Deep Q-Learning Architecture for PSR-DRL', 
#         fontsize=14, fontweight='bold', ha='center', va='center')

# Add notation box (where training parameters were)
ax.text(1, 1.2, 'Notation:\n• s: Current state (9D vector)\n• a: Action (CH/Sink/Sleep)\n• s\': Next state\n• θ: Policy network parameters\n• θ⁻: Target network parameters', 
        fontsize=13, ha='left', va='center',
        bbox=dict(boxstyle="round,pad=0.6", facecolor='lightcyan', alpha=0.8, 
                 edgecolor='darkblue', linewidth=1.5))

plt.tight_layout()
plt.savefig('/home/ishtiyak/Desktop/Thesis/MatLab/All figures/DRL_Detailed_Architecture.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/ishtiyak/Desktop/Thesis/MatLab/All figures/DRL_Detailed_Architecture.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("✅ Detailed DRL Architecture diagram created successfully!")
print("📁 Files saved:")
print("   - DRL_Detailed_Architecture.png")
print("   - DRL_Detailed_Architecture.pdf")
