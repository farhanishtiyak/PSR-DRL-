import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle, Rectangle, Polygon
import numpy as np

# Create figure and axis for DRL Architecture
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors for DRL components
colors = {
    'environment': '#E8F4FD',   # Light blue
    'agent': '#FFF0F5',         # Light pink
    'network': '#F0FFF0',       # Light green
    'memory': '#FFFACD',        # Light yellow
    'training': '#FFE4E1',      # Light coral
    'border': '#2C3E50',        # Dark blue
    'arrow': '#4A90E2'          # Blue arrows
}

def create_simple_box(ax, x, y, width, height, title, color, fontsize=11):
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2",
                        facecolor=color, edgecolor=colors['border'], linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, title, fontsize=fontsize, fontweight='bold', 
            ha='center', va='center')

def create_neural_box(ax, x, y, width, height, title, color):
    # Main box
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1",
                        facecolor=color, edgecolor=colors['border'], linewidth=2)
    ax.add_patch(box)
    
    # Title
    ax.text(x + width/2, y + height - 0.25, title, fontsize=12, fontweight='bold', 
            ha='center', va='center')
    
    # Simple neural network visualization - smaller and cleaner
    layers = [3, 4, 4, 2]  # nodes per layer
    layer_spacing = (width - 0.8) / (len(layers) - 1)
    
    for i, nodes in enumerate(layers):
        layer_x = x + 0.4 + i * layer_spacing
        node_spacing = (height - 0.8) / max(nodes, 1)
        start_y = y + 0.4 + (height - 0.8) / 2 - (nodes * node_spacing) / 2
        
        for j in range(nodes):
            node_y = start_y + j * node_spacing
            circle = Circle((layer_x, node_y), 0.06, facecolor='white', 
                          edgecolor=colors['border'], linewidth=1)
            ax.add_patch(circle)
        
        # Connect to next layer with thinner lines
        if i < len(layers) - 1:
            for j in range(nodes):
                for k in range(layers[i + 1]):
                    start_node_y = start_y + j * node_spacing
                    next_layer_x = x + 0.4 + (i + 1) * layer_spacing
                    next_start_y = y + 0.4 + (height - 0.8) / 2 - (layers[i + 1] * (height - 0.8) / max(layers[i + 1], 1)) / 2
                    end_node_y = next_start_y + k * ((height - 0.8) / max(layers[i + 1], 1))
                    
                    ax.plot([layer_x + 0.06, next_layer_x - 0.06], 
                           [start_node_y, end_node_y], 
                           color=colors['border'], linewidth=0.2, alpha=0.5)

def create_arrow(ax, start, end, label='', curved=False):
    if curved:
        # Create curved arrow
        mid_x = (start[0] + end[0]) / 2
        mid_y = max(start[1], end[1]) + 1
        
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color=colors['arrow'],
                                 connectionstyle=f"arc3,rad=0.3"))
    else:
        arrow = ConnectionPatch(start, end, "data", "data", arrowstyle='->', 
                              shrinkA=5, shrinkB=5, color=colors['arrow'], linewidth=2)
        ax.add_patch(arrow)
    
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        if curved:
            mid_y += 0.5
        ax.text(mid_x, mid_y, label, fontsize=10, ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.25", facecolor='white', alpha=0.9, edgecolor='lightgray'))

# Title
ax.text(8, 9.5, 'Deep Reinforcement Learning Architecture for WSN Routing', 
        fontsize=16, fontweight='bold', ha='center', va='center')

# Row 1: Environment and Action (Top) - Reordered for better flow
create_simple_box(ax, 1, 8.2, 3, 1, 'WSN Environment', colors['environment'])
create_simple_box(ax, 12, 8.2, 3, 1, 'Action Selection\n(ε-greedy)', colors['agent'])

# Row 2: Agent Components (Middle-Top) - Better spacing
create_simple_box(ax, 1, 6.2, 2.8, 1, 'State\nObservation', colors['agent'])
create_neural_box(ax, 4.5, 5.8, 3.2, 1.5, 'Policy Network\n(DQN)', colors['network'])
create_neural_box(ax, 8.5, 5.8, 3.2, 1.5, 'Target Network\n(DQN)', colors['network'])
create_simple_box(ax, 12.5, 6.2, 2.8, 1, 'Reward\nCalculation', colors['agent'])

# Row 3: Memory & Processing (Middle) - Better spacing
create_simple_box(ax, 1.5, 3.8, 3, 1, 'Experience Replay\nBuffer', colors['memory'])
create_simple_box(ax, 6, 3.8, 2.8, 1, 'Mini-Batch\nSampling', colors['memory'])
create_simple_box(ax, 10.5, 3.8, 3, 1, 'Loss Calculation\n(Huber Loss)', colors['training'])

# Row 4: Training (Bottom) - Better spacing
create_simple_box(ax, 3.5, 1.8, 3.5, 1, 'Gradient Update\n& Backpropagation', colors['training'])
create_simple_box(ax, 8.5, 1.8, 3.5, 1, 'Target Network\nUpdate', colors['training'])

# Create logical flow arrows with better spacing and no overlapping labels
# Environment to State Observation
create_arrow(ax, (2.5, 8.2), (2.5, 7.2), 'State\nVector')

# State processing flow
create_arrow(ax, (3.8, 6.7), (4.5, 6.5), 'Features')
create_arrow(ax, (7.7, 6.5), (8.5, 6.5), 'Architecture')
create_arrow(ax, (11.7, 6.5), (12.5, 6.7), 'Q-values')

# Action execution
create_arrow(ax, (13.5, 8.2), (13.5, 7.2), 'Action')

# Reward feedback
create_arrow(ax, (3.5, 8.7), (12.5, 8.7), 'Environment\nFeedback')

# Experience storage
create_arrow(ax, (2.5, 6.2), (3, 4.8), 'Store\nExperience')

# Learning pipeline
create_arrow(ax, (4.5, 4.3), (6, 4.3), 'Sample')
create_arrow(ax, (8.8, 4.3), (10.5, 4.3), 'Batch')

# Training flow
create_arrow(ax, (12, 3.8), (6, 2.8), 'Loss')
create_arrow(ax, (5.2, 2.8), (5.2, 5.8), 'Updates')

# Target network sync
create_arrow(ax, (7, 2.3), (8.5, 2.3), 'Copy')
create_arrow(ax, (10, 2.8), (10, 5.8), 'Targets')

# Target to loss connection
create_arrow(ax, (10.5, 5.8), (11.5, 4.8), 'Target\nQ-values')

# Add learning cycle indicator
ax.text(8, 0.5, 'Continuous Learning: Interaction → Experience Collection → Batch Training → Policy Improvement', 
        fontsize=12, ha='center', va='center', style='italic',
        bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('/home/ishtiyak/Desktop/Thesis/MatLab/All figures/PSR_DRL_Architecture.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/ishtiyak/Desktop/Thesis/MatLab/All figures/PSR_DRL_Architecture.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("✅ PSR-DRL Architecture diagram created successfully!")
print("📁 Files saved:")
print("   - PSR_DRL_Architecture.png")
print("   - PSR_DRL_Architecture.pdf")
