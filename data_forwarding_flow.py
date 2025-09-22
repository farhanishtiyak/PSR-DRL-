import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

# Create separate figures for better clarity - more compact
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))

# Colors
colors = {
    'sensor': '#E8F4FD',      # Light blue
    'decision': '#FFF0F5',    # Light pink  
    'action': '#F0FFF0',      # Light green
    'cluster': '#FFFACD',     # Light yellow
    'sink': '#FFE4E1',        # Light coral
    'border': '#2C3E50'       # Dark blue
}

def create_box(ax, x, y, width, height, text, color, fontsize=14):
    """Create a rounded box with text"""
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1",
                        facecolor=color, edgecolor=colors['border'], linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, fontsize=fontsize+4, fontweight='bold',
            ha='center', va='center')

def create_diamond(ax, x, y, width, height, text, color):
    """Create diamond shape for decisions"""
    diamond = patches.RegularPolygon((x + width/2, y + height/2), 4, 
                                   radius=width/2, orientation=np.pi/4,
                                   facecolor=color, edgecolor=colors['border'], linewidth=2)
    ax.add_patch(diamond)
    ax.text(x + width/2, y + height/2, text, fontsize=16, fontweight='bold',
            ha='center', va='center')

def create_arrow(ax, start_x, start_y, end_x, end_y, label=''):
    """Create simple arrow"""
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#444444'))
    if label:
        mid_x, mid_y = (start_x + end_x)/2, (start_y + end_y)/2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=16, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.9))

# FIGURE 1: Regular Node Forwarding - Compact design
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 11)
# ax1.set_title('Regular Node Data Forwarding Strategy', fontsize=16, fontweight='bold', pad=20)
ax1.axis('off')

# Regular node flow with compact boxes
create_box(ax1, 3, 10, 4, 0.8, 'Sensor Data Collection', colors['sensor'])
create_diamond(ax1, 3.5, 8, 3, 1.3, 'Significant Data\nTransmission\nModule\n(Data > Threshold)', colors['decision'])
create_box(ax1, 0.5, 6, 3, 0.8, 'Sleep Mode', colors['action'])
create_box(ax1, 6.5, 6, 3, 0.8, 'DQL State\nEvaluation', colors['decision'])
create_diamond(ax1, 4, 4, 2, 1.5, 'Action\nSelection', colors['decision'])
create_box(ax1, 0.5, 2, 2.8, 0.8, 'Transmit to\nCluster Head', colors['action'])
create_box(ax1, 3.6, 2, 2.8, 0.8, 'Direct to\nSink', colors['action'])
create_box(ax1, 6.7, 2, 2.8, 0.8, 'Enter Sleep\nMode', colors['action'])
create_box(ax1, 3, 0.2, 4.5, 0.8, 'Energy Update & Reward Calculation', colors['cluster'])

# Arrows for regular node - compact spacing
create_arrow(ax1, 5, 10, 5, 9.5)
create_arrow(ax1, 4, 8, 2.5, 6.8, 'No')
create_arrow(ax1, 6, 8, 7.5, 6.8, 'Yes')
create_arrow(ax1, 8, 6, 5.5, 5.5)
create_arrow(ax1, 4.5, 4, 2.2, 2.8, 'CH')
create_arrow(ax1, 5, 4, 5, 2.8, 'Sink')
create_arrow(ax1, 5.5, 4, 7.8, 2.8, 'Sleep')
create_arrow(ax1, 2, 2, 4, 1)
create_arrow(ax1, 5, 2, 5, 1)
create_arrow(ax1, 8, 2, 6, 1)

# FIGURE 2: Cluster Head Forwarding - Compact design
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 11)
# ax2.set_title('Cluster Head Data Forwarding Strategy', fontsize=16, fontweight='bold', pad=20)
ax2.axis('off')

# Cluster head flow with compact boxes
create_box(ax2, 3, 10, 4, 0.8, 'Receive Data from Member Nodes', colors['sensor'])
create_box(ax2, 3, 8.8, 4, 0.8, 'Data Aggregation', colors['cluster'])
create_diamond(ax2, 3.4, 6.5, 3.6, 1.5, 'Significant Data\nTransmission\nModule\n(Data > Threshold)', colors['decision'])
create_box(ax2, 0.5, 4.5, 3, 0.8, 'Drop Data\n(No Transmission)', colors['action'])
create_box(ax2, 6.5, 4.5, 3, 0.8, 'DQL Routing\nDecision', colors['decision'])
create_diamond(ax2, 4, 2.5, 2, 1.5, 'Route\nSelection', colors['decision'])
create_box(ax2, 0.5, 0.2, 2.8, 0.8, 'Direct to\nSink', colors['action'])
create_box(ax2, 3.6, 0.2, 2.8, 0.8, 'Multi-hop via\nOther CH', colors['action'])
create_box(ax2, 6.7, 0.2, 2.8, 0.8, 'Cluster Head\nRotation Check', colors['cluster'])

# Arrows for cluster head - compact spacing
create_arrow(ax2, 5, 10, 5, 9.6)
create_arrow(ax2, 5, 8.8, 5, 8)
create_arrow(ax2, 4, 6.5, 2.5, 5.3, 'No')
create_arrow(ax2, 6.4, 6.5, 7.5, 5.3, 'Yes')
create_arrow(ax2, 8, 4.5, 5.5, 4)
create_arrow(ax2, 4.5, 2.5, 2.2, 1, 'Direct')
create_arrow(ax2, 5, 2.5, 5, 1, 'Multi-hop')
create_arrow(ax2, 5.5, 2.5, 7.8, 1, 'Rotate')

# Save both figures separately
plt.figure(fig1.number)
plt.tight_layout()
plt.savefig('/home/ishtiyak/Desktop/Thesis/MatLab/All figures/regular_node_forwarding.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/ishtiyak/Desktop/Thesis/MatLab/All figures/regular_node_forwarding.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')

plt.figure(fig2.number)
plt.tight_layout()
plt.savefig('/home/ishtiyak/Desktop/Thesis/MatLab/All figures/cluster_head_forwarding.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/ishtiyak/Desktop/Thesis/MatLab/All figures/cluster_head_forwarding.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')

plt.show()

print("✅ Data Forwarding Flow diagrams created successfully!")
print("📁 Files saved:")
print("   - regular_node_forwarding.png")
print("   - regular_node_forwarding.pdf")
print("   - cluster_head_forwarding.png")
print("   - cluster_head_forwarding.pdf")
