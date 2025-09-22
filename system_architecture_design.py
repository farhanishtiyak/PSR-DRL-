import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
colors = {
    'wsn': '#E8F4F8',
    'data': '#F0F8E8', 
    'drl': '#F8E8E8',
    'power': '#F8F0E8',
    'energy': '#E8E8F8',
    'sub_module': '#FFFFFF',
    'border': '#2C3E50',
    'arrow': '#34495E',
    # Different arrow colors
    'arrow_state': '#2E7D32',      # Dark green for state info
    'arrow_data': '#1976D2',       # Blue for data flow
    'arrow_control': '#D32F2F',    # Red for control commands
    'arrow_routing': '#FF6F00',    # Orange for routing
    'arrow_feedback': '#7B1FA2',   # Purple for feedback
    'arrow_status': '#00796B',     # Teal for status
    'arrow_topology': '#5D4037',   # Brown for topology
    'arrow_reports': '#388E3C'     # Green for reports
}

# Helper function to create rounded rectangles
def create_module_box(ax, x, y, width, height, title, details, color, text_color='black'):
    # Main box
    box = FancyBboxPatch(
        (x, y), width, height+0.3,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor=colors['border'],
        linewidth=2.5
    )
    ax.add_patch(box)
    
    # Title
    ax.text(x + width/2, y + height - 0.15, title, 
            fontsize=28, fontweight='bold', ha='center', va='center', color=text_color)
    
    # Details
    detail_text = '\n'.join(details)
    ax.text(x + width/2, y + height/2 - 0.25, detail_text,
            fontsize=19, ha='center', va='center', color=text_color,
            wrap=True)

# Helper function to create sub-module boxes
def create_sub_module_box(ax, x, y, width, height, title, color):
    sub_box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor=colors['border'],
        linewidth=1.5
    )
    ax.add_patch(sub_box)
    ax.text(x + width/2, y + height/2, title, 
            fontsize=15, fontweight='bold', ha='center', va='center')

# Helper function to create unidirectional arrows
def create_arrow(ax, start, end, label='', color=colors['arrow'], offset_x=0.0, offset_y=0.0):
    arrow = ConnectionPatch(start, end, "data", "data",
                          arrowstyle='->', shrinkA=8, shrinkB=8,
                          color=color, linewidth=2.5)
    ax.add_patch(arrow)
    
    # Add label if provided
    if label:
        mid_x = (start[0] + end[0]) / 2 + offset_x
        mid_y = (start[1] + end[1]) / 2 + offset_y
        ax.text(mid_x, mid_y, label, fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                fontweight='bold')

# WSN Module (Top Left) - Slightly reduced width
wsn_details = [
    '• Node deployment & topology',
    '• Neighbor discovery',
    '• Cluster head selection',
    '• Cluster Formation',
]
create_module_box(ax, 0.5, 7.5, 3, 2, 'WSN Module', wsn_details, colors['wsn'])

# Data Processing Module (Top Right) - Slightly reduced width
data_details = [
    '• Data collection & filtering',
    '• Data aggregation',
    '• Priority-based transmission',
]
create_module_box(ax, 8.3, 7.5, 3.4, 2, 'Data Processing\nModule', data_details, colors['data'])

# DRL Module (Center - Main Controller) - Adjusted for better sub-module placement
drl_details = [
    '',  # Empty lines to push text below sub-modules
    '',
    '',
    '',
    '• State analysis',
    '• Optimal action selection',
]
create_module_box(ax, 4.2, 4, 3.6, 2.8, 'DRL Module', drl_details, colors['drl'])

# Sub-modules inside DRL Module - Moved down to avoid text overlap
create_sub_module_box(ax, 4.5, 5.6, 1.4, 0.4, 'Policy Network', colors['sub_module'])
create_sub_module_box(ax, 6.1, 5.6, 1.4, 0.4, 'Target Network', colors['sub_module'])
create_sub_module_box(ax, 4.5, 5.0, 3, 0.4, 'Experience Replay Buffer', colors['sub_module'])

# Power Saving Module (Bottom Left) - Slightly reduced width
power_details = [
    '• Intelligent sleep scheduling',
    '• Activity control',
    '• Wake-up coordination',
]
create_module_box(ax, 0.5, 1, 3.2, 2, 'Power Saving\nModule', power_details, colors['power'])

# Energy Management Module (Bottom Right) - Slightly reduced width
energy_details = [
    '• Energy consumption monitoring',
    '• Performance metrics',
    '• Real-time feedback',
]
create_module_box(ax, 8.3, 1, 3.5, 2, 'Energy Management\nModule', energy_details, colors['energy'])

# Create unidirectional arrows between modules with improved labels

# WSN -> DRL: Network State Information
create_arrow(ax, (3.3, 7.5), (4.4, 6.3), 'Network State\nInformation', colors['arrow_state'], offset_x=-0.4, offset_y=-0.5)

# Data Processing -> DRL: Processed Data & Priorities
create_arrow(ax, (8.5, 7.8), (7.6, 6.3), 'Processed Data\n& Priorities', colors['arrow_data'], offset_x=0.6, offset_y=-0.3)

# DRL -> Power Saving: Sleep/Wake Commands
create_arrow(ax, (4.4, 4.6), (3.5, 3.2), 'Sleep Commands', colors['arrow_control'], offset_x=0.6, offset_y=-0.3)

# DRL -> Energy Management: Routing Decisions & Control Signals
create_arrow(ax, (7.6, 4.2), (8.5, 3.2), 'Smart Routing\n& Energy Control', colors['arrow_routing'], offset_x=-0.8, offset_y=-0.3)

# Energy Management -> DRL: Reward & Feedback
create_arrow(ax, (8.7, 3.3), (7.6, 5.2), 'Reward Signal\n& Feedback', colors['arrow_feedback'], offset_x=0.7, offset_y=0.2)

# WSN -> Power Saving: Node Activity Status
create_arrow(ax, (2.2, 7.5), (2.2, 3.2), 'Node Activity\nStatus', colors['arrow_status'], offset_x=-0.6, offset_y=0.0)

# WSN -> Data Processing: Network Topology
create_arrow(ax, (3.3, 8.5), (8.5, 8.5), 'Network Topology\n& Cluster Info', colors['arrow_topology'], offset_x=0.0, offset_y=0.4)

# Power Saving -> Energy Management: Energy Consumption Reports (corrected direction)
create_arrow(ax, (3.5, 2.2), (8.5, 2.2), 'Energy Consumption\nReports', colors['arrow_reports'], offset_x=0.0, offset_y=0.4)

plt.tight_layout()
plt.savefig('/home/ishtiyak/Desktop/Thesis/MatLab/All figures/PSR_DRL_System_Architecture_Refined.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/ishtiyak/Desktop/Thesis/MatLab/All figures/PSR_DRL_System_Architecture_Refined.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Refined system architecture diagram has been generated successfully!")
print("Files saved:")
print("- PSR_DRL_System_Architecture_Refined.png/pdf")
print("- PSR_DRL_Flow_Diagram.png/pdf")
