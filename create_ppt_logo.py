#!/usr/bin/env python3
"""
PowerPoint Logo Generator for AI Deepfake Detector
Creates professional logos in various formats for presentation use
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

def create_ai_deepfake_logo():
    """Create a professional AI Deepfake Detector logo for PowerPoint"""
    
    # Create figure with high DPI for crisp quality
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Color scheme - professional gradient colors
    primary_color = '#667eea'
    secondary_color = '#764ba2'
    accent_color = '#f093fb'
    text_color = '#2c3e50'
    
    # Background with subtle gradient effect
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, extent=[0, 12, 0, 4], aspect='auto', alpha=0.1, cmap='viridis')
    
    # Main logo container with rounded corners
    main_box = FancyBboxPatch(
        (0.5, 0.5), 11, 3,
        boxstyle="round,pad=0.1",
        facecolor='white',
        edgecolor=primary_color,
        linewidth=3,
        alpha=0.95
    )
    ax.add_patch(main_box)
    
    # AI Brain Icon (left side)
    brain_center_x, brain_center_y = 2, 2
    
    # Brain outline
    brain_outline = Circle((brain_center_x, brain_center_y), 0.8, 
                          facecolor=primary_color, alpha=0.2, 
                          edgecolor=primary_color, linewidth=2)
    ax.add_patch(brain_outline)
    
    # Neural network nodes
    node_positions = [
        (brain_center_x-0.4, brain_center_y+0.3),
        (brain_center_x, brain_center_y+0.4),
        (brain_center_x+0.4, brain_center_y+0.3),
        (brain_center_x-0.3, brain_center_y),
        (brain_center_x+0.3, brain_center_y),
        (brain_center_x-0.4, brain_center_y-0.3),
        (brain_center_x, brain_center_y-0.4),
        (brain_center_x+0.4, brain_center_y-0.3),
    ]
    
    # Draw neural network connections
    connections = [
        (0, 1), (1, 2), (0, 3), (2, 4), (3, 4),
        (3, 5), (4, 7), (5, 6), (6, 7), (1, 3), (1, 4)
    ]
    
    for start, end in connections:
        x1, y1 = node_positions[start]
        x2, y2 = node_positions[end]
        ax.plot([x1, x2], [y1, y2], color=secondary_color, alpha=0.6, linewidth=1.5)
    
    # Draw nodes
    for x, y in node_positions:
        circle = Circle((x, y), 0.08, facecolor=accent_color, 
                       edgecolor=secondary_color, linewidth=1)
        ax.add_patch(circle)
    
    # Central processing node
    central_node = Circle((brain_center_x, brain_center_y), 0.12, 
                         facecolor=primary_color, edgecolor='white', linewidth=2)
    ax.add_patch(central_node)
    
    # Shield/Security Icon (right side of brain)
    shield_x, shield_y = 3.2, 2
    shield_points = np.array([
        [shield_x, shield_y + 0.5],
        [shield_x + 0.3, shield_y + 0.4],
        [shield_x + 0.3, shield_y - 0.2],
        [shield_x, shield_y - 0.5],
        [shield_x - 0.3, shield_y - 0.2],
        [shield_x - 0.3, shield_y + 0.4]
    ])
    
    shield = patches.Polygon(shield_points, facecolor=accent_color, 
                           alpha=0.3, edgecolor=secondary_color, linewidth=2)
    ax.add_patch(shield)
    
    # Checkmark in shield
    ax.plot([shield_x-0.15, shield_x-0.05, shield_x+0.15], 
            [shield_y-0.05, shield_y-0.15, shield_y+0.1], 
            color=secondary_color, linewidth=3)
    
    # Main Title
    ax.text(6, 2.7, 'AI DEEPFAKE', fontsize=28, fontweight='bold', 
            color=primary_color, ha='center', va='center',
            fontfamily='sans-serif')
    
    ax.text(6, 2.2, 'DETECTOR', fontsize=28, fontweight='bold', 
            color=secondary_color, ha='center', va='center',
            fontfamily='sans-serif')
    
    # Subtitle
    ax.text(6, 1.6, 'Advanced AI-Powered Detection System', 
            fontsize=12, color=text_color, ha='center', va='center',
            fontfamily='sans-serif', style='italic')
    
    # Technology badges
    badges = ['CNN', 'TensorFlow', 'OpenCV', 'Flask']
    badge_colors = [primary_color, secondary_color, accent_color, '#28a745']
    
    for i, (badge, color) in enumerate(zip(badges, badge_colors)):
        badge_x = 4.5 + i * 1.2
        badge_y = 1
        
        badge_box = FancyBboxPatch(
            (badge_x - 0.3, badge_y - 0.15), 0.6, 0.3,
            boxstyle="round,pad=0.05",
            facecolor=color,
            alpha=0.8
        )
        ax.add_patch(badge_box)
        
        ax.text(badge_x, badge_y, badge, fontsize=8, fontweight='bold',
                color='white', ha='center', va='center')
    
    # Version and author info
    ax.text(10.5, 0.7, 'v2.1', fontsize=10, color=text_color, 
            ha='center', va='center', fontweight='bold')
    ax.text(10.5, 0.4, 'by Omkar Kalagi', fontsize=8, color=text_color, 
            ha='center', va='center', style='italic')
    
    # Save as high-quality PNG for PowerPoint
    plt.tight_layout()
    plt.savefig('static/ai_deepfake_logo_ppt.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', transparent=False)
    
    # Save as transparent PNG for overlays
    plt.savefig('static/ai_deepfake_logo_transparent.png', dpi=300, bbox_inches='tight', 
                facecolor='none', edgecolor='none', transparent=True)
    
    plt.close()

def create_compact_logo():
    """Create a compact version for slide headers"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 2), dpi=300)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 2)
    ax.axis('off')
    
    # Colors
    primary_color = '#667eea'
    secondary_color = '#764ba2'
    
    # Compact brain icon
    brain_center = (1, 1)
    brain_circle = Circle(brain_center, 0.4, facecolor=primary_color, 
                         alpha=0.2, edgecolor=primary_color, linewidth=2)
    ax.add_patch(brain_circle)
    
    # Simple neural network
    nodes = [(0.7, 1.2), (1, 1.3), (1.3, 1.2), (0.7, 0.8), (1.3, 0.8)]
    for i, (x, y) in enumerate(nodes):
        color = primary_color if i < 3 else secondary_color
        circle = Circle((x, y), 0.05, facecolor=color)
        ax.add_patch(circle)
    
    # Connections
    connections = [(0, 1), (1, 2), (0, 3), (2, 4), (3, 4)]
    for start, end in connections:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        ax.plot([x1, x2], [y1, y2], color=secondary_color, alpha=0.6, linewidth=1)
    
    # Compact text
    ax.text(4, 1.3, 'AI DEEPFAKE DETECTOR', fontsize=20, fontweight='bold', 
            color=primary_color, ha='center', va='center')
    ax.text(4, 0.7, 'Advanced Detection System', fontsize=10, 
            color=secondary_color, ha='center', va='center', style='italic')
    
    plt.tight_layout()
    plt.savefig('static/ai_deepfake_logo_compact.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def create_icon_only():
    """Create icon-only version for small spaces"""
    fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=300)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.axis('off')
    
    # Colors
    primary_color = '#667eea'
    secondary_color = '#764ba2'
    accent_color = '#f093fb'
    
    # Brain with neural network
    brain_center = (1, 1)
    brain_circle = Circle(brain_center, 0.8, facecolor=primary_color, 
                         alpha=0.1, edgecolor=primary_color, linewidth=3)
    ax.add_patch(brain_circle)
    
    # Neural network pattern
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    radius = 0.5
    
    for i, angle in enumerate(angles):
        x = 1 + radius * np.cos(angle)
        y = 1 + radius * np.sin(angle)
        
        # Connect to center
        ax.plot([1, x], [1, y], color=secondary_color, alpha=0.6, linewidth=2)
        
        # Node
        circle = Circle((x, y), 0.08, facecolor=accent_color, 
                       edgecolor=secondary_color, linewidth=1)
        ax.add_patch(circle)
    
    # Central node
    central = Circle((1, 1), 0.12, facecolor=primary_color, 
                    edgecolor='white', linewidth=2)
    ax.add_patch(central)
    
    plt.tight_layout()
    plt.savefig('static/ai_deepfake_icon.png', dpi=300, bbox_inches='tight', 
                facecolor='none', edgecolor='none', transparent=True)
    plt.close()

def main():
    """Generate all logo variations"""
    print("🎨 Creating AI Deepfake Detector logos for PowerPoint...")
    
    try:
        # Create main logo
        print("📊 Creating main presentation logo...")
        create_ai_deepfake_logo()
        
        # Create compact version
        print("📋 Creating compact header logo...")
        create_compact_logo()
        
        # Create icon version
        print("🔸 Creating icon-only version...")
        create_icon_only()
        
        print("\n✅ Logo creation completed!")
        print("📁 Generated files:")
        print("   • ai_deepfake_logo_ppt.png (Main presentation logo)")
        print("   • ai_deepfake_logo_transparent.png (Transparent overlay)")
        print("   • ai_deepfake_logo_compact.png (Compact header)")
        print("   • ai_deepfake_icon.png (Icon only)")
        print("\n💡 Use these in your PowerPoint presentation!")
        
    except Exception as e:
        print(f"❌ Error creating logos: {e}")
        print("📦 Make sure matplotlib and PIL are installed:")
        print("   pip install matplotlib pillow")

if __name__ == "__main__":
    main()
