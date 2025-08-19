#!/usr/bin/env python3
"""
Add Advanced Loading Animations to All Templates
"""

import os
import re
from pathlib import Path

def add_advanced_loader_to_template(file_path):
    """Add advanced loader to a single template file."""
    print(f"Adding advanced loader to {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Add loader script reference before closing head tag
    head_end = content.find('</head>')
    if head_end != -1 and 'advanced-loader.js' not in content:
        loader_script = '''    <script src="{{ url_for('static', filename='advanced-loader.js') }}" defer></script>
'''
        content = content[:head_end] + loader_script + content[head_end:]
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Added advanced loader to {file_path}")
        return True
    else:
        print(f"  ⚪ Advanced loader already present in {file_path}")
        return False

def main():
    """Add advanced loader to all templates."""
    print("⚡ Adding Advanced Loading Animations")
    print("=" * 50)
    
    templates_dir = Path('templates')
    html_files = list(templates_dir.glob('*.html'))
    
    updated_count = 0
    
    for html_file in html_files:
        if add_advanced_loader_to_template(html_file):
            updated_count += 1
    
    print(f"\n✅ Advanced loader addition completed!")
    print(f"📊 Updated {updated_count} out of {len(html_files)} templates")
    print("🎨 Beautiful AI-themed loading animations added!")
    print("⚡ Smooth page transitions and loading experiences ready!")

if __name__ == "__main__":
    main()
