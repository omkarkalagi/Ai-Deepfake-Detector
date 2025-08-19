#!/usr/bin/env python3
"""
Add Interactive Enhancements to All Templates
"""

import os
import re
from pathlib import Path

def add_interactive_enhancements_to_template(file_path):
    """Add interactive enhancements to a single template file."""
    print(f"Adding interactive enhancements to {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Add interactive enhancements script reference before closing head tag
    head_end = content.find('</head>')
    if head_end != -1 and 'interactive-enhancements.js' not in content:
        interactive_script = '''    <script src="{{ url_for('static', filename='interactive-enhancements.js') }}" defer></script>
'''
        content = content[:head_end] + interactive_script + content[head_end:]
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Added interactive enhancements to {file_path}")
        return True
    else:
        print(f"  ⚪ Interactive enhancements already present in {file_path}")
        return False

def main():
    """Add interactive enhancements to all templates."""
    print("🎨 Adding Interactive Enhancements")
    print("=" * 50)
    
    templates_dir = Path('templates')
    html_files = list(templates_dir.glob('*.html'))
    
    updated_count = 0
    
    for html_file in html_files:
        if add_interactive_enhancements_to_template(html_file):
            updated_count += 1
    
    print(f"\n✅ Interactive enhancements addition completed!")
    print(f"📊 Updated {updated_count} out of {len(html_files)} templates")
    print("🎭 Amazing interactive effects and animations added!")
    print("✨ Your website is now highly engaging and interactive!")

if __name__ == "__main__":
    main()
