#!/usr/bin/env python3
"""
Fix template syntax errors in HTML files
"""

import os
import re
from pathlib import Path

def fix_template_syntax():
    """Fix template syntax errors in all HTML files."""
    templates_dir = Path('templates')
    
    # Pattern to find escaped quotes in url_for calls
    pattern = r"url_for\(\\'static\\', filename=\\'theme-manager\.js\\'\)"
    replacement = "url_for('static', filename='theme-manager.js')"
    
    for html_file in templates_dir.glob('*.html'):
        try:
            print(f"Fixing {html_file}...")
            
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fix escaped quotes
            original_content = content
            content = re.sub(pattern, replacement, content)
            
            if content != original_content:
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✅ Fixed {html_file}")
            else:
                print(f"  ⚪ No changes needed for {html_file}")
                
        except Exception as e:
            print(f"  ❌ Error fixing {html_file}: {e}")

if __name__ == "__main__":
    print("🔧 Fixing template syntax errors...")
    fix_template_syntax()
    print("✅ Template syntax fixes completed!")
