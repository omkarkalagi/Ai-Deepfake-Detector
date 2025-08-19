#!/usr/bin/env python3
"""
Script to update all HTML pages with new navigation and theme system
"""

import os
import re
from pathlib import Path

def update_page_navigation(file_path):
    """Update navigation in a single HTML file."""
    print(f"Updating {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update body background
    content = re.sub(
        r'background: linear-gradient\(135deg, #667eea 0%, #764ba2 100%\);',
        'background: #f8f9fa;',
        content
    )
    
    # Add transition to body
    content = re.sub(
        r'(body\s*{[^}]*min-height: 100vh;[^}]*font-family: [^;]+;)',
        r'\1\n            transition: all 0.3s ease;',
        content
    )
    
    # Update dropdown navigation to direct links
    dropdown_pattern = r'<li class="nav-item dropdown">.*?</li>'
    new_nav_items = '''<li class="nav-item">
                        <a class="nav-link" href="{{ url_for('training') }}">
                            <i class="fas fa-cogs me-1"></i>Training
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('statistics') }}">
                            <i class="fas fa-chart-bar me-1"></i>Statistics
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('documentation') }}">
                            <i class="fas fa-book me-1"></i>Docs
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('about') }}">
                            <i class="fas fa-info-circle me-1"></i>About
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('contact') }}">
                            <i class="fas fa-envelope me-1"></i>Contact
                        </a>
                    </li>'''
    
    content = re.sub(dropdown_pattern, new_nav_items, content, flags=re.DOTALL)
    
    # Add theme toggle button
    nav_end_pattern = r'(</ul>\s*</div>\s*</div>\s*</nav>)'
    theme_button = '''</ul>
                <button class="theme-toggle" id="themeToggle" title="Toggle Theme">
                    <i class="fas fa-moon" id="themeIcon"></i>
                </button>
            </div>
        </div>
    </nav>'''
    
    content = re.sub(nav_end_pattern, theme_button, content)
    
    # Add theme manager script
    bootstrap_pattern = r'(<script src="https://cdn\.jsdelivr\.net/npm/bootstrap@5\.3\.0/dist/js/bootstrap\.bundle\.min\.js"></script>)'
    theme_script = r'\1\n    <!-- Theme Manager -->\n    <script src="{{ url_for(\'static\', filename=\'theme-manager.js\') }}"></script>'
    
    content = re.sub(bootstrap_pattern, theme_script, content)
    
    # Write updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated {file_path}")

def main():
    """Update all HTML template files."""
    templates_dir = Path('templates')
    
    # List of files to update (excluding home.html and gallery.html as they're already updated)
    files_to_update = [
        'api_explorer.html',
        'batch_processing.html', 
        'contact.html',
        'about.html',
        'documentation.html',
        'training.html',
        'statistics.html'
    ]
    
    print("🔄 Updating all HTML pages with new navigation and theme system...")
    
    for filename in files_to_update:
        file_path = templates_dir / filename
        if file_path.exists():
            try:
                update_page_navigation(file_path)
            except Exception as e:
                print(f"❌ Error updating {filename}: {e}")
        else:
            print(f"⚠️  File not found: {filename}")
    
    print("\n✅ All pages updated successfully!")
    print("🎨 Theme system is now active across all pages")
    print("📱 Navigation is now mobile-friendly without dropdowns")

if __name__ == "__main__":
    main()
