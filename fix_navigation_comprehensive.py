#!/usr/bin/env python3
"""
Comprehensive Navigation Bar Fix
Fixes all navigation issues across all templates
"""

import os
import re
from pathlib import Path

def fix_navigation_in_template(file_path):
    """Fix navigation in a single template file."""
    print(f"Fixing navigation in {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix route names to match actual Flask routes
    route_fixes = [
        # Fix batch processing route
        (r"url_for\('batch_processing'\)", "url_for('batch')"),
        # Fix API explorer route  
        (r"url_for\('api_explorer'\)", "url_for('api')"),
        # Ensure consistent route names
        (r"url_for\('batch_processing'\)", "url_for('batch')"),
    ]
    
    for pattern, replacement in route_fixes:
        content = re.sub(pattern, replacement, content)
    
    # Fix navigation structure - ensure consistent navigation across all pages
    nav_structure = '''                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('home') }}">
                            <i class="fas fa-home me-1"></i>Home
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('gallery') }}">
                            <i class="fas fa-images me-1"></i>Gallery
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('realtime') }}">
                            <i class="fas fa-video me-1"></i>Real-time
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('batch') }}">
                            <i class="fas fa-layer-group me-1"></i>Batch
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('api') }}">
                            <i class="fas fa-code me-1"></i>API
                        </a>
                    </li>
                    <li class="nav-item">
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
                    </li>
                </ul>
                <button class="theme-toggle" id="themeToggle" title="Toggle Theme">
                    <i class="fas fa-moon" id="themeIcon"></i>
                </button>'''
    
    # Replace navigation structure
    nav_pattern = r'<ul class="navbar-nav ms-auto">.*?</ul>\s*<button class="theme-toggle".*?</button>'
    if re.search(nav_pattern, content, re.DOTALL):
        content = re.sub(nav_pattern, nav_structure, content, flags=re.DOTALL)
    
    # Add enhanced navigation CSS if not present
    enhanced_nav_css = '''
        /* Enhanced Navigation Styles */
        .navbar {
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(10px);
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            padding: 0.8rem 0;
        }

        .navbar-brand {
            font-weight: 700;
            font-size: 1.4rem;
            color: #2c3e50 !important;
            transition: all 0.3s ease;
        }

        .navbar-brand:hover {
            color: #3498db !important;
            transform: scale(1.05);
        }

        .navbar-nav .nav-link {
            font-weight: 500;
            color: #2c3e50 !important;
            margin: 0 5px;
            padding: 8px 16px !important;
            border-radius: 25px;
            transition: all 0.3s ease;
            position: relative;
        }

        .navbar-nav .nav-link:hover {
            color: #3498db !important;
            background: rgba(52, 152, 219, 0.1);
            transform: translateY(-2px);
        }

        .navbar-nav .nav-link.active {
            color: #e74c3c !important;
            background: rgba(231, 76, 60, 0.1);
            font-weight: 600;
        }

        .navbar-nav .nav-link i {
            margin-right: 6px;
            transition: all 0.3s ease;
        }

        .navbar-nav .nav-link:hover i {
            transform: scale(1.2);
        }

        .navbar-toggler {
            border: none;
            padding: 4px 8px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }

        .navbar-toggler:focus {
            box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
        }

        .navbar-toggler:hover {
            background: rgba(52, 152, 219, 0.1);
        }

        /* Theme Toggle Enhanced */
        .theme-toggle {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
            border-radius: 50%;
            width: 45px;
            height: 45px;
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-left: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        .theme-toggle:hover {
            transform: scale(1.1) rotate(10deg);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }

        .theme-toggle:active {
            transform: scale(0.95);
        }

        /* Dark theme navigation */
        .dark-theme .navbar {
            background: rgba(44, 62, 80, 0.95) !important;
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.3);
        }

        .dark-theme .navbar-brand {
            color: #ecf0f1 !important;
        }

        .dark-theme .navbar-brand:hover {
            color: #3498db !important;
        }

        .dark-theme .navbar-nav .nav-link {
            color: #ecf0f1 !important;
        }

        .dark-theme .navbar-nav .nav-link:hover {
            color: #3498db !important;
            background: rgba(52, 152, 219, 0.2);
        }

        .dark-theme .navbar-nav .nav-link.active {
            color: #e74c3c !important;
            background: rgba(231, 76, 60, 0.2);
        }

        .dark-theme .theme-toggle {
            background: linear-gradient(135deg, #f39c12, #e67e22);
            box-shadow: 0 4px 15px rgba(243, 156, 18, 0.3);
        }

        .dark-theme .theme-toggle:hover {
            box-shadow: 0 6px 20px rgba(243, 156, 18, 0.4);
        }

        /* Mobile Navigation Enhancements */
        @media (max-width: 991px) {
            .navbar-collapse {
                background: rgba(255, 255, 255, 0.98);
                border-radius: 15px;
                margin-top: 15px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
            }

            .dark-theme .navbar-collapse {
                background: rgba(44, 62, 80, 0.98);
            }

            .navbar-nav {
                text-align: center;
            }

            .navbar-nav .nav-item {
                margin: 5px 0;
            }

            .navbar-nav .nav-link {
                padding: 12px 20px !important;
                margin: 2px 0;
                border-radius: 12px;
                font-size: 1.1rem;
            }

            .theme-toggle {
                margin: 15px auto 0;
                position: relative;
            }
        }

        /* Smooth scroll for navigation */
        html {
            scroll-behavior: smooth;
        }

        /* Navigation animation on load */
        .navbar {
            animation: slideDown 0.5s ease-out;
        }

        @keyframes slideDown {
            from {
                transform: translateY(-100%);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
    '''
    
    # Add enhanced CSS if not already present
    if 'Enhanced Navigation Styles' not in content:
        # Find the closing </style> tag and add before it
        style_end = content.rfind('</style>')
        if style_end != -1:
            content = content[:style_end] + enhanced_nav_css + '\n        ' + content[style_end:]
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Fixed {file_path}")
        return True
    else:
        print(f"  ⚪ No changes needed for {file_path}")
        return False

def main():
    """Fix navigation across all templates."""
    print("🔧 Comprehensive Navigation Bar Fix")
    print("=" * 50)
    
    templates_dir = Path('templates')
    html_files = list(templates_dir.glob('*.html'))
    
    fixed_count = 0
    
    for html_file in html_files:
        if fix_navigation_in_template(html_file):
            fixed_count += 1
    
    print(f"\n✅ Navigation fix completed!")
    print(f"📊 Fixed {fixed_count} out of {len(html_files)} templates")
    print("🎯 All navigation bars now have consistent styling and functionality")

if __name__ == "__main__":
    main()
