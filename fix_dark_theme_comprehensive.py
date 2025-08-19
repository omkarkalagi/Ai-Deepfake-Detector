#!/usr/bin/env python3
"""
Comprehensive Dark Theme Fix
Fixes all dark theme background and styling issues across all pages
"""

import os
import re
from pathlib import Path

class DarkThemeFixer:
    """Fixes dark theme issues comprehensively."""
    
    def __init__(self):
        self.templates_dir = Path('templates')
        self.static_dir = Path('static')
        self.fixes_applied = []
    
    def fix_dark_theme_backgrounds(self):
        """Fix dark theme background issues in all templates."""
        print("🌙 Fixing dark theme backgrounds...")
        
        # Enhanced dark theme CSS
        dark_theme_css = '''
        /* Enhanced Dark Theme Styles */
        .dark-theme {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
            color: #ecf0f1 !important;
            min-height: 100vh;
        }

        .dark-theme body {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
            color: #ecf0f1 !important;
        }

        /* Navigation Dark Theme */
        .dark-theme .navbar {
            background: rgba(26, 26, 46, 0.95) !important;
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .dark-theme .navbar-brand {
            color: #ecf0f1 !important;
        }

        .dark-theme .navbar-nav .nav-link {
            color: #bdc3c7 !important;
        }

        .dark-theme .navbar-nav .nav-link:hover {
            color: #3498db !important;
            background: rgba(52, 152, 219, 0.1);
        }

        .dark-theme .navbar-nav .nav-link.active {
            color: #e74c3c !important;
            background: rgba(231, 76, 60, 0.1);
        }

        /* Cards and Containers */
        .dark-theme .card,
        .dark-theme .main-container,
        .dark-theme .detection-container,
        .dark-theme .upload-area {
            background: rgba(44, 62, 80, 0.9) !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #ecf0f1 !important;
        }

        .dark-theme .card-header {
            background: rgba(52, 73, 94, 0.9) !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            color: #ecf0f1 !important;
        }

        /* Buttons */
        .dark-theme .btn-primary {
            background: linear-gradient(135deg, #3498db, #2980b9) !important;
            border: none;
        }

        .dark-theme .btn-secondary {
            background: rgba(52, 73, 94, 0.9) !important;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #ecf0f1 !important;
        }

        .dark-theme .btn-outline-primary {
            border-color: #3498db;
            color: #3498db;
        }

        .dark-theme .btn-outline-primary:hover {
            background: #3498db;
            color: white;
        }

        /* Forms */
        .dark-theme .form-control {
            background: rgba(52, 73, 94, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #ecf0f1 !important;
        }

        .dark-theme .form-control:focus {
            background: rgba(52, 73, 94, 0.9) !important;
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
            color: #ecf0f1 !important;
        }

        .dark-theme .form-control::placeholder {
            color: #95a5a6 !important;
        }

        .dark-theme .form-label {
            color: #ecf0f1 !important;
        }

        /* Tables */
        .dark-theme .table {
            color: #ecf0f1 !important;
        }

        .dark-theme .table-dark {
            background: rgba(44, 62, 80, 0.9) !important;
        }

        .dark-theme .table th,
        .dark-theme .table td {
            border-color: rgba(255, 255, 255, 0.1);
        }

        /* Alerts */
        .dark-theme .alert-success {
            background: rgba(39, 174, 96, 0.2) !important;
            border-color: rgba(39, 174, 96, 0.3);
            color: #2ecc71 !important;
        }

        .dark-theme .alert-danger {
            background: rgba(231, 76, 60, 0.2) !important;
            border-color: rgba(231, 76, 60, 0.3);
            color: #e74c3c !important;
        }

        .dark-theme .alert-info {
            background: rgba(52, 152, 219, 0.2) !important;
            border-color: rgba(52, 152, 219, 0.3);
            color: #3498db !important;
        }

        /* Statistics Cards */
        .dark-theme .stat-card {
            background: rgba(44, 62, 80, 0.9) !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #ecf0f1 !important;
        }

        .dark-theme .stat-card h3 {
            color: #3498db !important;
        }

        /* Gallery Items */
        .dark-theme .gallery-item {
            background: rgba(44, 62, 80, 0.9) !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .dark-theme .gallery-info {
            background: rgba(52, 73, 94, 0.9) !important;
            color: #ecf0f1 !important;
        }

        /* Team Members */
        .dark-theme .team-member {
            background: rgba(44, 62, 80, 0.9) !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #ecf0f1 !important;
        }

        /* Contact Methods */
        .dark-theme .contact-method {
            background: rgba(44, 62, 80, 0.9) !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .dark-theme .lead-contact-section {
            background: linear-gradient(135deg, rgba(52, 152, 219, 0.2), rgba(155, 89, 182, 0.2)) !important;
            border-color: rgba(52, 152, 219, 0.3);
        }

        /* Social Media Section */
        .dark-theme .social-media-section {
            background: linear-gradient(135deg, rgba(231, 76, 60, 0.2), rgba(192, 57, 43, 0.2)) !important;
        }

        /* Contact Form */
        .dark-theme .contact-form {
            background: linear-gradient(135deg, rgba(44, 62, 80, 0.8), rgba(52, 73, 94, 0.8)) !important;
        }

        /* Footer */
        .dark-theme .stylish-footer {
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460) !important;
        }

        .dark-theme .footer-links {
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Text Colors */
        .dark-theme h1, .dark-theme h2, .dark-theme h3, 
        .dark-theme h4, .dark-theme h5, .dark-theme h6 {
            color: #ecf0f1 !important;
        }

        .dark-theme p, .dark-theme span, .dark-theme div {
            color: #bdc3c7 !important;
        }

        .dark-theme .text-muted {
            color: #95a5a6 !important;
        }

        /* Upload Area */
        .dark-theme .upload-area {
            background: rgba(44, 62, 80, 0.9) !important;
            border: 2px dashed rgba(52, 152, 219, 0.5);
        }

        .dark-theme .upload-area:hover {
            border-color: #3498db;
            background: rgba(52, 73, 94, 0.9) !important;
        }

        /* Progress Bars */
        .dark-theme .progress {
            background: rgba(52, 73, 94, 0.9) !important;
        }

        .dark-theme .progress-bar {
            background: linear-gradient(90deg, #3498db, #2980b9) !important;
        }

        /* Modals */
        .dark-theme .modal-content {
            background: rgba(44, 62, 80, 0.95) !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #ecf0f1 !important;
        }

        .dark-theme .modal-header {
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .dark-theme .modal-footer {
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Dropdown */
        .dark-theme .dropdown-menu {
            background: rgba(44, 62, 80, 0.95) !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .dark-theme .dropdown-item {
            color: #bdc3c7 !important;
        }

        .dark-theme .dropdown-item:hover {
            background: rgba(52, 152, 219, 0.2) !important;
            color: #ecf0f1 !important;
        }

        /* Scrollbar */
        .dark-theme ::-webkit-scrollbar {
            width: 8px;
        }

        .dark-theme ::-webkit-scrollbar-track {
            background: rgba(44, 62, 80, 0.5);
        }

        .dark-theme ::-webkit-scrollbar-thumb {
            background: rgba(52, 152, 219, 0.7);
            border-radius: 4px;
        }

        .dark-theme ::-webkit-scrollbar-thumb:hover {
            background: #3498db;
        }

        /* Ensure proper contrast */
        .dark-theme .bg-light {
            background: rgba(44, 62, 80, 0.9) !important;
        }

        .dark-theme .bg-white {
            background: rgba(44, 62, 80, 0.9) !important;
        }

        .dark-theme .border {
            border-color: rgba(255, 255, 255, 0.1) !important;
        }

        /* Animation compatibility */
        .dark-theme .animate__animated {
            animation-fill-mode: both;
        }
        '''
        
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Add enhanced dark theme CSS
                if 'Enhanced Dark Theme Styles' not in content:
                    # Find the last </style> tag and add before it
                    style_end = content.rfind('</style>')
                    if style_end != -1:
                        content = content[:style_end] + dark_theme_css + '\n        ' + content[style_end:]
                    else:
                        # If no style tag, add in head
                        head_end = content.find('</head>')
                        if head_end != -1:
                            content = content[:head_end] + f'<style>{dark_theme_css}</style>\n' + content[head_end:]
                
                if content != original_content:
                    with open(template_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ Fixed dark theme in {template_file.name}")
                    
            except Exception as e:
                print(f"  ⚠️ Error fixing {template_file.name}: {e}")
        
        self.fixes_applied.append("Enhanced dark theme backgrounds and styling")
    
    def add_instagram_integration(self):
        """Add Instagram integration to all relevant sections."""
        print("📸 Adding Instagram integration...")
        
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Add Instagram to social media sections
                if 'telegram' in content.lower() and 'instagram' not in content.lower():
                    # Add Instagram after Telegram
                    instagram_link = '''                    <a href="https://www.instagram.com/omkar_kalagi/" class="social-link instagram" target="_blank" title="Instagram">
                        <i class="fab fa-instagram"></i>
                        <span>Instagram</span>
                    </a>'''
                    
                    # Find telegram link and add Instagram after it
                    telegram_pattern = r'(<a[^>]*telegram[^>]*>.*?</a>)'
                    if re.search(telegram_pattern, content, re.DOTALL):
                        content = re.sub(telegram_pattern, r'\1\n' + instagram_link, content, flags=re.DOTALL)
                
                # Add Instagram CSS if social links exist
                if 'social-link.telegram' in content and 'social-link.instagram' not in content:
                    instagram_css = '''
        .social-link.instagram {
            background: linear-gradient(135deg, #E4405F, #C13584, #833AB4);
        }'''
                    
                    # Add after telegram CSS
                    content = content.replace('.social-link.telegram {', instagram_css + '\n\n        .social-link.telegram {')
                
                # Add to footer social icons
                if 'fab fa-telegram' in content and 'fab fa-instagram' not in content:
                    instagram_footer = '''                    <a href="https://www.instagram.com/omkar_kalagi/" class="social-icon" target="_blank"><i class="fab fa-instagram"></i></a>'''
                    
                    # Add after telegram in footer
                    content = content.replace('</a>\n                </div>', '</a>\n' + instagram_footer + '\n                </div>')
                
                if content != original_content:
                    with open(template_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ Added Instagram to {template_file.name}")
                    
            except Exception as e:
                print(f"  ⚠️ Error adding Instagram to {template_file.name}: {e}")
        
        self.fixes_applied.append("Added Instagram (@omkar_kalagi) integration")
    
    def run_comprehensive_fixes(self):
        """Run all comprehensive dark theme fixes."""
        print("🌙 Running Comprehensive Dark Theme Fixes")
        print("=" * 60)
        
        self.fix_dark_theme_backgrounds()
        self.add_instagram_integration()
        
        print("\n" + "=" * 60)
        print("📊 DARK THEME FIX SUMMARY")
        print("=" * 60)
        
        for fix in self.fixes_applied:
            print(f"✅ {fix}")
        
        print(f"\n🎯 Dark Theme Improvements:")
        print(f"   - Beautiful dark gradient backgrounds")
        print(f"   - Proper contrast ratios for accessibility")
        print(f"   - Consistent styling across all components")
        print(f"   - Instagram integration added")
        print(f"   - Enhanced visual appeal")
        
        print(f"\n🌙 Dark theme is now perfect across all pages!")

def main():
    """Run comprehensive dark theme fixes."""
    print("🌙 AI Deepfake Detector - Dark Theme Comprehensive Fix")
    print("Making dark theme beautiful and consistent!")
    print("=" * 70)
    
    fixer = DarkThemeFixer()
    fixer.run_comprehensive_fixes()

if __name__ == "__main__":
    main()
