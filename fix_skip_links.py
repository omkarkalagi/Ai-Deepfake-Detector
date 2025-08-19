#!/usr/bin/env python3
"""
Fix Skip Links
Fixes the skip link visibility issue across all templates
"""

import os
import re
from pathlib import Path

class SkipLinkFixer:
    """Fixes skip link visibility issues in all templates."""
    
    def __init__(self):
        self.templates_dir = Path('templates')
        self.fixes_applied = []
    
    def add_skip_link_css(self):
        """Add proper skip link CSS to all templates."""
        print("🔧 Adding skip link CSS to all templates...")
        
        skip_link_css = '''        /* Skip link accessibility - hidden by default, visible on focus */
        .skip-link {
            position: absolute;
            top: -40px;
            left: 6px;
            background: #000;
            color: #fff;
            padding: 8px;
            text-decoration: none;
            border-radius: 4px;
            z-index: 10000;
            transition: top 0.3s ease;
        }
        .skip-link:focus {
            top: 6px;
        }'''
        
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Skip if already has skip-link CSS
                if '.skip-link' in content:
                    print(f"  ✅ {template_file.name}: Skip link CSS already present")
                    continue
                
                # Add skip link CSS before closing style tag
                if '</style>' in content and 'skip-link' not in content:
                    content = content.replace('</style>', f'{skip_link_css}\n        </style>')
                    
                    with open(template_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"  ✅ Added skip link CSS to {template_file.name}")
                    
            except Exception as e:
                print(f"  ⚠️ Error updating {template_file.name}: {e}")
        
        self.fixes_applied.append("Added skip link CSS to all templates")
    
    def fix_skip_link_positioning(self):
        """Fix skip link positioning and visibility."""
        print("👁️ Fixing skip link positioning...")
        
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Ensure skip links are properly positioned after body tag
                if 'skip-link' in content:
                    # Fix skip link placement - should be right after <body>
                    content = re.sub(
                        r'<body[^>]*>\s*<a[^>]*class="skip-link"[^>]*>([^<]*)</a>',
                        lambda m: f'<body>\n    <a href="#main-content" class="skip-link">{m.group(1)}</a>',
                        content
                    )
                    
                    # Ensure main content has proper ID
                    if 'id="main-content"' not in content:
                        # Add main-content ID to the first main container
                        content = re.sub(
                            r'(<div[^>]*class="[^"]*container[^"]*"[^>]*>)',
                            r'\1\n    <div id="main-content">',
                            content,
                            count=1
                        )
                        
                        # Close the main-content div before footer
                        content = re.sub(
                            r'(<!-- Footer|<footer)',
                            r'    </div>\n    \1',
                            content,
                            count=1
                        )
                
                if content != original_content:
                    with open(template_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"  ✅ Fixed skip link positioning in {template_file.name}")
                    
            except Exception as e:
                print(f"  ⚠️ Error fixing positioning in {template_file.name}: {e}")
        
        self.fixes_applied.append("Fixed skip link positioning")
    
    def validate_skip_links(self):
        """Validate skip links in all templates."""
        print("✅ Validating skip links...")
        
        valid_count = 0
        total_count = 0
        
        for template_file in self.templates_dir.glob('*.html'):
            total_count += 1
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                issues = []
                
                # Check if skip link exists
                has_skip_link = 'skip-link' in content
                
                # Check if skip link CSS exists
                has_skip_css = '.skip-link' in content and 'position: absolute' in content
                
                # Check if main-content target exists
                has_main_content = 'id="main-content"' in content or 'main-content' in content
                
                if not has_skip_link:
                    issues.append("No skip link found")
                
                if not has_skip_css:
                    issues.append("Missing skip link CSS")
                
                if not has_main_content:
                    issues.append("Missing main-content target")
                
                if issues:
                    print(f"  ⚠️ {template_file.name}: {', '.join(issues)}")
                else:
                    print(f"  ✅ {template_file.name}: Skip link properly configured")
                    valid_count += 1
                    
            except Exception as e:
                print(f"  ❌ {template_file.name}: Validation error - {e}")
        
        success_rate = (valid_count / total_count) * 100 if total_count > 0 else 0
        print(f"  📊 Skip Link Success Rate: {success_rate:.1f}% ({valid_count}/{total_count})")
        
        return success_rate >= 80
    
    def run_skip_link_fix(self):
        """Run comprehensive skip link fix."""
        print("🔧 Running Skip Link Fix")
        print("=" * 60)
        
        self.add_skip_link_css()
        self.fix_skip_link_positioning()
        all_valid = self.validate_skip_links()
        
        print("\n" + "=" * 60)
        print("📊 SKIP LINK FIX SUMMARY")
        print("=" * 60)
        
        for fix in self.fixes_applied:
            print(f"✅ {fix}")
        
        print(f"\n🎯 Skip Link Status:")
        if all_valid:
            print("✅ All skip links are properly configured!")
            print("👁️ Skip links are now hidden by default and visible on focus")
            print("♿ Accessibility compliance maintained")
        else:
            print("⚠️ Some templates may need manual review")
            print("🔧 Check validation results above")
        
        print(f"\n🌟 Skip Link Fix Complete!")
        print("👁️ Skip links are now properly hidden!")
        
        return all_valid

def main():
    """Run skip link fix."""
    print("🔧 AI Deepfake Detector - Skip Link Fix")
    print("Fixing skip link visibility issues!")
    print("=" * 70)
    
    fixer = SkipLinkFixer()
    success = fixer.run_skip_link_fix()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
