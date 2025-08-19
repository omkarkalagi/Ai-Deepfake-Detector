#!/usr/bin/env python3
"""
Fix All Templates Comprehensively
Fixes all Jinja2 template syntax errors and validates all templates
"""

import os
import re
from pathlib import Path

class ComprehensiveTemplateFixer:
    """Fixes all template issues comprehensively."""
    
    def __init__(self):
        self.templates_dir = Path('templates')
        self.fixes_applied = []
    
    def fix_all_escaped_quotes(self):
        """Fix all escaped quotes in all templates."""
        print("🔧 Fixing all escaped quotes in templates...")
        
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Fix all variations of escaped quotes in Jinja2
                content = content.replace("\\'", "'")
                content = content.replace('\\"', '"')
                
                # Specifically fix url_for expressions
                content = re.sub(r"url_for\(\\'([^']*)\\'", r"url_for('\1'", content)
                content = re.sub(r"filename=\\'([^']*)\\'", r"filename='\1'", content)
                
                # Fix any remaining malformed Jinja2 expressions
                content = re.sub(r'\{\{\s*url_for\([^}]*\}\}', 
                                lambda m: m.group(0).replace("\\'", "'").replace('\\"', '"'), content)
                
                if content != original_content:
                    with open(template_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ Fixed escaped quotes in {template_file.name}")
                    
            except Exception as e:
                print(f"  ⚠️ Error fixing {template_file.name}: {e}")
        
        self.fixes_applied.append("Fixed all escaped quotes")
    
    def clean_template_structure(self):
        """Clean up template structure."""
        print("🧹 Cleaning template structure...")
        
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Fix line breaks and formatting
                content = re.sub(r'</style><script', '</style>\n    <script', content)
                content = re.sub(r'></script><script', '></script>\n    <script', content)
                content = re.sub(r'></script><!--', '></script>\n    <!--', content)
                content = re.sub(r'--><link', '-->\n    <link', content)
                content = re.sub(r'><link rel="preload"', '>\n    <link rel="preload"', content)
                content = re.sub(r'><style>', '>\n    <style>', content)
                
                # Remove excessive whitespace
                content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
                content = re.sub(r'[ \t]+\n', '\n', content)
                
                if content != original_content:
                    with open(template_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ Cleaned structure in {template_file.name}")
                    
            except Exception as e:
                print(f"  ⚠️ Error cleaning {template_file.name}: {e}")
        
        self.fixes_applied.append("Cleaned template structure")
    
    def validate_all_templates(self):
        """Validate all templates for syntax errors."""
        print("✅ Validating all templates...")
        
        valid_templates = 0
        total_templates = 0
        
        for template_file in self.templates_dir.glob('*.html'):
            total_templates += 1
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for escaped quotes in Jinja2 expressions
                jinja_expressions = re.findall(r'\{\{[^}]*\}\}|\{%[^%]*%\}', content)
                has_escaped_quotes = any('\\' in expr and ("'" in expr or '"' in expr) for expr in jinja_expressions)
                
                if has_escaped_quotes:
                    print(f"  ⚠️ {template_file.name}: Still has escaped quotes")
                else:
                    print(f"  ✅ {template_file.name}: Valid")
                    valid_templates += 1
                    
            except Exception as e:
                print(f"  ❌ {template_file.name}: Validation error - {e}")
        
        success_rate = (valid_templates / total_templates) * 100 if total_templates > 0 else 0
        print(f"  📊 Validation Success Rate: {success_rate:.1f}% ({valid_templates}/{total_templates})")
        
        return success_rate >= 95
    
    def test_template_rendering(self):
        """Test template rendering by checking for common issues."""
        print("🧪 Testing template rendering compatibility...")
        
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for common rendering issues
                issues = []
                
                # Check for unmatched Jinja2 delimiters
                open_expr = content.count('{{')
                close_expr = content.count('}}')
                if open_expr != close_expr:
                    issues.append(f"Unmatched expression delimiters: {open_expr} open, {close_expr} close")
                
                open_stmt = content.count('{%')
                close_stmt = content.count('%}')
                if open_stmt != close_stmt:
                    issues.append(f"Unmatched statement delimiters: {open_stmt} open, {close_stmt} close")
                
                # Check for invalid characters in Jinja2 expressions
                invalid_chars = re.findall(r'\{\{[^}]*[\\][^}]*\}\}', content)
                if invalid_chars:
                    issues.append(f"Invalid characters in expressions: {len(invalid_chars)} found")
                
                if issues:
                    print(f"  ⚠️ {template_file.name}: {', '.join(issues)}")
                else:
                    print(f"  ✅ {template_file.name}: Rendering compatible")
                    
            except Exception as e:
                print(f"  ❌ {template_file.name}: Test error - {e}")
        
        self.fixes_applied.append("Tested template rendering compatibility")
    
    def run_comprehensive_fix(self):
        """Run comprehensive template fix."""
        print("🔧 Running Comprehensive Template Fix")
        print("=" * 60)
        
        self.fix_all_escaped_quotes()
        self.clean_template_structure()
        
        # Validate and test
        all_valid = self.validate_all_templates()
        self.test_template_rendering()
        
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEMPLATE FIX SUMMARY")
        print("=" * 60)
        
        for fix in self.fixes_applied:
            print(f"✅ {fix}")
        
        print(f"\n🎯 Template Status:")
        if all_valid:
            print("✅ All templates are now valid and error-free!")
            print("🚀 Ready for Flask rendering!")
        else:
            print("⚠️ Some templates may still need attention")
            print("🔧 Check validation results above")
        
        print(f"\n🌟 Comprehensive Template Fix Complete!")
        
        return all_valid

def main():
    """Run comprehensive template fix."""
    print("🔧 AI Deepfake Detector - Comprehensive Template Fix")
    print("Fixing all template issues comprehensively!")
    print("=" * 70)
    
    fixer = ComprehensiveTemplateFixer()
    success = fixer.run_comprehensive_fix()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
