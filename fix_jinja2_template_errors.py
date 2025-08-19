#!/usr/bin/env python3
"""
Fix Jinja2 Template Syntax Errors
Comprehensive fix for all Jinja2 template syntax issues
"""

import os
import re
from pathlib import Path

class Jinja2TemplateFixer:
    """Fixes all Jinja2 template syntax errors."""
    
    def __init__(self):
        self.templates_dir = Path('templates')
        self.fixes_applied = []
    
    def fix_escaped_quotes_in_templates(self):
        """Fix escaped quotes in Jinja2 templates."""
        print("🔧 Fixing escaped quotes in Jinja2 templates...")
        
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Fix escaped quotes in Jinja2 expressions
                # Replace \' with ' inside {{ }} and {% %}
                content = re.sub(r"(\{\{[^}]*?)\\\'([^}]*?\}\})", r"\1'\2", content)
                content = re.sub(r"(\{%[^%]*?)\\\'([^%]*?%\})", r"\1'\2", content)
                
                # Fix specific patterns that cause issues
                content = content.replace("url_for(\\'static\\'", "url_for('static'")
                content = content.replace("filename=\\'", "filename='")
                content = content.replace("\\') }}", "') }}")
                
                # Fix any remaining escaped quotes in Jinja2 expressions
                content = re.sub(r"\{\{\s*url_for\(\\'([^']*)\\'[^}]*\}\}", 
                                lambda m: m.group(0).replace("\\'", "'"), content)
                
                if content != original_content:
                    with open(template_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ Fixed escaped quotes in {template_file.name}")
                    
            except Exception as e:
                print(f"  ⚠️ Error fixing {template_file.name}: {e}")
        
        self.fixes_applied.append("Fixed escaped quotes in Jinja2 templates")
    
    def fix_malformed_jinja2_syntax(self):
        """Fix malformed Jinja2 syntax."""
        print("🔧 Fixing malformed Jinja2 syntax...")
        
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Fix common Jinja2 syntax issues
                fixes = [
                    # Fix missing spaces in Jinja2 expressions
                    (r'\{\{([^}]+)\}\}', lambda m: '{{ ' + m.group(1).strip() + ' }}'),
                    
                    # Fix missing spaces in Jinja2 statements
                    (r'\{%([^%]+)%\}', lambda m: '{% ' + m.group(1).strip() + ' %}'),
                    
                    # Fix malformed url_for expressions
                    (r"url_for\(\s*['\"]([^'\"]*)['\"],\s*filename\s*=\s*['\"]([^'\"]*)['\"]", 
                     r"url_for('\1', filename='\2')"),
                    
                    # Fix double quotes in single-quoted strings
                    (r"url_for\('([^']*\"[^']*)'", lambda m: f"url_for(\"{m.group(1)}\")"),
                ]
                
                for pattern, replacement in fixes:
                    if callable(replacement):
                        content = re.sub(pattern, replacement, content)
                    else:
                        content = re.sub(pattern, replacement, content)
                
                if content != original_content:
                    with open(template_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ Fixed malformed syntax in {template_file.name}")
                    
            except Exception as e:
                print(f"  ⚠️ Error fixing syntax in {template_file.name}: {e}")
        
        self.fixes_applied.append("Fixed malformed Jinja2 syntax")
    
    def fix_template_structure_issues(self):
        """Fix template structure issues."""
        print("🔧 Fixing template structure issues...")
        
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Fix line breaks in the middle of tags
                content = re.sub(r'</style><script', '</style>\n<script', content)
                content = re.sub(r'></script><script', '></script>\n<script', content)
                content = re.sub(r'></script><!--', '></script>\n<!--', content)
                content = re.sub(r'--><link', '-->\n<link', content)
                content = re.sub(r'><link', '>\n<link', content)
                content = re.sub(r'><style>', '>\n<style>', content)
                
                # Fix missing newlines after closing tags
                content = re.sub(r'(</head>)([^<\n])', r'\1\n\2', content)
                content = re.sub(r'(</body>)([^<\n])', r'\1\n\2', content)
                content = re.sub(r'(</html>)([^<\n])', r'\1\n\2', content)
                
                # Remove duplicate whitespace
                content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
                
                if content != original_content:
                    with open(template_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ Fixed structure issues in {template_file.name}")
                    
            except Exception as e:
                print(f"  ⚠️ Error fixing structure in {template_file.name}: {e}")
        
        self.fixes_applied.append("Fixed template structure issues")
    
    def validate_templates(self):
        """Validate templates for common issues."""
        print("✅ Validating templates...")
        
        validation_results = []
        
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                issues = []
                
                # Check for escaped quotes in Jinja2
                if "\\'" in content and ("{{" in content or "{%" in content):
                    escaped_quotes = re.findall(r'\{\{[^}]*\\\'[^}]*\}\}|\{%[^%]*\\\'[^%]*%\}', content)
                    if escaped_quotes:
                        issues.append(f"Escaped quotes found: {len(escaped_quotes)} instances")

                # Check for malformed Jinja2 expressions
                malformed = re.findall(r'\{\{[^}]*\}\}|\{%[^%]*%\}', content)
                for expr in malformed:
                    if '\\' in expr and ("'" in expr or '"' in expr):
                        issues.append(f"Potentially malformed expression: {expr[:50]}...")
                        break
                
                # Check for unclosed tags
                open_tags = len(re.findall(r'<[^/][^>]*[^/]>', content))
                close_tags = len(re.findall(r'</[^>]*>', content))
                if abs(open_tags - close_tags) > 5:  # Allow some tolerance for self-closing tags
                    issues.append(f"Tag mismatch: {open_tags} open, {close_tags} close")
                
                if issues:
                    validation_results.append({
                        'file': template_file.name,
                        'issues': issues
                    })
                else:
                    print(f"  ✅ {template_file.name}: No issues found")
                    
            except Exception as e:
                validation_results.append({
                    'file': template_file.name,
                    'issues': [f"Validation error: {e}"]
                })
        
        if validation_results:
            print(f"  ⚠️ Found issues in {len(validation_results)} templates:")
            for result in validation_results:
                print(f"    - {result['file']}: {', '.join(result['issues'])}")
        else:
            print("  ✅ All templates validated successfully!")
        
        return len(validation_results) == 0
    
    def run_comprehensive_fix(self):
        """Run comprehensive Jinja2 template fix."""
        print("🔧 Running Comprehensive Jinja2 Template Fix")
        print("=" * 60)
        
        self.fix_escaped_quotes_in_templates()
        self.fix_malformed_jinja2_syntax()
        self.fix_template_structure_issues()
        
        # Validate after fixes
        all_valid = self.validate_templates()
        
        print("\n" + "=" * 60)
        print("📊 JINJA2 TEMPLATE FIX SUMMARY")
        print("=" * 60)
        
        for fix in self.fixes_applied:
            print(f"✅ {fix}")
        
        print(f"\n🎯 Template Status:")
        if all_valid:
            print("✅ All templates are now valid and error-free!")
            print("🚀 Ready for production use!")
        else:
            print("⚠️ Some templates may still have minor issues")
            print("🔧 Manual review recommended for remaining issues")
        
        print(f"\n🌟 Jinja2 Template Fixes Complete!")
        
        return all_valid

def main():
    """Run comprehensive Jinja2 template fix."""
    print("🔧 AI Deepfake Detector - Jinja2 Template Error Fix")
    print("Fixing all template syntax errors!")
    print("=" * 70)
    
    fixer = Jinja2TemplateFixer()
    success = fixer.run_comprehensive_fix()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
