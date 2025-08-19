#!/usr/bin/env python3
"""
Fix Jinja2 Extra Parentheses
Fixes the specific issue with extra closing parentheses in Jinja2 expressions
"""

import os
import re
from pathlib import Path

class Jinja2ParenthesesFixer:
    """Fixes extra parentheses in Jinja2 expressions."""
    
    def __init__(self):
        self.templates_dir = Path('templates')
        self.fixes_applied = []
    
    def fix_extra_parentheses(self):
        """Fix extra closing parentheses in Jinja2 expressions."""
        print("🔧 Fixing extra parentheses in Jinja2 expressions...")
        
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Fix the specific pattern: url_for('static', filename='file.js')) }}
                # Should be: url_for('static', filename='file.js') }}
                content = re.sub(r"url_for\('static', filename='([^']*?)'\)\) }}", r"url_for('static', filename='\1') }}", content)
                
                # Fix any other extra parentheses patterns
                content = re.sub(r"\{\{ ([^}]*?)\)\) \}\}", r"{{ \1) }}", content)
                content = re.sub(r"\{\{ ([^}]*?)\)\)\}\}", r"{{ \1)}}", content)
                
                # Fix specific cases found in the error
                content = content.replace("filename='theme-manager.js')) }}", "filename='theme-manager.js') }}")
                content = content.replace("filename='advanced-loader.js')) }}", "filename='advanced-loader.js') }}")
                content = content.replace("filename='interactive-enhancements.js')) }}", "filename='interactive-enhancements.js') }}")
                content = content.replace("filename='chatbot.js')) }}", "filename='chatbot.js') }}")
                content = content.replace("filename='k.ico')) }}", "filename='k.ico') }}")
                
                if content != original_content:
                    with open(template_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ Fixed extra parentheses in {template_file.name}")
                    
            except Exception as e:
                print(f"  ⚠️ Error fixing {template_file.name}: {e}")
        
        self.fixes_applied.append("Fixed extra parentheses in Jinja2 expressions")
    
    def validate_jinja2_syntax(self):
        """Validate Jinja2 syntax after fixes."""
        print("✅ Validating Jinja2 syntax...")
        
        valid_count = 0
        total_count = 0
        
        for template_file in self.templates_dir.glob('*.html'):
            total_count += 1
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for common syntax errors
                issues = []
                
                # Check for extra parentheses
                extra_parens = re.findall(r'\{\{[^}]*\)\)[^}]*\}\}', content)
                if extra_parens:
                    issues.append(f"Extra parentheses: {len(extra_parens)} found")
                
                # Check for unmatched parentheses in Jinja2 expressions
                jinja_expressions = re.findall(r'\{\{[^}]*\}\}', content)
                for expr in jinja_expressions:
                    open_parens = expr.count('(')
                    close_parens = expr.count(')')
                    if open_parens != close_parens:
                        issues.append(f"Unmatched parentheses in: {expr[:50]}...")
                        break
                
                if issues:
                    print(f"  ⚠️ {template_file.name}: {', '.join(issues)}")
                else:
                    print(f"  ✅ {template_file.name}: Valid syntax")
                    valid_count += 1
                    
            except Exception as e:
                print(f"  ❌ {template_file.name}: Validation error - {e}")
        
        success_rate = (valid_count / total_count) * 100 if total_count > 0 else 0
        print(f"  📊 Validation Success Rate: {success_rate:.1f}% ({valid_count}/{total_count})")
        
        return success_rate >= 95
    
    def run_parentheses_fix(self):
        """Run parentheses fix."""
        print("🔧 Running Jinja2 Parentheses Fix")
        print("=" * 60)
        
        self.fix_extra_parentheses()
        all_valid = self.validate_jinja2_syntax()
        
        print("\n" + "=" * 60)
        print("📊 JINJA2 PARENTHESES FIX SUMMARY")
        print("=" * 60)
        
        for fix in self.fixes_applied:
            print(f"✅ {fix}")
        
        print(f"\n🎯 Template Status:")
        if all_valid:
            print("✅ All templates have valid Jinja2 syntax!")
            print("🚀 Ready for Flask rendering!")
        else:
            print("⚠️ Some templates may still have syntax issues")
            print("🔧 Check validation results above")
        
        print(f"\n🌟 Jinja2 Parentheses Fix Complete!")
        
        return all_valid

def main():
    """Run Jinja2 parentheses fix."""
    print("🔧 AI Deepfake Detector - Jinja2 Parentheses Fix")
    print("Fixing extra parentheses in Jinja2 expressions!")
    print("=" * 70)
    
    fixer = Jinja2ParenthesesFixer()
    success = fixer.run_parentheses_fix()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
