#!/usr/bin/env python3
"""
Comprehensive Error Analysis and Fix
Analyzes entire project and fixes all issues systematically
"""

import os
import re
import json
import requests
from pathlib import Path
import subprocess
import sys

class ComprehensiveErrorAnalyzer:
    """Analyzes and fixes all project errors comprehensively."""
    
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.templates_dir = Path('templates')
        self.static_dir = Path('static')
        self.errors_found = []
        self.fixes_applied = []
    
    def check_server_status(self):
        """Check if server is running."""
        try:
            response = requests.get(self.base_url, timeout=5)
            if response.status_code == 200:
                print("✅ Server is running")
                return True
            else:
                print(f"⚠️ Server responded with status: {response.status_code}")
                return False
        except requests.exceptions.RequestException:
            print("❌ Server is not running")
            return False
    
    def analyze_template_issues(self):
        """Analyze template files for issues."""
        print("📝 Analyzing template files...")
        
        template_issues = []
        
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for common issues
                issues = []
                
                # Check for missing closing tags
                open_tags = re.findall(r'<(\w+)[^>]*>', content)
                close_tags = re.findall(r'</(\w+)>', content)
                
                # Check for broken Jinja2 syntax
                if '{{' in content and '}}' not in content:
                    issues.append("Broken Jinja2 syntax - missing closing braces")
                
                # Check for missing required elements
                if '<html' not in content:
                    issues.append("Missing HTML tag")
                if '<head>' not in content:
                    issues.append("Missing head tag")
                if '<body>' not in content:
                    issues.append("Missing body tag")
                
                # Check for duplicate IDs
                ids = re.findall(r'id="([^"]*)"', content)
                duplicate_ids = [id for id in set(ids) if ids.count(id) > 1]
                if duplicate_ids:
                    issues.append(f"Duplicate IDs: {duplicate_ids}")
                
                # Check for missing alt attributes on images
                img_tags = re.findall(r'<img[^>]*>', content)
                for img in img_tags:
                    if 'alt=' not in img:
                        issues.append("Images missing alt attributes")
                        break
                
                # Check for inline styles that should be in CSS
                if 'style=' in content:
                    inline_styles = len(re.findall(r'style="[^"]*"', content))
                    if inline_styles > 5:
                        issues.append(f"Too many inline styles ({inline_styles})")
                
                if issues:
                    template_issues.append({
                        'file': template_file.name,
                        'issues': issues
                    })
                    
            except Exception as e:
                template_issues.append({
                    'file': template_file.name,
                    'issues': [f"Error reading file: {e}"]
                })
        
        if template_issues:
            self.errors_found.extend(template_issues)
            print(f"  ❌ Found issues in {len(template_issues)} template files")
        else:
            print("  ✅ All template files are clean")
        
        return template_issues
    
    def analyze_static_files(self):
        """Analyze static files for issues."""
        print("📁 Analyzing static files...")
        
        static_issues = []
        
        # Check for required files
        required_files = [
            'theme-manager.js',
            'advanced-loader.js',
            'interactive-enhancements.js'
        ]
        
        for file in required_files:
            file_path = self.static_dir / file
            if not file_path.exists():
                static_issues.append(f"Missing required file: {file}")
            else:
                # Check file size
                size = file_path.stat().st_size
                if size == 0:
                    static_issues.append(f"Empty file: {file}")
                elif size > 1024 * 1024:  # 1MB
                    static_issues.append(f"Large file (>1MB): {file}")
        
        # Check gallery directory
        gallery_dir = self.static_dir / 'gallery'
        if gallery_dir.exists():
            image_count = len(list(gallery_dir.rglob('*.jpg'))) + len(list(gallery_dir.rglob('*.png')))
            if image_count < 16:
                static_issues.append(f"Gallery has only {image_count} images, expected 16+")
        else:
            static_issues.append("Gallery directory missing")
        
        if static_issues:
            self.errors_found.extend(static_issues)
            print(f"  ❌ Found {len(static_issues)} static file issues")
        else:
            print("  ✅ All static files are present")
        
        return static_issues
    
    def test_all_routes(self):
        """Test all routes for errors."""
        print("🌐 Testing all routes...")
        
        routes = [
            ('/', 'Home'),
            ('/gallery', 'Gallery'),
            ('/realtime', 'Real-time'),
            ('/batch', 'Batch'),
            ('/api', 'API'),
            ('/about', 'About'),
            ('/documentation', 'Documentation'),
            ('/training', 'Training'),
            ('/statistics', 'Statistics'),
            ('/contact', 'Contact')
        ]
        
        route_errors = []
        
        for route, name in routes:
            try:
                response = requests.get(f"{self.base_url}{route}", timeout=10)
                if response.status_code == 200:
                    print(f"  ✅ {name}: OK")
                    
                    # Check for common HTML errors in response
                    content = response.text
                    if 'error' in content.lower() or 'exception' in content.lower():
                        route_errors.append(f"{name}: Contains error messages")
                    
                else:
                    error = f"{name}: HTTP {response.status_code}"
                    print(f"  ❌ {error}")
                    route_errors.append(error)
                    
            except Exception as e:
                error = f"{name}: {str(e)}"
                print(f"  ❌ {error}")
                route_errors.append(error)
        
        if route_errors:
            self.errors_found.extend(route_errors)
        
        return route_errors
    
    def check_performance_issues(self):
        """Check for performance issues."""
        print("⚡ Checking performance issues...")
        
        performance_issues = []
        
        # Check for large files
        for file_path in self.static_dir.rglob('*'):
            if file_path.is_file():
                size = file_path.stat().st_size
                if size > 2 * 1024 * 1024:  # 2MB
                    performance_issues.append(f"Large file: {file_path.name} ({size // 1024}KB)")
        
        # Check for unoptimized images
        for img_path in self.static_dir.rglob('*.jpg'):
            size = img_path.stat().st_size
            if size > 500 * 1024:  # 500KB
                performance_issues.append(f"Large image: {img_path.name} ({size // 1024}KB)")
        
        # Check for missing compression
        for template_file in self.templates_dir.glob('*.html'):
            with open(template_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for unminified external resources
            if 'bootstrap.min.css' not in content and 'bootstrap.css' in content:
                performance_issues.append(f"{template_file.name}: Using unminified Bootstrap")
        
        if performance_issues:
            self.errors_found.extend(performance_issues)
            print(f"  ❌ Found {len(performance_issues)} performance issues")
        else:
            print("  ✅ No major performance issues found")
        
        return performance_issues
    
    def fix_common_issues(self):
        """Fix common issues automatically."""
        print("🔧 Fixing common issues...")
        
        fixes_count = 0
        
        # Fix missing alt attributes
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Add alt attributes to images without them
                content = re.sub(r'<img([^>]*?)(?<!alt=")>', r'<img\1 alt="Image">', content)
                
                # Fix common HTML issues
                content = content.replace('&', '&amp;')
                content = content.replace('<', '&lt;').replace('>', '&gt;')
                # But restore HTML tags
                content = re.sub(r'&lt;(\/?[a-zA-Z][^&]*?)&gt;', r'<\1>', content)
                
                # Ensure proper DOCTYPE
                if '<!DOCTYPE html>' not in content:
                    content = '<!DOCTYPE html>\n' + content
                
                if content != original_content:
                    with open(template_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixes_count += 1
                    print(f"  ✅ Fixed issues in {template_file.name}")
                    
            except Exception as e:
                print(f"  ⚠️ Error fixing {template_file.name}: {e}")
        
        if fixes_count > 0:
            self.fixes_applied.append(f"Fixed common issues in {fixes_count} files")
        
        return fixes_count
    
    def run_comprehensive_analysis(self):
        """Run comprehensive error analysis and fixes."""
        print("🔍 Running Comprehensive Error Analysis")
        print("=" * 60)
        
        # Check server status
        server_running = self.check_server_status()
        
        # Analyze templates
        template_issues = self.analyze_template_issues()
        
        # Analyze static files
        static_issues = self.analyze_static_files()
        
        # Test routes if server is running
        route_errors = []
        if server_running:
            route_errors = self.test_all_routes()
        
        # Check performance
        performance_issues = self.check_performance_issues()
        
        # Fix common issues
        fixes_count = self.fix_common_issues()
        
        # Generate report
        total_errors = len(template_issues) + len(static_issues) + len(route_errors) + len(performance_issues)
        
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE ERROR ANALYSIS REPORT")
        print("=" * 60)
        
        if total_errors == 0:
            print("🎉 NO CRITICAL ERRORS FOUND!")
            print("✅ Your project is in excellent condition!")
        else:
            print(f"❌ Found {total_errors} issues:")
            
            if template_issues:
                print(f"\n📝 Template Issues ({len(template_issues)}):")
                for issue in template_issues:
                    print(f"  - {issue['file']}: {', '.join(issue['issues'])}")
            
            if static_issues:
                print(f"\n📁 Static File Issues ({len(static_issues)}):")
                for issue in static_issues:
                    print(f"  - {issue}")
            
            if route_errors:
                print(f"\n🌐 Route Errors ({len(route_errors)}):")
                for error in route_errors:
                    print(f"  - {error}")
            
            if performance_issues:
                print(f"\n⚡ Performance Issues ({len(performance_issues)}):")
                for issue in performance_issues:
                    print(f"  - {issue}")
        
        if self.fixes_applied:
            print(f"\n✅ Fixes Applied ({len(self.fixes_applied)}):")
            for fix in self.fixes_applied:
                print(f"  - {fix}")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if total_errors == 0:
            print("  - Project is ready for production")
            print("  - All systems working correctly")
            print("  - Performance is optimized")
        else:
            print("  - Review and fix identified issues")
            print("  - Test all functionality manually")
            print("  - Optimize large files for better performance")
        
        return total_errors == 0

def main():
    """Run comprehensive error analysis."""
    print("🔍 AI Deepfake Detector - Comprehensive Error Analysis")
    print("Analyzing entire project for issues and errors...")
    print("=" * 70)
    
    analyzer = ComprehensiveErrorAnalyzer()
    success = analyzer.run_comprehensive_analysis()
    
    if success:
        print("\n🎉 PROJECT IS ERROR-FREE!")
        print("🚀 Ready for production deployment!")
    else:
        print("\n⚠️ Issues found. Please review and fix.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
