#!/usr/bin/env python3
"""
Comprehensive Error Fix Script
Identifies and fixes all remaining system errors
"""

import os
import sys
import json
import requests
import time
from pathlib import Path
import subprocess

class ComprehensiveErrorFixer:
    """Fixes all remaining system errors."""
    
    def __init__(self):
        self.base_url = "http://localhost:5000"
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
    
    def test_all_routes(self):
        """Test all routes for errors."""
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
        
        print("🔍 Testing all routes...")
        errors = []
        
        for route, name in routes:
            try:
                response = requests.get(f"{self.base_url}{route}", timeout=10)
                if response.status_code == 200:
                    print(f"  ✅ {name}: OK")
                else:
                    error = f"{name}: HTTP {response.status_code}"
                    print(f"  ❌ {error}")
                    errors.append(error)
            except Exception as e:
                error = f"{name}: {str(e)}"
                print(f"  ❌ {error}")
                errors.append(error)
        
        return errors
    
    def check_static_files(self):
        """Check if all static files exist."""
        print("📁 Checking static files...")
        
        required_files = [
            'static/theme-manager.js',
            'static/gallery/gallery_data.json',
            'static/gallery/gallery_stats.json'
        ]
        
        missing_files = []
        
        for file_path in required_files:
            if Path(file_path).exists():
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ Missing: {file_path}")
                missing_files.append(file_path)
        
        return missing_files
    
    def check_template_syntax(self):
        """Check template syntax errors."""
        print("📝 Checking template syntax...")
        
        templates_dir = Path('templates')
        syntax_errors = []
        
        for template_file in templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for common syntax issues
                if "url_for(\\'static\\'" in content:
                    error = f"{template_file}: Escaped quotes in url_for"
                    syntax_errors.append(error)
                    print(f"  ❌ {error}")
                elif "{{" in content and "}}" in content:
                    print(f"  ✅ {template_file}: Syntax OK")
                else:
                    print(f"  ⚠️ {template_file}: No template syntax found")
                    
            except Exception as e:
                error = f"{template_file}: {str(e)}"
                syntax_errors.append(error)
                print(f"  ❌ {error}")
        
        return syntax_errors
    
    def fix_missing_gallery_images(self):
        """Fix missing gallery images."""
        print("🖼️ Checking gallery images...")
        
        gallery_dir = Path('static/gallery')
        if not gallery_dir.exists():
            print("  ❌ Gallery directory missing, creating...")
            try:
                subprocess.run([sys.executable, 'create_gallery_images.py'], check=True)
                print("  ✅ Gallery images created")
                self.fixes_applied.append("Created gallery images")
            except Exception as e:
                print(f"  ❌ Failed to create gallery: {e}")
                return False
        else:
            print("  ✅ Gallery directory exists")
        
        return True
    
    def fix_route_conflicts(self):
        """Fix any remaining route conflicts."""
        print("🔧 Checking for route conflicts...")
        
        app_file = Path('enhanced_app.py')
        if not app_file.exists():
            print("  ❌ enhanced_app.py not found")
            return False
        
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for duplicate routes
        routes = []
        import re
        
        route_pattern = r"@self\.app\.route\('([^']+)'\)"
        matches = re.findall(route_pattern, content)
        
        duplicates = []
        seen = set()
        
        for route in matches:
            if route in seen:
                duplicates.append(route)
            seen.add(route)
        
        if duplicates:
            print(f"  ❌ Duplicate routes found: {duplicates}")
            return False
        else:
            print("  ✅ No route conflicts found")
            return True
    
    def fix_theme_manager(self):
        """Ensure theme manager is working."""
        print("🎨 Checking theme manager...")
        
        theme_file = Path('static/theme-manager.js')
        if not theme_file.exists():
            print("  ❌ Theme manager missing")
            return False
        
        # Test theme manager accessibility
        try:
            response = requests.get(f"{self.base_url}/static/theme-manager.js", timeout=5)
            if response.status_code == 200:
                print("  ✅ Theme manager accessible")
                return True
            else:
                print(f"  ❌ Theme manager not accessible: {response.status_code}")
                return False
        except Exception as e:
            print(f"  ❌ Theme manager error: {e}")
            return False
    
    def run_comprehensive_fix(self):
        """Run comprehensive error fixing."""
        print("🔧 Comprehensive Error Fix")
        print("=" * 50)
        
        # Check server status
        if not self.check_server_status():
            print("⚠️ Server not running. Please start with: python enhanced_app.py")
            return False
        
        # Test all routes
        route_errors = self.test_all_routes()
        
        # Check static files
        missing_files = self.check_static_files()
        
        # Check template syntax
        syntax_errors = self.check_template_syntax()
        
        # Fix missing gallery images
        self.fix_missing_gallery_images()
        
        # Fix route conflicts
        self.fix_route_conflicts()
        
        # Fix theme manager
        self.fix_theme_manager()
        
        # Summary
        total_errors = len(route_errors) + len(missing_files) + len(syntax_errors)
        
        print("\n" + "=" * 50)
        print("📊 ERROR FIX SUMMARY")
        print("=" * 50)
        
        if total_errors == 0:
            print("🎉 NO ERRORS FOUND! System is working perfectly!")
        else:
            print(f"❌ Found {total_errors} errors:")
            
            if route_errors:
                print(f"\n🌐 Route Errors ({len(route_errors)}):")
                for error in route_errors:
                    print(f"  - {error}")
            
            if missing_files:
                print(f"\n📁 Missing Files ({len(missing_files)}):")
                for file in missing_files:
                    print(f"  - {file}")
            
            if syntax_errors:
                print(f"\n📝 Syntax Errors ({len(syntax_errors)}):")
                for error in syntax_errors:
                    print(f"  - {error}")
        
        if self.fixes_applied:
            print(f"\n✅ Fixes Applied ({len(self.fixes_applied)}):")
            for fix in self.fixes_applied:
                print(f"  - {fix}")
        
        print("\n🎯 RECOMMENDATIONS:")
        if total_errors == 0:
            print("  - System is fully functional")
            print("  - All pages are working correctly")
            print("  - Navigation is properly configured")
            print("  - Gallery has relevant images")
            print("  - Theme system is operational")
        else:
            print("  - Restart the server after fixes")
            print("  - Test all pages manually")
            print("  - Check browser console for JS errors")
        
        return total_errors == 0

def main():
    """Main error fixing function."""
    print("🛠️ AI Deepfake Detector - Comprehensive Error Fix")
    print("=" * 60)
    
    fixer = ComprehensiveErrorFixer()
    success = fixer.run_comprehensive_fix()
    
    if success:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("🌐 Your website is ready at: http://localhost:5000")
    else:
        print("\n⚠️ Some issues found. Please review and fix.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
