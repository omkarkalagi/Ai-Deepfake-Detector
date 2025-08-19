#!/usr/bin/env python3
"""
Comprehensive Component Tester
Tests all components, pages, and features to ensure everything works perfectly
"""

import os
import re
import json
import requests
import time
from pathlib import Path

class ComprehensiveComponentTester:
    """Tests all components and features comprehensively."""
    
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.test_results = []
        self.passed_tests = 0
        self.total_tests = 0
    
    def test_server_connectivity(self):
        """Test server connectivity and basic response."""
        print("🌐 Testing server connectivity...")
        
        try:
            response = requests.get(self.base_url, timeout=10)
            if response.status_code == 200:
                print("  ✅ Server is responding correctly")
                self.passed_tests += 1
            else:
                print(f"  ❌ Server responded with status: {response.status_code}")
            self.total_tests += 1
            
            # Test response time
            start_time = time.time()
            response = requests.get(self.base_url, timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            if response_time < 3000:  # Less than 3 seconds
                print(f"  ✅ Response time: {response_time:.0f}ms (Good)")
                self.passed_tests += 1
            else:
                print(f"  ⚠️ Response time: {response_time:.0f}ms (Slow)")
            self.total_tests += 1
            
        except Exception as e:
            print(f"  ❌ Server connectivity error: {e}")
            self.total_tests += 1
    
    def test_all_routes(self):
        """Test all application routes."""
        print("🔗 Testing all routes...")
        
        routes = [
            ('/', 'Home Page'),
            ('/gallery', 'Gallery Page'),
            ('/realtime', 'Real-time Detection'),
            ('/batch', 'Batch Processing'),
            ('/api', 'API Explorer'),
            ('/about', 'About Page'),
            ('/documentation', 'Documentation'),
            ('/training', 'Training Interface'),
            ('/statistics', 'Statistics Dashboard'),
            ('/contact', 'Contact Page')
        ]
        
        for route, name in routes:
            try:
                response = requests.get(f"{self.base_url}{route}", timeout=10)
                if response.status_code == 200:
                    print(f"  ✅ {name}: OK")
                    
                    # Check for template rendering errors
                    content = response.text
                    if 'TemplateSyntaxError' in content or 'Jinja2' in content:
                        print(f"    ⚠️ Template syntax issues detected")
                    elif 'error' in content.lower() and 'exception' in content.lower():
                        print(f"    ⚠️ Runtime errors detected")
                    else:
                        self.passed_tests += 1
                else:
                    print(f"  ❌ {name}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ {name}: {str(e)}")
            
            self.total_tests += 1
    
    def test_static_files(self):
        """Test static file accessibility."""
        print("📁 Testing static files...")
        
        static_files = [
            '/static/theme-manager.js',
            '/static/advanced-loader.js',
            '/static/interactive-enhancements.js',
            '/static/chatbot.js',
            '/static/k.ico'
        ]
        
        for file_path in static_files:
            try:
                response = requests.get(f"{self.base_url}{file_path}", timeout=5)
                if response.status_code == 200:
                    print(f"  ✅ {file_path}: Available")
                    self.passed_tests += 1
                else:
                    print(f"  ❌ {file_path}: HTTP {response.status_code}")
            except Exception as e:
                print(f"  ❌ {file_path}: {str(e)}")
            
            self.total_tests += 1
    
    def test_template_rendering(self):
        """Test template rendering for common issues."""
        print("📝 Testing template rendering...")
        
        test_pages = ['/', '/contact', '/about', '/gallery']
        
        for page in test_pages:
            try:
                response = requests.get(f"{self.base_url}{page}", timeout=10)
                if response.status_code == 200:
                    content = response.text
                    
                    # Check for essential elements
                    checks = [
                        ('DOCTYPE', '<!DOCTYPE html>' in content),
                        ('Title', '<title>' in content),
                        ('Bootstrap CSS', 'bootstrap' in content),
                        ('Font Awesome', 'font-awesome' in content or 'fontawesome' in content),
                        ('Navigation', 'navbar' in content),
                        ('Footer', 'footer' in content or 'stylish-footer' in content)
                    ]
                    
                    page_passed = 0
                    for check_name, check_result in checks:
                        if check_result:
                            page_passed += 1
                        else:
                            print(f"    ⚠️ {page} missing: {check_name}")
                    
                    if page_passed >= 5:  # At least 5 out of 6 checks
                        print(f"  ✅ {page}: Template rendering OK ({page_passed}/6)")
                        self.passed_tests += 1
                    else:
                        print(f"  ❌ {page}: Template issues ({page_passed}/6)")
                        
            except Exception as e:
                print(f"  ❌ {page}: Template test error - {e}")
            
            self.total_tests += 1
    
    def test_javascript_functionality(self):
        """Test JavaScript functionality by checking for script tags."""
        print("⚡ Testing JavaScript functionality...")
        
        try:
            response = requests.get(self.base_url, timeout=10)
            if response.status_code == 200:
                content = response.text
                
                js_checks = [
                    ('Theme Manager', 'theme-manager.js' in content),
                    ('Advanced Loader', 'advanced-loader.js' in content),
                    ('Interactive Enhancements', 'interactive-enhancements.js' in content),
                    ('Chatbot', 'chatbot.js' in content),
                    ('Bootstrap JS', 'bootstrap.bundle.min.js' in content)
                ]
                
                js_passed = 0
                for check_name, check_result in js_checks:
                    if check_result:
                        print(f"  ✅ {check_name}: Script included")
                        js_passed += 1
                    else:
                        print(f"  ❌ {check_name}: Script missing")
                
                if js_passed >= 4:  # At least 4 out of 5 scripts
                    self.passed_tests += 1
                
                self.total_tests += 1
                
        except Exception as e:
            print(f"  ❌ JavaScript test error: {e}")
            self.total_tests += 1
    
    def test_css_loading(self):
        """Test CSS loading and styling."""
        print("🎨 Testing CSS loading...")
        
        try:
            response = requests.get(self.base_url, timeout=10)
            if response.status_code == 200:
                content = response.text
                
                css_checks = [
                    ('Bootstrap CSS', 'bootstrap.min.css' in content),
                    ('Font Awesome', 'font-awesome' in content or 'fontawesome' in content),
                    ('Google Fonts', 'fonts.googleapis.com' in content),
                    ('Animate CSS', 'animate.css' in content or 'animate.min.css' in content),
                    ('Custom Styles', '<style>' in content)
                ]
                
                css_passed = 0
                for check_name, check_result in css_checks:
                    if check_result:
                        print(f"  ✅ {check_name}: Loaded")
                        css_passed += 1
                    else:
                        print(f"  ❌ {check_name}: Missing")
                
                if css_passed >= 4:  # At least 4 out of 5 CSS resources
                    self.passed_tests += 1
                
                self.total_tests += 1
                
        except Exception as e:
            print(f"  ❌ CSS test error: {e}")
            self.total_tests += 1
    
    def test_contact_page_features(self):
        """Test contact page specific features."""
        print("📧 Testing contact page features...")
        
        try:
            response = requests.get(f"{self.base_url}/contact", timeout=10)
            if response.status_code == 200:
                content = response.text
                
                contact_checks = [
                    ('Contact Form', 'formspree.io' in content),
                    ('Phone Number', '+91 7624828106' in content),
                    ('Email Address', 'omkardigambar4@gmail.com' in content),
                    ('Social Media Links', 'instagram.com/omkar_kalagi' in content),
                    ('Team Section', 'team-member' in content),
                    ('Project Lead', 'Omkar Digambar' in content)
                ]
                
                contact_passed = 0
                for check_name, check_result in contact_checks:
                    if check_result:
                        print(f"  ✅ {check_name}: Present")
                        contact_passed += 1
                    else:
                        print(f"  ❌ {check_name}: Missing")
                
                if contact_passed >= 5:  # At least 5 out of 6 features
                    self.passed_tests += 1
                
                self.total_tests += 1
                
        except Exception as e:
            print(f"  ❌ Contact page test error: {e}")
            self.total_tests += 1
    
    def test_performance_optimizations(self):
        """Test performance optimization features."""
        print("🚀 Testing performance optimizations...")
        
        try:
            response = requests.get(self.base_url, timeout=10)
            if response.status_code == 200:
                content = response.text
                
                perf_checks = [
                    ('DNS Prefetch', 'dns-prefetch' in content),
                    ('Preconnect', 'preconnect' in content),
                    ('Preload Scripts', 'preload' in content and 'script' in content),
                    ('Deferred Scripts', 'defer' in content),
                    ('Critical CSS', 'Critical' in content and '<style>' in content)
                ]
                
                perf_passed = 0
                for check_name, check_result in perf_checks:
                    if check_result:
                        print(f"  ✅ {check_name}: Implemented")
                        perf_passed += 1
                    else:
                        print(f"  ❌ {check_name}: Missing")
                
                if perf_passed >= 3:  # At least 3 out of 5 optimizations
                    self.passed_tests += 1
                
                self.total_tests += 1
                
        except Exception as e:
            print(f"  ❌ Performance test error: {e}")
            self.total_tests += 1
    
    def run_comprehensive_test(self):
        """Run comprehensive component testing."""
        print("🧪 Running Comprehensive Component Testing")
        print("=" * 60)
        
        self.test_server_connectivity()
        self.test_all_routes()
        self.test_static_files()
        self.test_template_rendering()
        self.test_javascript_functionality()
        self.test_css_loading()
        self.test_contact_page_features()
        self.test_performance_optimizations()
        
        # Calculate success rate
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TESTING RESULTS")
        print("=" * 60)
        
        print(f"🎯 Tests Passed: {self.passed_tests}/{self.total_tests}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 EXCELLENT - All components working perfectly!")
            print("✅ Your website is production-ready!")
            status = "EXCELLENT"
        elif success_rate >= 75:
            print("👍 GOOD - Most components working well!")
            print("⚠️ Minor issues detected, but overall functional")
            status = "GOOD"
        elif success_rate >= 50:
            print("⚠️ FAIR - Some components need attention")
            print("🔧 Several issues need to be addressed")
            status = "FAIR"
        else:
            print("❌ POOR - Major issues detected")
            print("🚨 Significant problems need immediate attention")
            status = "POOR"
        
        print(f"\n🏆 OVERALL STATUS: {status}")
        print(f"🌟 Component testing complete!")
        
        return success_rate >= 75

def main():
    """Run comprehensive component testing."""
    print("🧪 AI Deepfake Detector - Comprehensive Component Testing")
    print("Testing all components, pages, and features!")
    print("=" * 70)
    
    tester = ComprehensiveComponentTester()
    success = tester.run_comprehensive_test()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
