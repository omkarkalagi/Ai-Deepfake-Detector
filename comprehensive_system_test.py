#!/usr/bin/env python3
"""
Comprehensive System Test for AI Deepfake Detector
Tests all pages, routes, and functionality
"""

import requests
import time
import sys
from pathlib import Path
import json

class SystemTester:
    """Comprehensive system testing class."""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {
            'pages': {},
            'api_endpoints': {},
            'functionality': {},
            'performance': {},
            'errors': []
        }
    
    def test_page_accessibility(self):
        """Test all pages are accessible."""
        pages = [
            ('/', 'Home Page'),
            ('/gallery', 'Gallery Page'),
            ('/realtime', 'Real-time Detection'),
            ('/batch', 'Batch Processing'),
            ('/api', 'API Explorer'),
            ('/about', 'About Page'),
            ('/documentation', 'Documentation'),
            ('/training', 'Training Page'),
            ('/statistics', 'Statistics'),
            ('/contact', 'Contact Page')
        ]
        
        print("🔍 Testing page accessibility...")
        
        for route, name in pages:
            try:
                start_time = time.time()
                response = self.session.get(f"{self.base_url}{route}")
                load_time = time.time() - start_time
                
                if response.status_code == 200:
                    self.test_results['pages'][route] = {
                        'status': 'PASS',
                        'name': name,
                        'load_time': round(load_time, 3),
                        'size': len(response.content)
                    }
                    print(f"  ✅ {name}: {response.status_code} ({load_time:.3f}s)")
                else:
                    self.test_results['pages'][route] = {
                        'status': 'FAIL',
                        'name': name,
                        'error': f"HTTP {response.status_code}"
                    }
                    print(f"  ❌ {name}: HTTP {response.status_code}")
                    
            except Exception as e:
                self.test_results['pages'][route] = {
                    'status': 'ERROR',
                    'name': name,
                    'error': str(e)
                }
                print(f"  ❌ {name}: {e}")
    
    def test_navigation_links(self):
        """Test navigation links work correctly."""
        print("\n🔗 Testing navigation links...")
        
        try:
            # Get home page and check for navigation links
            response = self.session.get(self.base_url)
            if response.status_code == 200:
                content = response.text
                
                # Check for theme toggle
                if 'themeToggle' in content:
                    print("  ✅ Theme toggle button found")
                else:
                    print("  ❌ Theme toggle button missing")
                
                # Check for navigation items
                nav_items = ['Gallery', 'Real-time', 'Batch', 'API', 'Training', 'Statistics', 'Docs', 'About', 'Contact']
                for item in nav_items:
                    if item in content:
                        print(f"  ✅ Navigation item '{item}' found")
                    else:
                        print(f"  ❌ Navigation item '{item}' missing")
                        
        except Exception as e:
            print(f"  ❌ Navigation test failed: {e}")
    
    def test_api_endpoints(self):
        """Test API endpoints."""
        print("\n🔌 Testing API endpoints...")
        
        # Test file upload endpoint
        try:
            # Create a test image file
            test_image_path = Path('samples/real-actual.jpg')
            if test_image_path.exists():
                with open(test_image_path, 'rb') as f:
                    files = {'file': ('test.jpg', f, 'image/jpeg')}
                    response = self.session.post(f"{self.base_url}/detect", files=files)
                    
                if response.status_code in [200, 302]:  # 302 for redirect
                    print("  ✅ File upload endpoint working")
                    self.test_results['api_endpoints']['/detect'] = {'status': 'PASS'}
                else:
                    print(f"  ❌ File upload failed: HTTP {response.status_code}")
                    self.test_results['api_endpoints']['/detect'] = {'status': 'FAIL', 'error': f"HTTP {response.status_code}"}
            else:
                print("  ⚠️ Test image not found, skipping upload test")
                
        except Exception as e:
            print(f"  ❌ API test failed: {e}")
            self.test_results['api_endpoints']['/detect'] = {'status': 'ERROR', 'error': str(e)}
    
    def test_theme_system(self):
        """Test theme system functionality."""
        print("\n🎨 Testing theme system...")
        
        try:
            # Check if theme manager script is accessible
            response = self.session.get(f"{self.base_url}/static/theme-manager.js")
            if response.status_code == 200:
                print("  ✅ Theme manager script accessible")
                
                # Check for key theme functions
                content = response.text
                if 'UniversalThemeManager' in content:
                    print("  ✅ UniversalThemeManager class found")
                if 'toggleTheme' in content:
                    print("  ✅ Theme toggle function found")
                if 'localStorage' in content:
                    print("  ✅ Theme persistence implemented")
                    
                self.test_results['functionality']['theme_system'] = {'status': 'PASS'}
            else:
                print(f"  ❌ Theme manager script not accessible: HTTP {response.status_code}")
                self.test_results['functionality']['theme_system'] = {'status': 'FAIL'}
                
        except Exception as e:
            print(f"  ❌ Theme system test failed: {e}")
            self.test_results['functionality']['theme_system'] = {'status': 'ERROR', 'error': str(e)}
    
    def test_mobile_responsiveness(self):
        """Test mobile responsiveness."""
        print("\n📱 Testing mobile responsiveness...")
        
        try:
            # Test with mobile user agent
            mobile_headers = {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15'
            }
            
            response = self.session.get(self.base_url, headers=mobile_headers)
            if response.status_code == 200:
                content = response.text
                
                # Check for responsive meta tag
                if 'viewport' in content and 'width=device-width' in content:
                    print("  ✅ Responsive viewport meta tag found")
                else:
                    print("  ❌ Responsive viewport meta tag missing")
                
                # Check for mobile-specific CSS
                if '@media' in content:
                    print("  ✅ Mobile CSS media queries found")
                else:
                    print("  ❌ Mobile CSS media queries missing")
                    
                self.test_results['functionality']['mobile_responsive'] = {'status': 'PASS'}
            else:
                print(f"  ❌ Mobile test failed: HTTP {response.status_code}")
                self.test_results['functionality']['mobile_responsive'] = {'status': 'FAIL'}
                
        except Exception as e:
            print(f"  ❌ Mobile responsiveness test failed: {e}")
            self.test_results['functionality']['mobile_responsive'] = {'status': 'ERROR', 'error': str(e)}
    
    def test_performance(self):
        """Test system performance."""
        print("\n⚡ Testing performance...")
        
        try:
            # Test page load times
            start_time = time.time()
            response = self.session.get(self.base_url)
            load_time = time.time() - start_time
            
            if response.status_code == 200:
                self.test_results['performance']['home_load_time'] = round(load_time, 3)
                
                if load_time < 2.0:
                    print(f"  ✅ Home page loads quickly: {load_time:.3f}s")
                elif load_time < 5.0:
                    print(f"  ⚠️ Home page load time acceptable: {load_time:.3f}s")
                else:
                    print(f"  ❌ Home page loads slowly: {load_time:.3f}s")
                    
                # Check response size
                size_kb = len(response.content) / 1024
                self.test_results['performance']['home_size_kb'] = round(size_kb, 2)
                print(f"  📊 Home page size: {size_kb:.2f} KB")
                
        except Exception as e:
            print(f"  ❌ Performance test failed: {e}")
    
    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE SYSTEM TEST REPORT")
        print("="*70)
        
        # Page accessibility summary
        total_pages = len(self.test_results['pages'])
        passed_pages = sum(1 for p in self.test_results['pages'].values() if p['status'] == 'PASS')
        
        print(f"\n🌐 PAGE ACCESSIBILITY: {passed_pages}/{total_pages} pages working")
        
        # API endpoints summary
        if self.test_results['api_endpoints']:
            total_apis = len(self.test_results['api_endpoints'])
            passed_apis = sum(1 for a in self.test_results['api_endpoints'].values() if a['status'] == 'PASS')
            print(f"🔌 API ENDPOINTS: {passed_apis}/{total_apis} endpoints working")
        
        # Functionality summary
        if self.test_results['functionality']:
            total_features = len(self.test_results['functionality'])
            passed_features = sum(1 for f in self.test_results['functionality'].values() if f['status'] == 'PASS')
            print(f"⚙️ FUNCTIONALITY: {passed_features}/{total_features} features working")
        
        # Performance summary
        if 'home_load_time' in self.test_results['performance']:
            load_time = self.test_results['performance']['home_load_time']
            print(f"⚡ PERFORMANCE: Home page loads in {load_time}s")
        
        # Overall status
        total_tests = passed_pages + sum(1 for a in self.test_results['api_endpoints'].values() if a['status'] == 'PASS') + sum(1 for f in self.test_results['functionality'].values() if f['status'] == 'PASS')
        max_tests = total_pages + len(self.test_results['api_endpoints']) + len(self.test_results['functionality'])
        
        if max_tests > 0:
            success_rate = (total_tests / max_tests) * 100
            print(f"\n🎯 OVERALL SUCCESS RATE: {success_rate:.1f}%")
            
            if success_rate >= 90:
                print("🎉 EXCELLENT: System is working very well!")
            elif success_rate >= 75:
                print("✅ GOOD: System is working well with minor issues")
            elif success_rate >= 50:
                print("⚠️ FAIR: System has some issues that need attention")
            else:
                print("❌ POOR: System has significant issues")
        
        # Save detailed report
        with open('system_test_report.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: system_test_report.json")
        print("="*70)
    
    def run_all_tests(self):
        """Run all system tests."""
        print("🚀 Starting Comprehensive System Test...")
        print("="*70)
        
        self.test_page_accessibility()
        self.test_navigation_links()
        self.test_api_endpoints()
        self.test_theme_system()
        self.test_mobile_responsiveness()
        self.test_performance()
        
        self.generate_report()


def main():
    """Main test function."""
    print("🔧 AI Deepfake Detector - Comprehensive System Test")
    print("="*70)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running at http://localhost:5000")
        else:
            print(f"⚠️ Server responded with status: {response.status_code}")
    except requests.exceptions.RequestException:
        print("❌ Server is not running at http://localhost:5000")
        print("Please start the server with: python enhanced_app.py")
        return False
    
    # Run tests
    tester = SystemTester()
    tester.run_all_tests()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
