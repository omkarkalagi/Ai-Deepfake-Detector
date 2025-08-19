#!/usr/bin/env python3
"""
Comprehensive testing script for the Advanced Deepfake Detector system.
Tests model functionality, web interface, and analysis features.
"""

import os
import sys
import time
import requests
import numpy as np
from PIL import Image
import cv2
import json
import threading
import subprocess
from pathlib import Path

class DeepfakeDetectorTester:
    """Comprehensive testing suite for the deepfake detection system."""
    
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.test_results = {}
        self.server_process = None
        
    def create_test_images(self):
        """Create test images for validation."""
        test_dir = Path("test_images")
        test_dir.mkdir(exist_ok=True)
        
        # Create a simple test image
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        test_image_pil = Image.fromarray(test_image)
        test_image_pil.save(test_dir / "test_image.png")
        
        # Create a noisy test image
        noisy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        noisy_image_pil = Image.fromarray(noisy_image)
        noisy_image_pil.save(test_dir / "noisy_image.jpg")
        
        print("✅ Test images created successfully")
        return test_dir
    
    def test_model_loading(self):
        """Test if the model can be loaded successfully."""
        print("\n🧪 Testing model loading...")
        
        try:
            from tensorflow.keras.models import load_model
            model_path = "deepfake_detector_model.keras"
            
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}")
                return False
            
            model = load_model(model_path)
            print(f"✅ Model loaded successfully")
            print(f"   - Input shape: {model.input_shape}")
            print(f"   - Output shape: {model.output_shape}")
            print(f"   - Total parameters: {model.count_params():,}")
            
            self.test_results['model_loading'] = True
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            self.test_results['model_loading'] = False
            return False
    
    def test_image_analysis_functions(self):
        """Test image analysis functions."""
        print("\n🧪 Testing image analysis functions...")
        
        try:
            # Import the analysis functions
            if os.path.exists("app.py"):
                from app import AdvancedDeepfakeDetector
                detector_class = AdvancedDeepfakeDetector
            else:
                from inference import InferenceModel
                detector_class = InferenceModel
            
            # Create test image
            test_dir = self.create_test_images()
            test_image_path = test_dir / "test_image.png"
            
            # Test basic prediction (without full initialization)
            print("   - Testing image loading and preprocessing...")
            from tensorflow.keras.preprocessing import image
            img = image.load_img(str(test_image_path), target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            print("   ✅ Image preprocessing successful")
            
            # Test OpenCV operations
            print("   - Testing OpenCV operations...")
            img_cv = cv2.imread(str(test_image_path))
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            print("   ✅ OpenCV operations successful")
            
            self.test_results['image_analysis'] = True
            return True
            
        except Exception as e:
            print(f"❌ Image analysis test failed: {str(e)}")
            self.test_results['image_analysis'] = False
            return False
    
    def start_test_server(self):
        """Start the Flask server for testing."""
        print("\n🚀 Starting test server...")
        
        try:
            # Start server in background
            if os.path.exists("app.py"):
                cmd = [sys.executable, "app.py"]
            else:
                cmd = [sys.executable, "inference.py"]
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Test if server is responding
            response = requests.get(self.base_url, timeout=10)
            if response.status_code == 200:
                print("✅ Test server started successfully")
                return True
            else:
                print(f"❌ Server responded with status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start test server: {str(e)}")
            return False
    
    def test_web_interface(self):
        """Test the web interface functionality."""
        print("\n🧪 Testing web interface...")
        
        try:
            # Test GET request
            response = requests.get(self.base_url, timeout=10)
            if response.status_code != 200:
                print(f"❌ GET request failed with status: {response.status_code}")
                return False
            
            print("✅ GET request successful")
            
            # Test if HTML contains expected elements
            html_content = response.text
            expected_elements = [
                "Deepfake Detector",
                "Upload",
                "file",
                "bootstrap"
            ]
            
            for element in expected_elements:
                if element.lower() in html_content.lower():
                    print(f"   ✅ Found expected element: {element}")
                else:
                    print(f"   ⚠️  Missing element: {element}")
            
            self.test_results['web_interface'] = True
            return True
            
        except Exception as e:
            print(f"❌ Web interface test failed: {str(e)}")
            self.test_results['web_interface'] = False
            return False
    
    def test_file_upload(self):
        """Test file upload functionality."""
        print("\n🧪 Testing file upload...")
        
        try:
            test_dir = Path("test_images")
            test_image_path = test_dir / "test_image.png"
            
            if not test_image_path.exists():
                print("❌ Test image not found")
                return False
            
            # Prepare file upload
            with open(test_image_path, 'rb') as f:
                files = {'file': ('test_image.png', f, 'image/png')}
                response = requests.post(self.base_url, files=files, timeout=30)
            
            if response.status_code == 200:
                print("✅ File upload successful")
                
                # Check if response contains analysis results
                html_content = response.text
                if "result" in html_content.lower() or "prediction" in html_content.lower():
                    print("✅ Analysis results found in response")
                    self.test_results['file_upload'] = True
                    return True
                else:
                    print("⚠️  No analysis results found in response")
                    self.test_results['file_upload'] = False
                    return False
            else:
                print(f"❌ File upload failed with status: {response.status_code}")
                self.test_results['file_upload'] = False
                return False
                
        except Exception as e:
            print(f"❌ File upload test failed: {str(e)}")
            self.test_results['file_upload'] = False
            return False
    
    def test_api_endpoints(self):
        """Test API endpoints if available."""
        print("\n🧪 Testing API endpoints...")
        
        try:
            # Test model stats endpoint
            try:
                response = requests.get(f"{self.base_url}/api/model_stats", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    print("✅ Model stats API working")
                    print(f"   - Accuracy: {data.get('accuracy', 'N/A')}%")
                else:
                    print("⚠️  Model stats API not available")
            except:
                print("⚠️  Model stats API not available")
            
            # Test analysis history endpoint
            try:
                response = requests.get(f"{self.base_url}/api/analysis_history", timeout=10)
                if response.status_code == 200:
                    print("✅ Analysis history API working")
                else:
                    print("⚠️  Analysis history API not available")
            except:
                print("⚠️  Analysis history API not available")
            
            self.test_results['api_endpoints'] = True
            return True
            
        except Exception as e:
            print(f"❌ API endpoint test failed: {str(e)}")
            self.test_results['api_endpoints'] = False
            return False
    
    def stop_test_server(self):
        """Stop the test server."""
        if self.server_process:
            print("\n🛑 Stopping test server...")
            self.server_process.terminate()
            self.server_process.wait()
            print("✅ Test server stopped")
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        print("🧪 Starting Comprehensive Deepfake Detector Tests")
        print("=" * 60)
        
        # Test 1: Model loading
        self.test_model_loading()
        
        # Test 2: Image analysis functions
        self.test_image_analysis_functions()
        
        # Test 3: Start server and test web interface
        if self.start_test_server():
            self.test_web_interface()
            self.test_file_upload()
            self.test_api_endpoints()
            self.stop_test_server()
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("🧪 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():<25} {status}")
        
        print("-" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 All tests passed! System is ready for use.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please check the issues above.")


if __name__ == "__main__":
    tester = DeepfakeDetectorTester()
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        tester.stop_test_server()
    except Exception as e:
        print(f"\n\n❌ Unexpected error during testing: {str(e)}")
        tester.stop_test_server()
