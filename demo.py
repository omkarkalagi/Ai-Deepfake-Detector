#!/usr/bin/env python3
"""
Demo script for the Advanced Deepfake Detector system.
Showcases the capabilities and provides interactive examples.
"""

import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path
import threading


class DeepfakeDetectorDemo:
    """Interactive demo for the deepfake detection system."""
    
    def __init__(self):
        self.demo_running = False
        self.server_process = None
    
    def print_banner(self):
        """Print the demo banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🧠 ADVANCED AI DEEPFAKE DETECTOR                          ║
║                                                                              ║
║                        Interactive Demo System                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Welcome to the Advanced Deepfake Detector Demo!

This system provides:
✨ Beautiful web interface with real-time analysis
📊 Comprehensive analytics and visualizations  
🔬 Advanced image feature extraction
🎯 High-accuracy deepfake detection
📈 Interactive charts and metrics

"""
        print(banner)
    
    def check_requirements(self):
        """Check if all requirements are met."""
        print("🔍 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major >= 3 and python_version.minor >= 8:
            print(f"✅ Python {python_version.major}.{python_version.minor} - OK")
        else:
            print(f"❌ Python {python_version.major}.{python_version.minor} - Requires Python 3.8+")
            return False
        
        # Check required files
        required_files = [
            "app.py",
            "templates/index.html",
            "requirements.txt"
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} - Found")
            else:
                print(f"❌ {file_path} - Missing")
                return False
        
        # Check model file
        model_files = ["deepfake_detector_model.keras", "deepfake_detector_model_enhanced.keras"]
        model_found = any(os.path.exists(f) for f in model_files)
        
        if model_found:
            found_model = next(f for f in model_files if os.path.exists(f))
            print(f"✅ Model file - Found ({found_model})")
        else:
            print("⚠️  No model file found - You may need to train a model first")
        
        # Check uploads directory
        if not os.path.exists("uploads"):
            print("📁 Creating uploads directory...")
            os.makedirs("uploads")
            print("✅ Uploads directory created")
        else:
            print("✅ Uploads directory - Found")
        
        return True
    
    def install_dependencies(self):
        """Install required dependencies."""
        print("\n📦 Installing dependencies...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def start_demo_server(self):
        """Start the demo server."""
        print("\n🚀 Starting demo server...")
        
        try:
            # Start the advanced app
            self.server_process = subprocess.Popen(
                [sys.executable, "app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            print("⏳ Waiting for server to start...")
            time.sleep(3)
            
            # Check if server is running
            if self.server_process.poll() is None:
                print("✅ Demo server started successfully!")
                print("🌐 Server running at: http://localhost:5000")
                return True
            else:
                print("❌ Server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def open_browser(self):
        """Open the demo in browser."""
        print("\n🌐 Opening demo in your default browser...")
        
        try:
            webbrowser.open("http://localhost:5000")
            print("✅ Browser opened successfully")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print("Please manually open: http://localhost:5000")
    
    def show_demo_instructions(self):
        """Show demo instructions."""
        instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                              DEMO INSTRUCTIONS                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 How to use the demo:

1. 📁 UPLOAD AN IMAGE
   • Click "Choose Image File" button
   • Select a PNG, JPG, or JPEG image (max 10MB)
   • Supported formats: .png, .jpg, .jpeg

2. 🔍 ANALYZE THE IMAGE  
   • Click "Analyze Image" button
   • Wait for the AI analysis to complete
   • View comprehensive results

3. 📊 EXPLORE THE RESULTS
   • Check the prediction result (Real/Fake)
   • View confidence scores and percentages
   • Explore detailed feature analysis
   • Examine interactive charts and graphs

4. 🔬 ADVANCED FEATURES
   • Image quality assessment
   • Edge detection analysis
   • Color and texture evaluation
   • Compression artifact detection
   • Processing time metrics

5. 📈 VISUALIZATIONS
   • Prediction distribution pie chart
   • Feature analysis bar chart
   • Technical metrics breakdown
   • Model performance statistics

💡 Tips:
   • Try different types of images
   • Compare real photos vs AI-generated images
   • Notice the detailed analytics provided
   • Explore the beautiful, responsive interface

🎨 Interface Features:
   • Modern gradient design
   • Smooth animations
   • Interactive charts
   • Responsive layout
   • Real-time feedback

"""
        print(instructions)
    
    def monitor_server(self):
        """Monitor server status."""
        while self.demo_running:
            if self.server_process and self.server_process.poll() is not None:
                print("\n⚠️  Server has stopped unexpectedly")
                break
            time.sleep(5)
    
    def run_demo(self):
        """Run the complete demo."""
        self.print_banner()
        
        # Check requirements
        if not self.check_requirements():
            print("\n❌ Requirements check failed. Please fix the issues above.")
            return
        
        # Ask about dependency installation
        install_deps = input("\n📦 Install/update dependencies? (y/n): ").lower().strip()
        if install_deps in ['y', 'yes']:
            if not self.install_dependencies():
                print("\n❌ Dependency installation failed.")
                return
        
        # Start server
        if not self.start_demo_server():
            print("\n❌ Failed to start demo server.")
            return
        
        # Show instructions
        self.show_demo_instructions()
        
        # Open browser
        open_browser = input("🌐 Open demo in browser automatically? (y/n): ").lower().strip()
        if open_browser in ['y', 'yes']:
            self.open_browser()
        
        # Start monitoring
        self.demo_running = True
        monitor_thread = threading.Thread(target=self.monitor_server, daemon=True)
        monitor_thread.start()
        
        # Keep demo running
        try:
            print("\n" + "="*80)
            print("🎉 DEMO IS NOW RUNNING!")
            print("="*80)
            print("📍 URL: http://localhost:5000")
            print("⌨️  Press Ctrl+C to stop the demo")
            print("="*80)
            
            while self.demo_running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Demo stopped by user")
        finally:
            self.stop_demo()
    
    def stop_demo(self):
        """Stop the demo server."""
        self.demo_running = False
        
        if self.server_process:
            print("🛑 Stopping demo server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                print("✅ Demo server stopped successfully")
            except subprocess.TimeoutExpired:
                print("⚠️  Force killing server...")
                self.server_process.kill()
                self.server_process.wait()
        
        print("\n🎉 Thank you for trying the Advanced Deepfake Detector!")
        print("💡 For more information, check the README.md file")


def main():
    """Main function to run the demo."""
    demo = DeepfakeDetectorDemo()
    
    try:
        demo.run_demo()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        demo.stop_demo()


if __name__ == "__main__":
    main()
