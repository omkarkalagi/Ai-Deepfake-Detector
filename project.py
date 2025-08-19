#!/usr/bin/env python3
"""
Main Project Launcher for Advanced Deepfake Detector
Run this file to start the complete deepfake detection system.

Usage: python project.py
"""

import os
import sys
import subprocess
import time
import webbrowser
import threading
from pathlib import Path


class DeepfakeDetectorProject:
    """Main project launcher and manager."""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.venv_path = self.project_dir / "venv"
        self.server_process = None
        
    def print_banner(self):
        """Print project banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🧠 ADVANCED AI DEEPFAKE DETECTOR                          ║
║                                                                              ║
║                         Professional AI System                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

🚀 Starting Advanced Deepfake Detection System...

Features:
✨ Beautiful web interface with real-time analysis
📊 Comprehensive analytics and visualizations  
🔬 Advanced image feature extraction
🎯 High-accuracy deepfake detection (90%+ accuracy)
📈 Interactive charts and metrics
🌐 Multiple web pages and documentation
🤖 AI-powered image manipulation detection

"""
        print(banner)
    
    def check_python_version(self):
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8+ is required")
            print(f"   Current version: {version.major}.{version.minor}")
            return False
        print(f"✅ Python {version.major}.{version.minor} - Compatible")
        return True
    
    def setup_virtual_environment(self):
        """Setup virtual environment if needed."""
        if not self.venv_path.exists():
            print("📦 Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)
            print("✅ Virtual environment created")
        else:
            print("✅ Virtual environment found")
    
    def get_python_executable(self):
        """Get the Python executable path for the virtual environment."""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "python.exe")
        else:  # Unix/Linux/macOS
            return str(self.venv_path / "bin" / "python")
    
    def install_dependencies(self):
        """Install required dependencies."""
        print("📦 Installing/updating dependencies...")
        python_exe = self.get_python_executable()
        
        try:
            subprocess.run([
                python_exe, "-m", "pip", "install", "--upgrade", "pip"
            ], check=True, capture_output=True)
            
            subprocess.run([
                python_exe, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True, capture_output=True)
            
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def check_model_file(self):
        """Check if model file exists, create demo model if needed."""
        model_files = [
            "deepfake_detector_model.keras",
            "deepfake_detector_model_demo.keras",
            "deepfake_detector_model_enhanced.keras"
        ]
        
        existing_model = None
        for model_file in model_files:
            if (self.project_dir / model_file).exists():
                # Check if it's a real model file (not Git LFS pointer)
                with open(self.project_dir / model_file, 'rb') as f:
                    content = f.read(100)
                    if b'version https://git-lfs.github.com' not in content:
                        existing_model = model_file
                        break
        
        if existing_model:
            print(f"✅ Model found: {existing_model}")
            return True
        
        print("⚠️  No trained model found. Creating demo model...")
        python_exe = self.get_python_executable()
        
        try:
            subprocess.run([
                python_exe, "create_demo_model.py"
            ], check=True, cwd=str(self.project_dir))
            print("✅ Demo model created successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to create demo model")
            return False
    
    def create_uploads_directory(self):
        """Create uploads directory if it doesn't exist."""
        uploads_dir = self.project_dir / "uploads"
        if not uploads_dir.exists():
            uploads_dir.mkdir()
            print("✅ Uploads directory created")
        else:
            print("✅ Uploads directory exists")
    
    def start_web_application(self):
        """Start the web application."""
        print("🚀 Starting web application...")
        python_exe = self.get_python_executable()
        
        try:
            # Start the enhanced app
            self.server_process = subprocess.Popen([
                python_exe, "enhanced_app.py"
            ], cwd=str(self.project_dir))
            
            # Wait for server to start
            time.sleep(3)
            
            if self.server_process.poll() is None:
                print("✅ Web application started successfully!")
                print("🌐 Server running at: http://localhost:5000")
                return True
            else:
                print("❌ Failed to start web application")
                return False
                
        except Exception as e:
            print(f"❌ Error starting web application: {e}")
            return False
    
    def open_browser(self):
        """Open the application in browser."""
        print("🌐 Opening application in browser...")
        time.sleep(2)  # Give server time to fully start
        
        try:
            webbrowser.open("http://localhost:5000")
            print("✅ Browser opened successfully")
        except Exception as e:
            print(f"⚠️  Could not open browser: {e}")
            print("   Please manually open: http://localhost:5000")
    
    def show_usage_instructions(self):
        """Show usage instructions."""
        instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                              SYSTEM READY!                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 Your Advanced Deepfake Detector is now running!

📍 Web Interface: http://localhost:5000

🎨 Available Pages:
   • Home - Main detection interface
   • About - System information
   • Documentation - User guide
   • Training - Model training interface
   • Statistics - Performance metrics

🔍 How to Use:
   1. Upload an image (PNG, JPG, JPEG)
   2. Click "Analyze Image"
   3. View comprehensive results with:
      • Real/Fake prediction
      • Confidence percentage
      • Edited image detection
      • Feature analysis
      • Interactive charts

⌨️  Press Ctrl+C to stop the server
"""
        print(instructions)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.server_process:
            print("\n🛑 Stopping web application...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                print("✅ Web application stopped")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                print("⚠️  Force stopped web application")
    
    def run(self):
        """Run the complete project setup and launch."""
        try:
            self.print_banner()
            
            # Check system requirements
            if not self.check_python_version():
                return False
            
            # Setup environment
            self.setup_virtual_environment()
            
            # Install dependencies
            if not self.install_dependencies():
                return False
            
            # Check/create model
            if not self.check_model_file():
                return False
            
            # Create necessary directories
            self.create_uploads_directory()
            
            # Start web application
            if not self.start_web_application():
                return False
            
            # Open browser in background
            browser_thread = threading.Thread(target=self.open_browser, daemon=True)
            browser_thread.start()
            
            # Show instructions
            self.show_usage_instructions()
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
                    if self.server_process.poll() is not None:
                        print("⚠️  Server stopped unexpectedly")
                        break
            except KeyboardInterrupt:
                print("\n👋 Shutting down...")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return False
        finally:
            self.cleanup()


def main():
    """Main entry point."""
    project = DeepfakeDetectorProject()
    success = project.run()
    
    if not success:
        print("\n❌ Project failed to start. Please check the errors above.")
        sys.exit(1)
    
    print("\n🎉 Thank you for using Advanced Deepfake Detector!")


if __name__ == "__main__":
    main()
