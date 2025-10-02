#!/usr/bin/env python
"""
Simple launcher script for the AI Deepfake Detector
This provides an easy way to run the application locally
"""

import os
import sys
import subprocess

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Path to the main application
    app_path = os.path.join(script_dir, 'api', 'index.py')
    
    if not os.path.exists(app_path):
        print(f"❌ Error: Could not find {app_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("🚀 AI Deepfake Detector - Starting Server")
    print("=" * 60)
    print(f"📁 Project Directory: {script_dir}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🌐 Server will start at: http://localhost:5000")
    print("=" * 60)
    print("\n⏳ Starting Flask application...\n")
    
    # Run the Flask app
    try:
        subprocess.run([sys.executable, app_path], check=True)
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
