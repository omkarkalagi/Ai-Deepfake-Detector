#!/usr/bin/env python3
"""
GitHub Repository Setup Script
Helps set up the GitHub repository for the AI Deepfake Detector
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr.strip()}")
        return False

def check_git_status():
    """Check if git is initialized and has commits."""
    try:
        subprocess.run(["git", "status"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main setup function."""
    print("🚀 AI Deepfake Detector - GitHub Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check if git is initialized
    if not check_git_status():
        print("❌ Error: Git repository not initialized. Please run 'git init' first.")
        sys.exit(1)
    
    print("📋 Before proceeding, make sure you have:")
    print("   1. Created a GitHub repository named 'ai-deepfake-detector'")
    print("   2. Have your GitHub username ready")
    print("   3. Have Git configured with your credentials")
    
    username = input("\n👤 Enter your GitHub username: ").strip()
    if not username:
        print("❌ Username is required!")
        sys.exit(1)
    
    repo_name = input("📁 Enter repository name (default: ai-deepfake-detector): ").strip()
    if not repo_name:
        repo_name = "ai-deepfake-detector"
    
    # Construct GitHub URL
    github_url = f"https://github.com/{username}/{repo_name}.git"
    
    print(f"\n🔗 GitHub URL: {github_url}")
    confirm = input("Is this correct? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Setup cancelled.")
        sys.exit(1)
    
    # Setup commands
    commands = [
        (f"git remote add origin {github_url}", "Adding GitHub remote"),
        ("git branch -M main", "Setting main branch"),
        ("git push -u origin main", "Pushing to GitHub")
    ]
    
    # Execute commands
    for command, description in commands:
        if not run_command(command, description):
            print(f"\n❌ Setup failed at: {description}")
            print("Please check the error and try again.")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 GitHub Setup Complete!")
    print("=" * 50)
    print(f"📁 Repository: https://github.com/{username}/{repo_name}")
    print(f"🚀 Deploy to Vercel: https://vercel.com/new/clone?repository-url=https://github.com/{username}/{repo_name}")
    
    print("\n📋 Next Steps:")
    print("1. Go to your GitHub repository")
    print("2. Click the 'Deploy to Vercel' button in the README")
    print("3. Or manually deploy at vercel.com")
    print("4. Your app will be live in 2-3 minutes!")
    
    print("\n📖 For detailed instructions, see DEPLOYMENT_GUIDE.md")

if __name__ == "__main__":
    main()
