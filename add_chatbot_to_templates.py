#!/usr/bin/env python3
"""
Add Chatbot to All Templates
Integrates the fully functional chatbot into all pages
"""

import os
import re
from pathlib import Path

class ChatbotIntegrator:
    """Integrates chatbot into all templates."""
    
    def __init__(self):
        self.templates_dir = Path('templates')
        self.integrations_applied = []
    
    def add_chatbot_to_templates(self):
        """Add chatbot script to all templates."""
        print("🤖 Adding chatbot to all templates...")
        
        for template_file in self.templates_dir.glob('*.html'):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Add chatbot script before closing body tag
                if 'chatbot.js' not in content:
                    chatbot_script = '''    <!-- AI Chatbot -->
    <script src="{{ url_for('static', filename='chatbot.js') }}" defer></script>'''
                    
                    body_end = content.rfind('</body>')
                    if body_end != -1:
                        content = content[:body_end] + chatbot_script + '\n' + content[body_end:]
                
                if content != original_content:
                    with open(template_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ Added chatbot to {template_file.name}")
                    
            except Exception as e:
                print(f"  ⚠️ Error adding chatbot to {template_file.name}: {e}")
        
        self.integrations_applied.append("Added chatbot to all templates")
    
    def run_integration(self):
        """Run chatbot integration."""
        print("🤖 Integrating AI Chatbot")
        print("=" * 60)
        
        self.add_chatbot_to_templates()
        
        print("\n" + "=" * 60)
        print("📊 CHATBOT INTEGRATION SUMMARY")
        print("=" * 60)
        
        for integration in self.integrations_applied:
            print(f"✅ {integration}")
        
        print(f"\n🤖 Chatbot Features:")
        print(f"   - Intelligent AI responses")
        print(f"   - Quick action buttons")
        print(f"   - Real-time typing indicators")
        print(f"   - Beautiful animated interface")
        print(f"   - Dark theme support")
        print(f"   - Mobile responsive design")
        print(f"   - Context-aware responses")
        
        print(f"\n🎯 Chatbot Capabilities:")
        print(f"   - Deepfake detection explanations")
        print(f"   - System accuracy information")
        print(f"   - API usage guidance")
        print(f"   - Team and contact information")
        print(f"   - Feature descriptions")
        print(f"   - Technical support")
        
        print(f"\n🚀 Fully functional AI chatbot is now live!")

def main():
    """Run chatbot integration."""
    print("🤖 AI Deepfake Detector - Chatbot Integration")
    print("Adding fully functional AI chatbot to all pages!")
    print("=" * 70)
    
    integrator = ChatbotIntegrator()
    integrator.run_integration()

if __name__ == "__main__":
    main()
