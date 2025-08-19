#!/usr/bin/env python3
"""
Add Stylish Footer to All Templates
Creates beautiful handwriting-style footer with love symbol
"""

import os
import re
from pathlib import Path

def add_stylish_footer_to_template(file_path):
    """Add stylish footer to a single template file."""
    print(f"Adding stylish footer to {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Footer HTML with handwriting style
    footer_html = '''
    <!-- Stylish Footer -->
    <footer class="stylish-footer">
        <div class="footer-content">
            <div class="footer-heart">
                <i class="fas fa-heart"></i>
            </div>
            <div class="footer-text">
                <p class="handwriting-text">
                    Made with <span class="love-symbol">♥</span> by 
                    <span class="creator-name">Omkar Kalagi</span>
                </p>
                <p class="company-text">
                    <span class="company-name">Kalagi Group of Companies</span>
                </p>
                <p class="footer-tagline">
                    "Innovating the Future with AI Excellence"
                </p>
            </div>
            <div class="footer-decoration">
                <div class="decoration-line"></div>
                <div class="decoration-dots">
                    <span></span><span></span><span></span>
                </div>
                <div class="decoration-line"></div>
            </div>
        </div>
        
        <!-- Footer Links -->
        <div class="footer-links">
            <div class="footer-section">
                <h6>AI Solutions</h6>
                <ul>
                    <li><a href="{{ url_for('home') }}">Deepfake Detection</a></li>
                    <li><a href="{{ url_for('realtime') }}">Real-time Analysis</a></li>
                    <li><a href="{{ url_for('batch') }}">Batch Processing</a></li>
                    <li><a href="{{ url_for('api') }}">API Services</a></li>
                </ul>
            </div>
            <div class="footer-section">
                <h6>Resources</h6>
                <ul>
                    <li><a href="{{ url_for('documentation') }}">Documentation</a></li>
                    <li><a href="{{ url_for('training') }}">Model Training</a></li>
                    <li><a href="{{ url_for('statistics') }}">Performance Stats</a></li>
                    <li><a href="{{ url_for('gallery') }}">Image Gallery</a></li>
                </ul>
            </div>
            <div class="footer-section">
                <h6>Company</h6>
                <ul>
                    <li><a href="{{ url_for('about') }}">About Us</a></li>
                    <li><a href="{{ url_for('contact') }}">Contact Team</a></li>
                    <li><a href="tel:+917624828106">Call: +91 7624828106</a></li>
                    <li><a href="mailto:omkar.digambar@kalagigroup.com">Email Support</a></li>
                </ul>
            </div>
            <div class="footer-section">
                <h6>Connect</h6>
                <div class="social-icons">
                    <a href="#" class="social-icon"><i class="fab fa-github"></i></a>
                    <a href="#" class="social-icon"><i class="fab fa-linkedin"></i></a>
                    <a href="#" class="social-icon"><i class="fab fa-twitter"></i></a>
                    <a href="#" class="social-icon"><i class="fab fa-instagram"></i></a>
                </div>
                <p class="footer-copyright">
                    © 2024 Kalagi Group of Companies<br>
                    All Rights Reserved
                </p>
            </div>
        </div>
    </footer>'''
    
    # Footer CSS with handwriting style
    footer_css = '''
        /* Stylish Footer Styles */
        .stylish-footer {
            background: linear-gradient(135deg, #2c3e50, #34495e, #2c3e50);
            color: #ecf0f1;
            padding: 3rem 0 1rem;
            margin-top: 4rem;
            position: relative;
            overflow: hidden;
        }

        .stylish-footer::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #e74c3c, #f39c12, #f1c40f, #27ae60, #3498db, #9b59b6, #e74c3c);
            background-size: 200% 100%;
            animation: rainbow 3s linear infinite;
        }

        @keyframes rainbow {
            0% { background-position: 0% 50%; }
            100% { background-position: 200% 50%; }
        }

        .footer-content {
            text-align: center;
            margin-bottom: 2rem;
            padding: 0 2rem;
        }

        .footer-heart {
            font-size: 3rem;
            color: #e74c3c;
            margin-bottom: 1rem;
            animation: heartbeat 2s ease-in-out infinite;
        }

        @keyframes heartbeat {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }

        .handwriting-text {
            font-family: 'Dancing Script', 'Brush Script MT', cursive;
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #ecf0f1;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }

        .love-symbol {
            color: #e74c3c;
            font-size: 2.2rem;
            animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.2); }
        }

        .creator-name {
            color: #f39c12;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }

        .company-text {
            font-family: 'Dancing Script', cursive;
            font-size: 1.5rem;
            color: #3498db;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }

        .company-name {
            background: linear-gradient(45deg, #3498db, #9b59b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
        }

        .footer-tagline {
            font-style: italic;
            color: #bdc3c7;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }

        .footer-decoration {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 1.5rem 0;
        }

        .decoration-line {
            width: 100px;
            height: 2px;
            background: linear-gradient(90deg, transparent, #3498db, transparent);
        }

        .decoration-dots {
            margin: 0 1rem;
            display: flex;
            gap: 0.5rem;
        }

        .decoration-dots span {
            width: 8px;
            height: 8px;
            background: #3498db;
            border-radius: 50%;
            animation: dotPulse 2s ease-in-out infinite;
        }

        .decoration-dots span:nth-child(2) {
            animation-delay: 0.3s;
        }

        .decoration-dots span:nth-child(3) {
            animation-delay: 0.6s;
        }

        @keyframes dotPulse {
            0%, 100% { opacity: 0.3; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.2); }
        }

        .footer-links {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            padding: 2rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 2rem;
        }

        .footer-section h6 {
            color: #f39c12;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .footer-section ul {
            list-style: none;
            padding: 0;
        }

        .footer-section ul li {
            margin-bottom: 0.5rem;
        }

        .footer-section ul li a {
            color: #bdc3c7;
            text-decoration: none;
            transition: all 0.3s ease;
            position: relative;
        }

        .footer-section ul li a:hover {
            color: #3498db;
            padding-left: 10px;
        }

        .footer-section ul li a::before {
            content: '→';
            position: absolute;
            left: -15px;
            opacity: 0;
            transition: all 0.3s ease;
        }

        .footer-section ul li a:hover::before {
            opacity: 1;
            left: -10px;
        }

        .social-icons {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .social-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #3498db, #2980b9);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            text-decoration: none;
            transition: all 0.3s ease;
        }

        .social-icon:hover {
            transform: translateY(-3px) scale(1.1);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }

        .footer-copyright {
            color: #7f8c8d;
            font-size: 0.9rem;
            line-height: 1.4;
        }

        /* Mobile Responsiveness */
        @media (max-width: 768px) {
            .handwriting-text {
                font-size: 1.5rem;
            }
            
            .company-text {
                font-size: 1.2rem;
            }
            
            .footer-links {
                grid-template-columns: 1fr;
                gap: 1.5rem;
                padding: 1rem;
            }
            
            .decoration-line {
                width: 50px;
            }
        }

        /* Dark theme adjustments */
        .dark-theme .stylish-footer {
            background: linear-gradient(135deg, #1a1a1a, #2d2d2d, #1a1a1a);
        }
    '''
    
    # Add footer CSS to the style section
    style_end = content.rfind('</style>')
    if style_end != -1:
        content = content[:style_end] + footer_css + '\n        ' + content[style_end:]
    
    # Add footer HTML before closing body tag
    body_end = content.rfind('</body>')
    if body_end != -1:
        content = content[:body_end] + footer_html + '\n' + content[body_end:]
    
    # Add Google Fonts for handwriting style
    head_end = content.find('</head>')
    if head_end != -1 and 'Dancing Script' not in content:
        google_fonts = '''    <link href="https://fonts.googleapis.com/css2?family=Dancing+Script:wght@400;600;700&display=swap" rel="stylesheet">
'''
        content = content[:head_end] + google_fonts + content[head_end:]
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Added stylish footer to {file_path}")
        return True
    else:
        print(f"  ⚪ No changes needed for {file_path}")
        return False

def main():
    """Add stylish footer to all templates."""
    print("✨ Adding Stylish Footer with Handwriting Style")
    print("=" * 60)
    
    templates_dir = Path('templates')
    html_files = list(templates_dir.glob('*.html'))
    
    updated_count = 0
    
    for html_file in html_files:
        if add_stylish_footer_to_template(html_file):
            updated_count += 1
    
    print(f"\n✅ Stylish footer addition completed!")
    print(f"📊 Updated {updated_count} out of {len(html_files)} templates")
    print("💝 Beautiful handwriting-style footer with love from Omkar Kalagi!")
    print("🏢 Proudly by Kalagi Group of Companies")

if __name__ == "__main__":
    main()
