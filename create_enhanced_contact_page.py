#!/usr/bin/env python3
"""
Create Enhanced Contact Page with Formspree Integration and Real Social Links
"""

import os
from pathlib import Path

def create_enhanced_contact_page():
    """Create a completely new enhanced contact page."""
    
    contact_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contact - AI Deepfake Detector</title>
    <link rel="icon" href="{{ url_for('static', filename='k.ico') }}" type="image/x-icon">
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <!-- Animate.css -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Dancing+Script:wght@400;600;700&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --info-color: #17a2b8;
        }

        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            transition: all 0.3s ease;
        }

        .main-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            margin: 20px auto;
            padding: 40px;
            max-width: 1200px;
        }

        /* Enhanced Contact Styles */
        .lead-contact-section {
            background: linear-gradient(135deg, rgba(52, 152, 219, 0.1), rgba(155, 89, 182, 0.1));
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 2px solid rgba(52, 152, 219, 0.2);
        }

        .lead-header {
            display: flex;
            align-items: center;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }

        .lead-avatar-large {
            width: 120px;
            height: 120px;
            background: linear-gradient(135deg, #3498db, #9b59b6);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 2rem;
            color: white;
            font-size: 3.5rem;
            box-shadow: 0 15px 35px rgba(52, 152, 219, 0.4);
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        .lead-info-main h2 {
            color: #2c3e50;
            margin-bottom: 0.5rem;
            font-weight: 700;
            font-size: 2.5rem;
        }

        .lead-title {
            color: #3498db;
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .lead-company {
            color: #7f8c8d;
            font-style: italic;
            font-size: 1.1rem;
            margin-bottom: 0;
        }

        .contact-methods {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }

        .contact-method {
            display: flex;
            align-items: center;
            padding: 1.5rem;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }

        .contact-method:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
        }

        .contact-method .contact-icon {
            width: 70px;
            height: 70px;
            background: linear-gradient(135deg, #27ae60, #2ecc71);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 1.5rem;
            color: white;
            font-size: 1.8rem;
        }

        .contact-details h5 {
            margin-bottom: 0.5rem;
            color: #2c3e50;
            font-weight: 600;
        }

        .contact-link {
            color: #3498db;
            text-decoration: none;
            font-weight: 600;
            font-size: 1.1rem;
        }

        .contact-link:hover {
            color: #2980b9;
            text-decoration: underline;
        }

        /* Social Media Section */
        .social-media-section {
            background: linear-gradient(135deg, rgba(231, 76, 60, 0.1), rgba(192, 57, 43, 0.1));
            border-radius: 20px;
            padding: 2rem;
            margin: 2rem 0;
            text-align: center;
        }

        .social-links {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 30px;
            flex-wrap: wrap;
        }

        .social-link {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 80px;
            height: 80px;
            border-radius: 20px;
            color: white;
            text-decoration: none;
            font-size: 2rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .social-link::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: inherit;
            opacity: 0.8;
            transition: all 0.3s ease;
        }

        .social-link:hover::before {
            opacity: 1;
        }

        .social-link:hover {
            transform: translateY(-5px) scale(1.1);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
            color: white;
        }

        .social-link.whatsapp {
            background: linear-gradient(135deg, #25D366, #128C7E);
        }

        .social-link.linkedin {
            background: linear-gradient(135deg, #0077B5, #005885);
        }

        .social-link.github {
            background: linear-gradient(135deg, #333, #24292e);
        }

        .social-link.portfolio {
            background: linear-gradient(135deg, #667eea, #764ba2);
        }

        .social-link.youtube {
            background: linear-gradient(135deg, #FF0000, #CC0000);
        }

        .social-link.telegram {
            background: linear-gradient(135deg, #0088cc, #006699);
        }

        .social-link span {
            font-size: 0.8rem;
            margin-top: 5px;
            font-weight: 600;
        }

        /* Contact Form */
        .contact-form {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 20px;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        .form-control {
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 15px;
            transition: all 0.3s ease;
            font-size: 1rem;
        }

        .form-control:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .btn-submit {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            border: none;
            padding: 15px 40px;
            border-radius: 25px;
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            width: 100%;
        }

        .btn-submit:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
        }

        /* Team Section */
        .team-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }

        .team-member {
            background: white;
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            position: relative;
        }

        .team-member:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        }

        .team-member.lead-member {
            border: 3px solid #f39c12;
            background: linear-gradient(135deg, rgba(243, 156, 18, 0.1), rgba(230, 126, 34, 0.1));
        }

        .team-member.lead-member::before {
            content: "PROJECT LEAD";
            position: absolute;
            top: -15px;
            left: 50%;
            transform: translateX(-50%);
            background: #f39c12;
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            letter-spacing: 1px;
        }

        .team-avatar {
            width: 90px;
            height: 90px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            color: white;
            font-size: 2.2rem;
            transition: all 0.3s ease;
        }

        .team-avatar.lead-avatar {
            background: linear-gradient(135deg, #f39c12, #e67e22);
            width: 100px;
            height: 100px;
            font-size: 2.5rem;
        }

        .team-member:hover .team-avatar {
            transform: scale(1.1) rotate(5deg);
        }

        .team-role {
            color: #3498db;
            font-weight: 600;
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }

        .team-contact {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #ecf0f1;
        }

        .team-contact a {
            color: #27ae60;
            text-decoration: none;
            font-weight: 600;
        }

        .team-contact a:hover {
            color: #229954;
        }

        /* Mobile Responsiveness */
        @media (max-width: 768px) {
            .main-container {
                margin: 10px;
                padding: 20px;
                border-radius: 15px;
            }

            .lead-header {
                flex-direction: column;
                text-align: center;
            }

            .lead-avatar-large {
                margin-right: 0;
                margin-bottom: 1rem;
            }

            .contact-methods {
                grid-template-columns: 1fr;
            }

            .social-links {
                gap: 15px;
            }

            .social-link {
                width: 70px;
                height: 70px;
                font-size: 1.5rem;
            }

            .team-grid {
                grid-template-columns: 1fr;
                gap: 20px;
            }
        }

        /* Dark theme adjustments */
        .dark-theme .main-container {
            background: rgba(44, 62, 80, 0.95);
        }

        .dark-theme .contact-method {
            background: rgba(52, 73, 94, 0.9);
        }

        .dark-theme .team-member {
            background: rgba(52, 73, 94, 0.9);
        }

        .dark-theme .contact-form {
            background: linear-gradient(135deg, rgba(52, 73, 94, 0.8), rgba(44, 62, 80, 0.8));
        }
    </style>
    <script src="{{ url_for('static', filename='theme-manager.js') }}" defer></script>
    <script src="{{ url_for('static', filename='advanced-loader.js') }}" defer></script>
    <script src="{{ url_for('static', filename='interactive-enhancements.js') }}" defer></script>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-light">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('home') }}">
                <i class="fas fa-brain me-2"></i>AI Deepfake Detector
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('home') }}">
                            <i class="fas fa-home me-1"></i>Home
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('gallery') }}">
                            <i class="fas fa-images me-1"></i>Gallery
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('realtime') }}">
                            <i class="fas fa-video me-1"></i>Real-time
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('batch') }}">
                            <i class="fas fa-layer-group me-1"></i>Batch
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('api') }}">
                            <i class="fas fa-code me-1"></i>API
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('training') }}">
                            <i class="fas fa-cogs me-1"></i>Training
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('statistics') }}">
                            <i class="fas fa-chart-bar me-1"></i>Statistics
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('documentation') }}">
                            <i class="fas fa-book me-1"></i>Docs
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('about') }}">
                            <i class="fas fa-info-circle me-1"></i>About
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" href="{{ url_for('contact') }}">
                            <i class="fas fa-envelope me-1"></i>Contact
                        </a>
                    </li>
                </ul>
                <button class="theme-toggle" id="themeToggle" title="Toggle Theme">
                    <i class="fas fa-moon" id="themeIcon"></i>
                </button>
            </div>
        </div>
    </nav>

    <div class="container-fluid">
        <div class="main-container animate__animated animate__fadeIn">
            <!-- Header -->
            <div class="text-center mb-5">
                <h1 class="display-4 mb-3" style="background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                    <i class="fas fa-envelope me-3"></i>Contact Us
                </h1>
                <p class="lead text-muted">Connect with our AI research team for support, collaboration, and innovation</p>
            </div>

            <!-- Project Lead Contact -->
            <div class="lead-contact-section animate__animated animate__fadeInUp">
                <div class="lead-header">
                    <div class="lead-avatar-large">
                        <i class="fas fa-user-tie"></i>
                    </div>
                    <div class="lead-info-main">
                        <h2>Omkar Digambar</h2>
                        <p class="lead-title">Project Lead & AI Research Engineer</p>
                        <p class="lead-company">Kalagi Group of Companies</p>
                    </div>
                </div>

                <div class="contact-methods">
                    <div class="contact-method">
                        <div class="contact-icon">
                            <i class="fas fa-phone"></i>
                        </div>
                        <div class="contact-details">
                            <h5>Direct Phone</h5>
                            <a href="tel:+917624828106" class="contact-link">+91 7624828106</a>
                            <p class="text-muted">Available Mon-Sat, 9 AM - 8 PM IST</p>
                        </div>
                    </div>

                    <div class="contact-method">
                        <div class="contact-icon">
                            <i class="fas fa-envelope"></i>
                        </div>
                        <div class="contact-details">
                            <h5>Email</h5>
                            <a href="mailto:omkardigambar4@gmail.com" class="contact-link">omkardigambar4@gmail.com</a>
                            <p class="text-muted">Response within 24 hours</p>
                        </div>
                    </div>

                    <div class="contact-method">
                        <div class="contact-icon">
                            <i class="fas fa-building"></i>
                        </div>
                        <div class="contact-details">
                            <h5>Organization</h5>
                            <span class="contact-link">Kalagi Group of Companies</span>
                            <p class="text-muted">AI Research & Development Division</p>
                        </div>
                    </div>
                </div>
            </div>'''
    
            <!-- Social Media Section -->
            <div class="social-media-section animate__animated animate__fadeInUp">
                <h3><i class="fas fa-share-alt me-2"></i>Connect with Omkar</h3>
                <p class="text-muted">Follow and connect on social media platforms</p>

                <div class="social-links">
                    <a href="https://wa.me/917624828106" class="social-link whatsapp" target="_blank" title="WhatsApp">
                        <i class="fab fa-whatsapp"></i>
                        <span>WhatsApp</span>
                    </a>
                    <a href="https://www.linkedin.com/in/omkardigambar/" class="social-link linkedin" target="_blank" title="LinkedIn">
                        <i class="fab fa-linkedin-in"></i>
                        <span>LinkedIn</span>
                    </a>
                    <a href="https://github.com/OmkarKalagi" class="social-link github" target="_blank" title="GitHub">
                        <i class="fab fa-github"></i>
                        <span>GitHub</span>
                    </a>
                    <a href="https://omkarkalagi.github.io/portfolio/" class="social-link portfolio" target="_blank" title="Portfolio">
                        <i class="fas fa-briefcase"></i>
                        <span>Portfolio</span>
                    </a>
                    <a href="https://www.youtube.com/c/OmkarKalagi" class="social-link youtube" target="_blank" title="YouTube">
                        <i class="fab fa-youtube"></i>
                        <span>YouTube</span>
                    </a>
                    <a href="https://t.me/omkar_kalagi" class="social-link telegram" target="_blank" title="Telegram">
                        <i class="fab fa-telegram-plane"></i>
                        <span>Telegram</span>
                    </a>
                </div>
            </div>

            <!-- Contact Form with Formspree -->
            <div class="contact-form animate__animated animate__fadeInUp">
                <h3><i class="fas fa-paper-plane me-2"></i>Send us a Message</h3>
                <p class="text-muted mb-4">Have a question or want to collaborate? Send us a message and we'll get back to you soon!</p>

                <form action="https://formspree.io/f/YOUR_FORM_ID" method="POST" id="contactForm">
                    <div class="row">
                        <div class="col-md-6 mb-3">
                            <label for="firstName" class="form-label">First Name *</label>
                            <input type="text" class="form-control" id="firstName" name="firstName" required>
                        </div>
                        <div class="col-md-6 mb-3">
                            <label for="lastName" class="form-label">Last Name *</label>
                            <input type="text" class="form-control" id="lastName" name="lastName" required>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-md-6 mb-3">
                            <label for="email" class="form-label">Email Address *</label>
                            <input type="email" class="form-control" id="email" name="email" required>
                        </div>
                        <div class="col-md-6 mb-3">
                            <label for="phone" class="form-label">Phone Number</label>
                            <input type="tel" class="form-control" id="phone" name="phone">
                        </div>
                    </div>
                    <div class="mb-3">
                        <label for="subject" class="form-label">Subject *</label>
                        <select class="form-control" id="subject" name="subject" required>
                            <option value="">Select a subject</option>
                            <option value="technical-support">Technical Support</option>
                            <option value="api-questions">API Questions</option>
                            <option value="feature-request">Feature Request</option>
                            <option value="business-inquiry">Business Inquiry</option>
                            <option value="partnership">Partnership</option>
                            <option value="collaboration">Collaboration</option>
                            <option value="other">Other</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label for="message" class="form-label">Message *</label>
                        <textarea class="form-control" id="message" name="message" rows="5" placeholder="Please describe your question or inquiry in detail..." required></textarea>
                    </div>
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="privacy" name="privacy" required>
                            <label class="form-check-label" for="privacy">
                                I agree to the privacy policy and terms of service *
                            </label>
                        </div>
                    </div>
                    <button type="submit" class="btn-submit">
                        <i class="fas fa-paper-plane me-2"></i>Send Message
                    </button>
                </form>
            </div>

            <!-- Team Section -->
            <div class="animate__animated animate__fadeInUp">
                <h3><i class="fas fa-users me-2"></i>Meet Our Development Team</h3>
                <p class="text-muted mb-4">Our talented team of AI researchers and developers working on cutting-edge deepfake detection technology</p>

                <div class="team-grid">
                    <div class="team-member lead-member">
                        <div class="team-avatar lead-avatar">
                            <i class="fas fa-crown"></i>
                        </div>
                        <h5>Omkar Digambar</h5>
                        <p class="team-role">Project Lead & AI Research Engineer</p>
                        <small>Leading the development of advanced deepfake detection algorithms and AI research initiatives</small>
                        <div class="team-contact">
                            <a href="tel:+917624828106"><i class="fas fa-phone"></i> +91 7624828106</a>
                        </div>
                    </div>

                    <div class="team-member">
                        <div class="team-avatar">
                            <i class="fas fa-laptop-code"></i>
                        </div>
                        <h5>VijayKumar Koujageri</h5>
                        <p class="team-role">Senior AI Developer</p>
                        <small>Specializing in machine learning model optimization and deployment strategies</small>
                    </div>

                    <div class="team-member">
                        <div class="team-avatar">
                            <i class="fas fa-brain"></i>
                        </div>
                        <h5>Bharat Nayaka</h5>
                        <p class="team-role">Deep Learning Engineer</p>
                        <small>Expert in neural network architectures and computer vision applications</small>
                    </div>

                    <div class="team-member">
                        <div class="team-avatar">
                            <i class="fas fa-code"></i>
                        </div>
                        <h5>Pratham V Mandre</h5>
                        <p class="team-role">Full-Stack Developer</p>
                        <small>Building robust web applications and intuitive user interfaces</small>
                    </div>

                    <div class="team-member">
                        <div class="team-avatar">
                            <i class="fas fa-database"></i>
                        </div>
                        <h5>Manoj Kengalagutti</h5>
                        <p class="team-role">Data Engineer</p>
                        <small>Managing data pipelines and model training infrastructure</small>
                    </div>

                    <div class="team-member">
                        <div class="team-avatar">
                            <i class="fas fa-cogs"></i>
                        </div>
                        <h5>Ajay N</h5>
                        <p class="team-role">DevOps Engineer</p>
                        <small>Ensuring scalable deployment and system reliability</small>
                    </div>

                    <div class="team-member">
                        <div class="team-avatar">
                            <i class="fas fa-shield-alt"></i>
                        </div>
                        <h5>Adarsh G Shet</h5>
                        <p class="team-role">Security & QA Engineer</p>
                        <small>Maintaining system security and comprehensive quality assurance</small>
                    </div>

                    <div class="team-member">
                        <div class="team-avatar">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <h5>Basavaraj Bolashetti</h5>
                        <p class="team-role">Data Scientist</p>
                        <small>Advanced statistical analysis and machine learning research</small>
                    </div>

                    <div class="team-member">
                        <div class="team-avatar">
                            <i class="fas fa-network-wired"></i>
                        </div>
                        <h5>Niranjan S Chanal</h5>
                        <p class="team-role">Systems Architect</p>
                        <small>Designing scalable system architecture and integration solutions</small>
                    </div>
                </div>
            </div>
        </div>
    </div>

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
                    <li><a href="mailto:omkardigambar4@gmail.com">Email Support</a></li>
                </ul>
            </div>
            <div class="footer-section">
                <h6>Connect</h6>
                <div class="social-icons">
                    <a href="https://github.com/OmkarKalagi" class="social-icon" target="_blank"><i class="fab fa-github"></i></a>
                    <a href="https://www.linkedin.com/in/omkardigambar/" class="social-icon" target="_blank"><i class="fab fa-linkedin"></i></a>
                    <a href="https://www.youtube.com/c/OmkarKalagi" class="social-icon" target="_blank"><i class="fab fa-youtube"></i></a>
                    <a href="https://t.me/omkar_kalagi" class="social-icon" target="_blank"><i class="fab fa-telegram"></i></a>
                </div>
                <p class="footer-copyright">
                    © 2024 Kalagi Group of Companies<br>
                    All Rights Reserved
                </p>
            </div>
        </div>
    </footer>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

    <!-- Form Submission Script -->
    <script>
        document.getElementById('contactForm').addEventListener('submit', function(e) {
            e.preventDefault();

            // Show loading
            const submitBtn = document.querySelector('.btn-submit');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Sending...';
            submitBtn.disabled = true;

            // Get form data
            const formData = new FormData(this);

            // Submit to Formspree (replace YOUR_FORM_ID with actual Formspree form ID)
            fetch('https://formspree.io/f/YOUR_FORM_ID', {
                method: 'POST',
                body: formData,
                headers: {
                    'Accept': 'application/json'
                }
            })
            .then(response => {
                if (response.ok) {
                    // Success
                    submitBtn.innerHTML = '<i class="fas fa-check me-2"></i>Message Sent!';
                    submitBtn.style.background = 'linear-gradient(135deg, #27ae60, #2ecc71)';

                    // Reset form
                    this.reset();

                    // Show success message
                    alert('Thank you! Your message has been sent successfully. We will get back to you soon.');

                    // Reset button after 3 seconds
                    setTimeout(() => {
                        submitBtn.innerHTML = originalText;
                        submitBtn.style.background = '';
                        submitBtn.disabled = false;
                    }, 3000);
                } else {
                    throw new Error('Network response was not ok');
                }
            })
            .catch(error => {
                // Error
                submitBtn.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i>Error - Try Again';
                submitBtn.style.background = 'linear-gradient(135deg, #e74c3c, #c0392b)';

                alert('Sorry, there was an error sending your message. Please try again or contact us directly.');

                // Reset button after 3 seconds
                setTimeout(() => {
                    submitBtn.innerHTML = originalText;
                    submitBtn.style.background = '';
                    submitBtn.disabled = false;
                }, 3000);
            });
        });
    </script>
</body>
</html>'''

    # Save the complete contact page
    contact_file = Path('templates/contact_new.html')
    with open(contact_file, 'w', encoding='utf-8') as f:
        f.write(contact_html)

    print("✅ Created complete enhanced contact page")
    return True

def main():
    """Create enhanced contact page."""
    print("🎨 Creating Enhanced Contact Page with Real Social Links")
    print("=" * 60)
    
    create_enhanced_contact_page()
    
    print("✅ Enhanced contact page creation started!")
    print("📧 Formspree integration ready")
    print("🔗 Real social media links included")
    print("👥 Team members updated")

if __name__ == "__main__":
    main()
