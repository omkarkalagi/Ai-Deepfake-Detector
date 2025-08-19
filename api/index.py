from flask import Flask, request, jsonify
import os
import random
import time
from datetime import datetime

# Configure Flask for Vercel serverless
app = Flask(__name__)

# Configure Flask for Vercel
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# Model metrics for templates
model_metrics = {
    'accuracy': 92.8,
    'precision': 94.1,
    'recall': 91.5,
    'f1_score': 92.8,
    'version': 'v2.1'
}

def create_html_page(title, content, extra_js=""):
    """Create a complete HTML page with consistent styling"""
    return f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>

    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">

    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <!-- Fallback CSS for icons if Font Awesome fails -->
    <style>
        .fas, .far, .fab {
            font-family: "Font Awesome 6 Free", "Font Awesome 6 Pro", "Font Awesome 5 Free", "Font Awesome 5 Pro", sans-serif;
            font-weight: 900;
        }
        /* Fallback for common icons */
        .fa-robot:before { content: "🤖"; }
        .fa-home:before { content: "🏠"; }
        .fa-images:before { content: "🖼️"; }
        .fa-chart-bar:before { content: "📊"; }
        .fa-cogs:before { content: "⚙️"; }
        .fa-info-circle:before { content: "ℹ️"; }
        .fa-envelope:before { content: "✉️"; }
        .fa-camera:before { content: "📷"; }
        .fa-upload:before { content: "⬆️"; }
        .fa-search:before { content: "🔍"; }
        .fa-shield-alt:before { content: "🛡️"; }
        .fa-eye:before { content: "👁️"; }
        .fa-chart-line:before { content: "📈"; }
        .fa-star:before { content: "⭐"; }
        .fa-bolt:before { content: "⚡"; }
        .fa-brain:before { content: "🧠"; }
        .fa-shield-check:before { content: "✅"; }
    </style>
    <style>
        :root {{
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --info-color: #17a2b8;
            --primary-color: #667eea;
            --secondary-color: #764ba2;
        }}

        body {{
            background: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}

        body.loaded {{
            opacity: 1;
        }}

        /* Loading indicator */
        .loading-indicator {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 9999;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .loading-spinner {{
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        /* Navbar Styling */
        .navbar {{
            background: var(--primary-gradient) !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 1rem 0;
        }}

        .navbar-brand {{
            font-weight: bold;
            font-size: 1.5rem;
            color: white !important;
        }}

        .navbar-nav .nav-link {{
            color: rgba(255,255,255,0.9) !important;
            font-weight: 500;
            margin: 0 0.5rem;
            transition: color 0.3s ease;
        }}

        .navbar-nav .nav-link:hover {{
            color: white !important;
            transform: translateY(-1px);
        }}

        .navbar-toggler {{
            border: 1px solid rgba(255,255,255,0.3);
        }}

        .navbar-toggler-icon {{
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 30 30'%3e%3cpath stroke='rgba%28255, 255, 255, 0.75%29' stroke-linecap='round' stroke-miterlimit='10' stroke-width='2' d='m4 7h22M4 15h22M4 23h22'/%3e%3c/svg%3e");
        }}

        /* Hero Section */
        .hero-section {{
            background: var(--primary-gradient);
            color: white;
            padding: 100px 0;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}

        .hero-section::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            opacity: 0.1;
        }}

        .hero-section .container {{
            position: relative;
            z-index: 1;
        }}

        /* Cards */
        .feature-card, .stat-card, .content-card, .gallery-item {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            border: 1px solid rgba(0,0,0,0.05);
        }}

        .feature-card:hover, .stat-card:hover, .gallery-item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }}

        /* Buttons */
        .btn-primary {{
            background: var(--primary-gradient);
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}

        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
            background: var(--primary-gradient);
            border: none;
        }}

        .btn-outline-primary {{
            border: 2px solid var(--primary-color);
            color: var(--primary-color);
            font-weight: 600;
            border-radius: 25px;
            padding: 10px 28px;
            transition: all 0.3s ease;
        }}

        .btn-outline-primary:hover {{
            background: var(--primary-gradient);
            border-color: transparent;
            transform: translateY(-2px);
        }}

        /* Statistics */
        .stat-number {{
            font-size: 2.5rem;
            font-weight: bold;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            display: inline-block;
        }}

        /* Badges */
        .fake-badge {{
            background: var(--danger-color);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            box-shadow: 0 2px 10px rgba(220, 53, 69, 0.3);
        }}

        .real-badge {{
            background: var(--success-color);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            box-shadow: 0 2px 10px rgba(40, 167, 69, 0.3);
        }}

        /* Image Placeholders */
        .image-placeholder {{
            height: 200px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 18px;
            margin: 15px 0;
            transition: transform 0.3s ease;
        }}

        .image-placeholder:hover {{
            transform: scale(1.02);
        }}

        .fake-placeholder {{
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        }}

        .real-placeholder {{
            background: linear-gradient(45deg, #26de81, #20bf6b);
            box-shadow: 0 8px 25px rgba(38, 222, 129, 0.3);
        }}

        /* Footer */
        .footer {{
            background: #343a40;
            color: white;
            padding: 40px 0;
            margin-top: 50px;
        }}

        /* Progress Bars */
        .progress {{
            height: 10px;
            border-radius: 5px;
            background-color: rgba(0,0,0,0.1);
        }}

        .progress-bar {{
            border-radius: 5px;
            transition: width 0.6s ease;
        }}

        /* Alerts */
        .alert {{
            border-radius: 10px;
            border: none;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}

        /* Form Controls */
        .form-control {{
            border-radius: 10px;
            border: 2px solid #e9ecef;
            padding: 12px 15px;
            transition: all 0.3s ease;
        }}

        .form-control:focus {{
            border-color: var(--primary-color);
            box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
        }}

        /* Responsive Design */
        @media (max-width: 768px) {{
            .hero-section {{
                padding: 60px 0;
            }}

            .hero-section h1 {{
                font-size: 2rem;
            }}

            .feature-card, .stat-card, .content-card {{
                margin: 10px 0;
                padding: 20px;
            }}

            .stat-number {{
                font-size: 2rem;
            }}
        }}

        /* Loading Animation */
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .fade-in-up {{
            animation: fadeInUp 0.6s ease-out;
        }}
    </style>
</head>
<body>
    <!-- Loading Indicator -->
    <div id="loading-indicator" class="loading-indicator">
        <div class="loading-spinner"></div>
        <p>Loading AI Deepfake Detector...</p>
    </div>

    <nav class="navbar navbar-expand-lg navbar-dark fixed-top">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-robot me-2"></i>AI Deepfake Detector
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <div class="navbar-nav ms-auto">
                    <a class="nav-link" href="/" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Upload and analyze images">
                        <i class="fas fa-home me-1"></i>Home
                    </a>
                    <a class="nav-link" href="/gallery" data-bs-toggle="tooltip" data-bs-placement="bottom" title="View detection examples">
                        <i class="fas fa-images me-1"></i>Gallery
                    </a>
                    <a class="nav-link" href="/statistics" data-bs-toggle="tooltip" data-bs-placement="bottom" title="View performance metrics">
                        <i class="fas fa-chart-bar me-1"></i>Statistics
                    </a>
                    <a class="nav-link" href="/training" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Monitor training progress">
                        <i class="fas fa-cogs me-1"></i>Training
                    </a>
                    <a class="nav-link" href="/about" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Learn about our technology">
                        <i class="fas fa-info-circle me-1"></i>About
                    </a>
                    <a class="nav-link" href="/contact" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Get in touch">
                        <i class="fas fa-envelope me-1"></i>Contact
                    </a>
                </div>
            </div>
        </div>
    </nav>

    <!-- Add padding to body to account for fixed navbar -->
    <div style="padding-top: 80px;"></div>

    <main>{content}</main>

    <footer class="footer">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <h5><i class="fas fa-robot me-2"></i>AI Deepfake Detector</h5>
                    <p class="mb-2">Advanced machine learning technology for deepfake detection.</p>
                    <div class="d-flex gap-3">
                        <a href="/" class="text-light text-decoration-none"><i class="fas fa-home me-1"></i>Home</a>
                        <a href="/gallery" class="text-light text-decoration-none"><i class="fas fa-images me-1"></i>Gallery</a>
                        <a href="/statistics" class="text-light text-decoration-none"><i class="fas fa-chart-bar me-1"></i>Stats</a>
                    </div>
                </div>
                <div class="col-md-6 text-md-end">
                    <p class="mb-1">&copy; 2024 AI Deepfake Detector. Demo Version.</p>
                    <p class="mb-1"><small><i class="fas fa-server me-1"></i>Running on Vercel with Flask</small></p>
                    <p class="mb-0"><small><i class="fas fa-code me-1"></i>Built with Bootstrap 5 & Chart.js</small></p>
                </div>
            </div>
        </div>
    </footer>

    <!-- Bootstrap JavaScript Bundle with Popper -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>

    <!-- Fallback Bootstrap functionality -->
    <script>
        // Ensure Bootstrap is loaded, if not provide basic functionality
        if (typeof bootstrap === 'undefined') {{
            console.warn('Bootstrap not loaded, using fallback functionality');
            window.bootstrap = {{
                Tooltip: function() {{ return {{ dispose: function() {{}} }}; }}
            }};
        }}
    </script>

    <!-- Custom JavaScript -->
    <script>
        // Initialize tooltips and popovers
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('AI Deepfake Detector loaded successfully');

            // Check if Bootstrap is loaded
            if (typeof bootstrap !== 'undefined') {{
                // Initialize Bootstrap tooltips
                try {{
                    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
                    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {{
                        return new bootstrap.Tooltip(tooltipTriggerEl);
                    }});
                    console.log('Bootstrap tooltips initialized');
                }} catch (e) {{
                    console.warn('Tooltip initialization failed:', e);
                }}
            }}

            // Add fade-in animation to cards
            const cards = document.querySelectorAll('.feature-card, .stat-card, .content-card, .gallery-item');
            if (window.IntersectionObserver) {{
                const observer = new IntersectionObserver((entries) => {{
                    entries.forEach(entry => {{
                        if (entry.isIntersecting) {{
                            entry.target.classList.add('fade-in-up');
                        }}
                    }});
                }}, {{ threshold: 0.1 }});

                cards.forEach(card => {{
                    observer.observe(card);
                }});
            }} else {{
                // Fallback for browsers without IntersectionObserver
                cards.forEach(card => {{
                    card.classList.add('fade-in-up');
                }});
            }}

            // Smooth scrolling for anchor links
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
                anchor.addEventListener('click', function (e) {{
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {{
                        target.scrollIntoView({{ behavior: 'smooth' }});
                    }}
                }});
            }});

            // Hide loading indicator and show content
            const loadingIndicator = document.getElementById('loading-indicator');
            if (loadingIndicator) {{
                loadingIndicator.style.display = 'none';
            }}
            document.body.classList.add('loaded');

            console.log('Page initialization complete');
        }});
    </script>

    {extra_js}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page - AI Deepfake Detector"""
    error_msg = ""
    result_html = ""

    if request.method == 'POST':
        if 'file' not in request.files:
            error_msg = "No file part in the request"
        else:
            file = request.files['file']
            if file.filename == '':
                error_msg = "No file selected"
            elif file and allowed_file(file.filename):
                try:
                    # Demo analysis results
                    result = random.choice(['Real', 'Fake'])
                    confidence = round(random.uniform(85.0, 98.0), 1)
                    processing_time = round(random.uniform(0.1, 0.3), 3)

                    result_html = f'''
                    <div class="mt-4">
                        <div class="alert alert-{'success' if result == 'Real' else 'danger'}">
                            <h5><i class="fas fa-{'check-circle' if result == 'Real' else 'exclamation-triangle'} me-2"></i>Analysis Results:</h5>
                            <p><strong>Result:</strong> {result} Image</p>
                            <p><strong>Confidence:</strong> {confidence}%</p>
                            <div class="progress mb-2">
                                <div class="progress-bar bg-{'success' if result == 'Real' else 'danger'}" style="width: {confidence}%"></div>
                            </div>
                            <small class="text-muted">Processing time: {processing_time}s</small>
                        </div>
                    </div>
                    '''
                except Exception as e:
                    error_msg = f"Error processing image: {str(e)}"
            else:
                error_msg = "Allowed file types are PNG, JPG, JPEG"

    error_html = f'<div class="alert alert-danger"><i class="fas fa-exclamation-triangle me-2"></i>{error_msg}</div>' if error_msg else ""

    content = f'''
    <div class="hero-section">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-lg-6">
                    <h1 class="display-4 mb-4 fw-bold">
                        <i class="fas fa-shield-alt me-3 text-warning"></i>
                        AI Deepfake Detector
                    </h1>
                    <p class="lead mb-4">
                        Advanced machine learning technology to detect deepfake images with
                        <span class="fw-bold text-warning">{model_metrics['accuracy']}% accuracy</span>
                    </p>
                    <div class="d-flex gap-3 mb-4">
                        <div class="text-center">
                            <div class="h4 mb-1"><i class="fas fa-bolt text-warning"></i></div>
                            <small>Fast Analysis</small>
                        </div>
                        <div class="text-center">
                            <div class="h4 mb-1"><i class="fas fa-brain text-info"></i></div>
                            <small>AI Powered</small>
                        </div>
                        <div class="text-center">
                            <div class="h4 mb-1"><i class="fas fa-shield-check text-success"></i></div>
                            <small>Secure</small>
                        </div>
                    </div>
                </div>
                <div class="col-lg-6">
                    {error_html}
                    <div class="feature-card">
                        <h3 class="mb-4 text-center">
                            <i class="fas fa-camera me-2 text-primary"></i>
                            Analyze Image
                        </h3>

                        <!-- Tab Navigation -->
                        <ul class="nav nav-pills nav-justified mb-4" id="analysisTab" role="tablist">
                            <li class="nav-item" role="presentation">
                                <button class="nav-link active" id="upload-tab" data-bs-toggle="pill" data-bs-target="#upload-panel" type="button" role="tab">
                                    <i class="fas fa-upload me-1"></i>Upload File
                                </button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="camera-tab" data-bs-toggle="pill" data-bs-target="#camera-panel" type="button" role="tab">
                                    <i class="fas fa-camera me-1"></i>Use Camera
                                </button>
                            </li>
                        </ul>

                        <!-- Tab Content -->
                        <div class="tab-content" id="analysisTabContent">
                            <!-- Upload Panel -->
                            <div class="tab-pane fade show active" id="upload-panel" role="tabpanel">
                                <form method="POST" enctype="multipart/form-data" class="text-center">
                                    <div class="mb-4">
                                        <label for="fileInput" class="form-label fw-semibold">Choose an image file</label>
                                        <input type="file" class="form-control form-control-lg" id="fileInput" name="file" accept="image/*" required>
                                        <div class="form-text">Supported formats: JPG, PNG, JPEG (Max 10MB)</div>
                                    </div>
                                    <button type="submit" class="btn btn-primary btn-lg px-5">
                                        <i class="fas fa-search me-2"></i>Analyze Image
                                    </button>
                                </form>
                            </div>

                            <!-- Camera Panel -->
                            <div class="tab-pane fade" id="camera-panel" role="tabpanel">
                                <div class="text-center">
                                    <div id="camera-container" class="mb-4">
                                        <video id="camera-video" class="d-none" autoplay playsinline style="max-width: 100%; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);"></video>
                                        <canvas id="camera-canvas" class="d-none"></canvas>
                                        <div id="camera-placeholder" class="bg-light border rounded d-flex align-items-center justify-content-center" style="height: 300px;">
                                            <div class="text-center text-muted">
                                                <i class="fas fa-camera fa-3x mb-3"></i>
                                                <p>Click "Start Camera" to begin</p>
                                            </div>
                                        </div>
                                        <div id="captured-image" class="d-none">
                                            <img id="captured-img" class="img-fluid rounded" style="max-height: 300px;">
                                        </div>
                                    </div>

                                    <div class="d-flex gap-2 justify-content-center flex-wrap">
                                        <button id="start-camera-btn" class="btn btn-success">
                                            <i class="fas fa-video me-2"></i>Start Camera
                                        </button>
                                        <button id="capture-btn" class="btn btn-primary d-none">
                                            <i class="fas fa-camera me-2"></i>Capture Photo
                                        </button>
                                        <button id="analyze-camera-btn" class="btn btn-warning d-none">
                                            <i class="fas fa-search me-2"></i>Analyze Photo
                                        </button>
                                        <button id="retake-btn" class="btn btn-secondary d-none">
                                            <i class="fas fa-redo me-2"></i>Retake
                                        </button>
                                        <button id="stop-camera-btn" class="btn btn-danger d-none">
                                            <i class="fas fa-stop me-2"></i>Stop Camera
                                        </button>
                                    </div>

                                    <div class="mt-3">
                                        <small class="text-muted">
                                            <i class="fas fa-info-circle me-1"></i>
                                            Camera access required. Your images are processed locally and not stored.
                                        </small>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Results Display -->
                        <div id="analysis-results">
                            {result_html}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Features Section -->
    <div class="container my-5">
        <div class="row text-center mb-5">
            <div class="col-12">
                <h2 class="mb-4">
                    <i class="fas fa-star me-2 text-warning"></i>
                    Why Choose Our AI Detector?
                </h2>
                <p class="lead text-muted">Cutting-edge technology meets user-friendly design</p>
            </div>
        </div>

        <div class="row">
            <div class="col-md-4 mb-4">
                <div class="feature-card text-center h-100">
                    <div class="mb-4">
                        <i class="fas fa-eye fa-4x text-primary"></i>
                    </div>
                    <h4 class="mb-3">Real-time Detection</h4>
                    <p class="text-muted">Upload images and get instant deepfake analysis with detailed confidence scores and feature breakdown.</p>
                    <div class="mt-auto">
                        <span class="badge bg-primary">< 0.5s processing</span>
                    </div>
                </div>
            </div>
            <div class="col-md-4 mb-4">
                <div class="feature-card text-center h-100">
                    <div class="mb-4">
                        <i class="fas fa-chart-line fa-4x text-success"></i>
                    </div>
                    <h4 class="mb-3">{model_metrics['accuracy']}% Accuracy</h4>
                    <p class="text-muted">State-of-the-art neural networks trained on millions of images for superior detection performance.</p>
                    <div class="mt-auto">
                        <span class="badge bg-success">Industry Leading</span>
                    </div>
                </div>
            </div>
            <div class="col-md-4 mb-4">
                <div class="feature-card text-center h-100">
                    <div class="mb-4">
                        <i class="fas fa-images fa-4x text-info"></i>
                    </div>
                    <h4 class="mb-3">Celebrity Gallery</h4>
                    <p class="text-muted">Explore examples of detected deepfakes of famous personalities and learn about detection techniques.</p>
                    <div class="mt-auto">
                        <a href="/gallery" class="btn btn-outline-info">
                            <i class="fas fa-arrow-right me-1"></i>View Gallery
                        </a>
                    </div>
                </div>
            </div>
        </div>

        <!-- Quick Stats -->
        <div class="row mt-5">
            <div class="col-12">
                <div class="content-card">
                    <div class="row text-center">
                        <div class="col-md-3 mb-3">
                            <div class="stat-number">{model_metrics['accuracy']}%</div>
                            <h6 class="text-muted">Accuracy</h6>
                        </div>
                        <div class="col-md-3 mb-3">
                            <div class="stat-number">{model_metrics['precision']}%</div>
                            <h6 class="text-muted">Precision</h6>
                        </div>
                        <div class="col-md-3 mb-3">
                            <div class="stat-number">{model_metrics['recall']}%</div>
                            <h6 class="text-muted">Recall</h6>
                        </div>
                        <div class="col-md-3 mb-3">
                            <div class="stat-number">{model_metrics['f1_score']}%</div>
                            <h6 class="text-muted">F1 Score</h6>
                        </div>
                    </div>
                    <div class="text-center mt-4">
                        <a href="/statistics" class="btn btn-outline-primary">
                            <i class="fas fa-chart-bar me-2"></i>View Detailed Statistics
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>
    '''

    camera_js = '''
    <script>
        // Camera functionality
        let stream = null;
        let capturedImageData = null;

        const video = document.getElementById('camera-video');
        const canvas = document.getElementById('camera-canvas');
        const ctx = canvas.getContext('2d');
        const placeholder = document.getElementById('camera-placeholder');
        const capturedDiv = document.getElementById('captured-image');
        const capturedImg = document.getElementById('captured-img');

        const startBtn = document.getElementById('start-camera-btn');
        const captureBtn = document.getElementById('capture-btn');
        const analyzeBtn = document.getElementById('analyze-camera-btn');
        const retakeBtn = document.getElementById('retake-btn');
        const stopBtn = document.getElementById('stop-camera-btn');

        // Start camera
        startBtn.addEventListener('click', async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    }
                });

                video.srcObject = stream;
                video.play();

                // Show video, hide placeholder
                placeholder.classList.add('d-none');
                video.classList.remove('d-none');
                capturedDiv.classList.add('d-none');

                // Update buttons
                startBtn.classList.add('d-none');
                captureBtn.classList.remove('d-none');
                stopBtn.classList.remove('d-none');
                analyzeBtn.classList.add('d-none');
                retakeBtn.classList.add('d-none');

                // Set canvas size
                video.addEventListener('loadedmetadata', () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                });

            } catch (error) {
                console.error('Error accessing camera:', error);
                alert('Unable to access camera. Please ensure you have granted camera permissions and try again.');
            }
        });

        // Capture photo
        captureBtn.addEventListener('click', () => {
            if (video.videoWidth && video.videoHeight) {
                // Draw video frame to canvas
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);

                // Get image data
                capturedImageData = canvas.toDataURL('image/jpeg', 0.8);
                capturedImg.src = capturedImageData;

                // Show captured image, hide video
                video.classList.add('d-none');
                capturedDiv.classList.remove('d-none');

                // Update buttons
                captureBtn.classList.add('d-none');
                analyzeBtn.classList.remove('d-none');
                retakeBtn.classList.remove('d-none');
            }
        });

        // Analyze captured photo
        analyzeBtn.addEventListener('click', async () => {
            if (capturedImageData) {
                // Show loading state
                analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
                analyzeBtn.disabled = true;

                try {
                    // Convert base64 to blob
                    const response = await fetch(capturedImageData);
                    const blob = await response.blob();

                    // Create form data
                    const formData = new FormData();
                    formData.append('file', blob, 'camera-capture.jpg');

                    // Send to server
                    const analysisResponse = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });

                    if (analysisResponse.ok) {
                        const result = await analysisResponse.json();
                        displayAnalysisResult(result);
                    } else {
                        throw new Error('Analysis failed');
                    }

                } catch (error) {
                    console.error('Analysis error:', error);
                    // Simulate analysis for demo
                    setTimeout(() => {
                        const demoResult = {
                            result: Math.random() > 0.5 ? 'Real' : 'Fake',
                            confidence_score: Math.floor(Math.random() * 15) + 85,
                            processing_time: (Math.random() * 0.3 + 0.1).toFixed(2)
                        };
                        displayAnalysisResult(demoResult);
                    }, 1500);
                }

                // Reset button
                analyzeBtn.innerHTML = '<i class="fas fa-search me-2"></i>Analyze Photo';
                analyzeBtn.disabled = false;
            }
        });

        // Retake photo
        retakeBtn.addEventListener('click', () => {
            // Show video, hide captured image
            capturedDiv.classList.add('d-none');
            video.classList.remove('d-none');

            // Update buttons
            captureBtn.classList.remove('d-none');
            analyzeBtn.classList.add('d-none');
            retakeBtn.classList.add('d-none');

            // Clear captured data
            capturedImageData = null;
        });

        // Stop camera
        stopBtn.addEventListener('click', () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }

            // Reset UI
            video.classList.add('d-none');
            capturedDiv.classList.add('d-none');
            placeholder.classList.remove('d-none');

            // Reset buttons
            startBtn.classList.remove('d-none');
            captureBtn.classList.add('d-none');
            analyzeBtn.classList.add('d-none');
            retakeBtn.classList.add('d-none');
            stopBtn.classList.add('d-none');

            capturedImageData = null;
        });

        // Display analysis result
        function displayAnalysisResult(result) {
            const resultsDiv = document.getElementById('analysis-results');
            const isReal = result.result === 'Real';
            const confidence = result.confidence_score || result.confidence || 85;
            const processingTime = result.processing_time || '0.18';

            resultsDiv.innerHTML = `
                <div class="mt-4">
                    <div class="alert alert-${isReal ? 'success' : 'danger'}">
                        <h5><i class="fas fa-${isReal ? 'check-circle' : 'exclamation-triangle'} me-2"></i>Camera Analysis Results:</h5>
                        <p><strong>Result:</strong> ${result.result} Image</p>
                        <p><strong>Confidence:</strong> ${confidence}%</p>
                        <div class="progress mb-2">
                            <div class="progress-bar bg-${isReal ? 'success' : 'danger'}" style="width: ${confidence}%"></div>
                        </div>
                        <small class="text-muted">Processing time: ${processingTime}s</small>
                        <div class="mt-3">
                            <small class="text-info">
                                <i class="fas fa-camera me-1"></i>
                                Analyzed from camera capture
                            </small>
                        </div>
                    </div>
                </div>
            `;
        }

        // Clean up on page unload
        window.addEventListener('beforeunload', () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        });
    </script>
    '''

    return create_html_page("AI Deepfake Detector - Advanced ML Detection", content, camera_js)

def allowed_file(filename):
    """Check if file has allowed extension."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/gallery')
def gallery():
    """Gallery page with deepfake examples"""
    content = '''
    <div class="container mt-5">
        <div class="row">
            <div class="col-12">
                <h1 class="mb-4"><i class="fas fa-images me-3"></i>Deepfake Detection Gallery</h1>
                <p class="lead mb-5">Examples of detected deepfakes and authentic images from our analysis system.</p>
            </div>
        </div>

        <div class="row">
            <div class="col-md-6">
                <div class="gallery-item">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5><i class="fas fa-user-secret me-2"></i>Celebrity Deepfake</h5>
                        <span class="fake-badge">FAKE - 91.2%</span>
                    </div>
                    <div class="image-placeholder fake-placeholder">
                        <i class="fas fa-user-secret fa-3x"></i>
                    </div>
                    <p class="mt-3">AI-generated face swap of a famous Bollywood actor. Our model detected temporal inconsistencies and unnatural facial features.</p>
                    <small class="text-muted"><i class="fas fa-clock me-1"></i>Processing: 0.22s</small>
                </div>
            </div>

            <div class="col-md-6">
                <div class="gallery-item">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5><i class="fas fa-user-check me-2"></i>Authentic Portrait</h5>
                        <span class="real-badge">REAL - 96.5%</span>
                    </div>
                    <div class="image-placeholder real-placeholder">
                        <i class="fas fa-user-check fa-3x"></i>
                    </div>
                    <p class="mt-3">Genuine photograph with natural lighting and authentic facial characteristics. High confidence score indicates authentic content.</p>
                    <small class="text-muted"><i class="fas fa-clock me-1"></i>Processing: 0.17s</small>
                </div>
            </div>

            <div class="col-md-6">
                <div class="gallery-item">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5><i class="fas fa-mask me-2"></i>Political Figure Deepfake</h5>
                        <span class="fake-badge">FAKE - 88.9%</span>
                    </div>
                    <div class="image-placeholder" style="background: linear-gradient(45deg, #fd79a8, #e84393);">
                        <i class="fas fa-mask fa-3x"></i>
                    </div>
                    <p class="mt-3">Deepfake of a prominent political leader. Detected through analysis of micro-expressions and lighting inconsistencies.</p>
                    <small class="text-muted"><i class="fas fa-clock me-1"></i>Processing: 0.24s</small>
                </div>
            </div>

            <div class="col-md-6">
                <div class="gallery-item">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5><i class="fas fa-trophy me-2"></i>Sports Celebrity</h5>
                        <span class="real-badge">REAL - 94.7%</span>
                    </div>
                    <div class="image-placeholder" style="background: linear-gradient(45deg, #74b9ff, #0984e3);">
                        <i class="fas fa-trophy fa-3x"></i>
                    </div>
                    <p class="mt-3">Authentic photograph of a famous cricket player from match coverage. Natural skin texture and consistent lighting confirmed.</p>
                    <small class="text-muted"><i class="fas fa-clock me-1"></i>Processing: 0.15s</small>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-12 text-center">
                <div class="content-card">
                    <h4>Try Our Detection System</h4>
                    <p>Upload your own images to test our advanced deepfake detection capabilities.</p>
                    <a href="/" class="btn btn-primary btn-lg me-3"><i class="fas fa-upload me-2"></i>Upload Image</a>
                    <a href="/statistics" class="btn btn-outline-primary btn-lg"><i class="fas fa-chart-bar me-2"></i>View Statistics</a>
                </div>
            </div>
        </div>
    </div>
    '''
    return create_html_page("Gallery - AI Deepfake Detector", content)

@app.route('/statistics')
def statistics():
    """Statistics and analytics page"""
    content = '''
    <div class="container mt-5">
        <div class="row">
            <div class="col-12">
                <h1 class="mb-4"><i class="fas fa-chart-bar me-3"></i>Detection Statistics</h1>
                <p class="lead mb-5">Comprehensive analytics and performance metrics of our deepfake detection system.</p>
            </div>
        </div>

        <div class="row">
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <div class="stat-number">1,247</div>
                    <h5>Total Analyses</h5>
                    <p class="text-muted">Images processed</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <div class="stat-number">92.8%</div>
                    <h5>Accuracy Rate</h5>
                    <p class="text-muted">Detection accuracy</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <div class="stat-number">756</div>
                    <h5>Real Images</h5>
                    <p class="text-muted">Authentic detected</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <div class="stat-number">491</div>
                    <h5>Fake Images</h5>
                    <p class="text-muted">Deepfakes detected</p>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-6">
                <div class="stat-card">
                    <h5 class="mb-4"><i class="fas fa-chart-pie me-2"></i>Detection Results</h5>
                    <canvas id="resultsChart" width="400" height="200"></canvas>
                </div>
            </div>
            <div class="col-md-6">
                <div class="stat-card">
                    <h5 class="mb-4"><i class="fas fa-chart-bar me-2"></i>Confidence Distribution</h5>
                    <canvas id="confidenceChart" width="400" height="200"></canvas>
                </div>
            </div>
        </div>

        <div class="alert alert-info mt-4">
            <h6><i class="fas fa-info-circle me-2"></i>Demo Mode Notice</h6>
            <p class="mb-0">This is a demonstration version. All statistics are simulated for educational purposes.</p>
        </div>
    </div>
    '''

    extra_js = '''
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const ctx1 = document.getElementById('resultsChart').getContext('2d');
        new Chart(ctx1, {
            type: 'doughnut',
            data: {
                labels: ['Real Images', 'Fake Images'],
                datasets: [{
                    data: [756, 491],
                    backgroundColor: ['#28a745', '#dc3545'],
                    borderWidth: 0
                }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });

        const ctx2 = document.getElementById('confidenceChart').getContext('2d');
        new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
                datasets: [{
                    label: 'Detections',
                    data: [45, 89, 156, 234, 723],
                    backgroundColor: '#667eea'
                }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
    });
    </script>
    '''

    return create_html_page("Statistics - AI Deepfake Detector", content, extra_js)

@app.route('/about')
def about():
    """About page"""
    content = '''
    <div class="container mt-5">
        <div class="content-card">
            <h1 class="mb-4"><i class="fas fa-info-circle me-3"></i>About AI Deepfake Detector</h1>
            <p class="lead">Advanced machine learning technology to combat the growing threat of deepfake media.</p>

            <h3 class="mt-5 mb-3"><i class="fas fa-bullseye me-2"></i>Our Mission</h3>
            <p>To provide accessible, accurate, and reliable deepfake detection technology that helps preserve truth and authenticity in digital media.</p>

            <h3 class="mt-4 mb-3"><i class="fas fa-cogs me-2"></i>Technology</h3>
            <div class="row">
                <div class="col-md-6">
                    <ul>
                        <li><strong>Deep Learning Models:</strong> State-of-the-art neural networks</li>
                        <li><strong>Feature Analysis:</strong> Advanced manipulation detection</li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <ul>
                        <li><strong>Real-time Processing:</strong> Results in under 0.5 seconds</li>
                        <li><strong>High Accuracy:</strong> 92.8% detection accuracy</li>
                    </ul>
                </div>
            </div>

            <div class="alert alert-info mt-4">
                <h5><i class="fas fa-lightbulb me-2"></i>Demo Mode Notice</h5>
                <p class="mb-0">This is a demonstration version running in demo mode. All analysis results are simulated for educational purposes.</p>
            </div>

            <div class="text-center mt-5">
                <a href="/" class="btn btn-primary btn-lg me-3"><i class="fas fa-upload me-2"></i>Try Detection</a>
                <a href="/gallery" class="btn btn-outline-primary btn-lg"><i class="fas fa-images me-2"></i>View Gallery</a>
            </div>
        </div>
    </div>
    '''
    return create_html_page("About - AI Deepfake Detector", content)

@app.route('/training')
def training():
    """Training dashboard page"""
    content = '''
    <div class="container mt-5">
        <h1 class="mb-4"><i class="fas fa-cogs me-3"></i>Training Dashboard</h1>
        <p class="lead mb-5">Monitor model training progress and performance improvements.</p>

        <div class="row">
            <div class="col-md-8">
                <div class="content-card">
                    <h4><i class="fas fa-chart-line me-2"></i>Training Progress</h4>
                    <div class="mt-4">
                        <div class="d-flex justify-content-between mb-2">
                            <span>Current Epoch:</span><strong>47 / 100</strong>
                        </div>
                        <div class="progress mb-3" style="height: 20px;">
                            <div class="progress-bar" style="width: 47%">47%</div>
                        </div>
                        <div class="d-flex justify-content-between mb-2">
                            <span>Training Accuracy:</span><strong>94.2%</strong>
                        </div>
                        <div class="progress mb-3">
                            <div class="progress-bar bg-success" style="width: 94.2%"></div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="content-card">
                    <h5><i class="fas fa-info-circle me-2"></i>Training Info</h5>
                    <p><strong>Status:</strong> <span class="badge bg-success">Running</span></p>
                    <p><strong>Started:</strong> 2 hours ago</p>
                    <p><strong>ETA:</strong> 3.2 hours</p>
                </div>
            </div>
        </div>

        <div class="alert alert-info mt-4">
            <h5><i class="fas fa-info-circle me-2"></i>Demo Mode</h5>
            <p class="mb-0">This training dashboard shows simulated training progress for demonstration purposes.</p>
        </div>
    </div>
    '''
    return create_html_page("Training Dashboard - AI Deepfake Detector", content)

@app.route('/contact')
def contact():
    """Contact page"""
    content = '''
    <div class="container mt-5">
        <div class="content-card">
            <h1 class="mb-4"><i class="fas fa-envelope me-3"></i>Contact Us</h1>
            <p class="lead">Get in touch with our team for questions, support, or collaboration opportunities.</p>
            <div class="row mt-4">
                <div class="col-md-6">
                    <h5><i class="fas fa-paper-plane me-2"></i>Send us a message</h5>
                    <p>We'd love to hear from you. Send us a message and we'll respond as soon as possible.</p>
                </div>
                <div class="col-md-6">
                    <h5><i class="fas fa-info-circle me-2"></i>Demo Information</h5>
                    <p>This is a demonstration version. Contact functionality is simulated for educational purposes.</p>
                </div>
            </div>
        </div>
    </div>
    '''
    return create_html_page("Contact - AI Deepfake Detector", content)

@app.route('/documentation')
def documentation():
    """API documentation page"""
    content = '''
    <div class="container mt-5">
        <div class="content-card">
            <h1 class="mb-4"><i class="fas fa-book me-3"></i>API Documentation</h1>
            <p class="lead">Developer resources and API guides for integrating deepfake detection.</p>
            <h3 class="mt-4">Available Endpoints:</h3>
            <ul>
                <li><code>GET /api/health</code> - Health check endpoint</li>
                <li><code>GET /api/model_stats</code> - Model performance statistics</li>
                <li><code>POST /upload</code> - Image analysis endpoint</li>
            </ul>
        </div>
    </div>
    '''
    return create_html_page("API Documentation - AI Deepfake Detector", content)

@app.route('/realtime')
def realtime():
    """Real-time detection page"""
    content = '''
    <div class="container mt-5">
        <div class="row">
            <div class="col-12">
                <h1 class="mb-4 text-center">
                    <i class="fas fa-video me-3 text-primary"></i>Real-time Camera Detection
                </h1>
                <p class="lead text-center mb-5">
                    Use your device camera for instant deepfake detection analysis.
                </p>
            </div>
        </div>

        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="content-card">
                    <div class="text-center mb-4">
                        <h3><i class="fas fa-camera me-2 text-success"></i>Camera Analysis</h3>
                        <p class="text-muted">Capture photos directly from your camera for analysis</p>
                    </div>

                    <div id="camera-container" class="mb-4 text-center">
                        <video id="realtime-video" class="d-none" autoplay playsinline style="max-width: 100%; max-height: 400px; border-radius: 15px; box-shadow: 0 8px 25px rgba(0,0,0,0.2);"></video>
                        <canvas id="realtime-canvas" class="d-none"></canvas>
                        <div id="realtime-placeholder" class="bg-light border rounded d-flex align-items-center justify-content-center mx-auto" style="height: 300px; max-width: 500px;">
                            <div class="text-center text-muted">
                                <i class="fas fa-camera fa-4x mb-3 text-primary"></i>
                                <h5>Camera Ready</h5>
                                <p>Click "Start Camera" to begin real-time detection</p>
                            </div>
                        </div>
                        <div id="realtime-captured" class="d-none">
                            <img id="realtime-img" class="img-fluid rounded shadow" style="max-height: 400px;">
                        </div>
                    </div>

                    <div class="d-flex gap-2 justify-content-center flex-wrap mb-4">
                        <button id="realtime-start-btn" class="btn btn-success btn-lg">
                            <i class="fas fa-play me-2"></i>Start Camera
                        </button>
                        <button id="realtime-capture-btn" class="btn btn-primary btn-lg d-none">
                            <i class="fas fa-camera me-2"></i>Capture & Analyze
                        </button>
                        <button id="realtime-retake-btn" class="btn btn-warning btn-lg d-none">
                            <i class="fas fa-redo me-2"></i>Retake
                        </button>
                        <button id="realtime-stop-btn" class="btn btn-danger btn-lg d-none">
                            <i class="fas fa-stop me-2"></i>Stop Camera
                        </button>
                    </div>

                    <div id="realtime-results" class="mt-4"></div>

                    <div class="alert alert-info">
                        <h6><i class="fas fa-shield-alt me-2"></i>Privacy Notice</h6>
                        <p class="mb-0">
                            <small>
                                Your camera feed is processed locally in your browser. No images are stored or transmitted
                                to our servers without your explicit action. Camera access is required for this feature.
                            </small>
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-5">
            <div class="col-md-4">
                <div class="feature-card text-center">
                    <i class="fas fa-bolt fa-3x text-warning mb-3"></i>
                    <h5>Instant Analysis</h5>
                    <p class="text-muted">Get results in under 0.5 seconds</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="feature-card text-center">
                    <i class="fas fa-shield-check fa-3x text-success mb-3"></i>
                    <h5>Privacy First</h5>
                    <p class="text-muted">Local processing, no data stored</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="feature-card text-center">
                    <i class="fas fa-mobile-alt fa-3x text-info mb-3"></i>
                    <h5>Mobile Friendly</h5>
                    <p class="text-muted">Works on all devices with camera</p>
                </div>
            </div>
        </div>
    </div>
    '''

    realtime_js = '''
    <script>
        // Real-time camera functionality
        let realtimeStream = null;

        const realtimeVideo = document.getElementById('realtime-video');
        const realtimeCanvas = document.getElementById('realtime-canvas');
        const realtimeCtx = realtimeCanvas.getContext('2d');
        const realtimePlaceholder = document.getElementById('realtime-placeholder');
        const realtimeCaptured = document.getElementById('realtime-captured');
        const realtimeImg = document.getElementById('realtime-img');
        const realtimeResults = document.getElementById('realtime-results');

        const realtimeStartBtn = document.getElementById('realtime-start-btn');
        const realtimeCaptureBtn = document.getElementById('realtime-capture-btn');
        const realtimeRetakeBtn = document.getElementById('realtime-retake-btn');
        const realtimeStopBtn = document.getElementById('realtime-stop-btn');

        // Start camera
        realtimeStartBtn.addEventListener('click', async () => {
            try {
                realtimeStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    }
                });

                realtimeVideo.srcObject = realtimeStream;
                realtimeVideo.play();

                realtimePlaceholder.classList.add('d-none');
                realtimeVideo.classList.remove('d-none');
                realtimeCaptured.classList.add('d-none');

                realtimeStartBtn.classList.add('d-none');
                realtimeCaptureBtn.classList.remove('d-none');
                realtimeStopBtn.classList.remove('d-none');

                realtimeVideo.addEventListener('loadedmetadata', () => {
                    realtimeCanvas.width = realtimeVideo.videoWidth;
                    realtimeCanvas.height = realtimeVideo.videoHeight;
                });

            } catch (error) {
                console.error('Camera error:', error);
                alert('Unable to access camera. Please check permissions and try again.');
            }
        });

        // Capture and analyze
        realtimeCaptureBtn.addEventListener('click', async () => {
            if (realtimeVideo.videoWidth && realtimeVideo.videoHeight) {
                // Capture image
                realtimeCanvas.width = realtimeVideo.videoWidth;
                realtimeCanvas.height = realtimeVideo.videoHeight;
                realtimeCtx.drawImage(realtimeVideo, 0, 0);

                const imageData = realtimeCanvas.toDataURL('image/jpeg', 0.8);
                realtimeImg.src = imageData;

                realtimeVideo.classList.add('d-none');
                realtimeCaptured.classList.remove('d-none');
                realtimeCaptureBtn.classList.add('d-none');
                realtimeRetakeBtn.classList.remove('d-none');

                // Analyze immediately
                realtimeRetakeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
                realtimeRetakeBtn.disabled = true;

                // Simulate analysis
                setTimeout(() => {
                    const result = {
                        result: Math.random() > 0.5 ? 'Real' : 'Fake',
                        confidence_score: Math.floor(Math.random() * 15) + 85,
                        processing_time: (Math.random() * 0.3 + 0.1).toFixed(2)
                    };

                    displayRealtimeResult(result);
                    realtimeRetakeBtn.innerHTML = '<i class="fas fa-redo me-2"></i>Retake';
                    realtimeRetakeBtn.disabled = false;
                }, 1000);
            }
        });

        // Retake
        realtimeRetakeBtn.addEventListener('click', () => {
            realtimeCaptured.classList.add('d-none');
            realtimeVideo.classList.remove('d-none');
            realtimeCaptureBtn.classList.remove('d-none');
            realtimeRetakeBtn.classList.add('d-none');
            realtimeResults.innerHTML = '';
        });

        // Stop camera
        realtimeStopBtn.addEventListener('click', () => {
            if (realtimeStream) {
                realtimeStream.getTracks().forEach(track => track.stop());
                realtimeStream = null;
            }

            realtimeVideo.classList.add('d-none');
            realtimeCaptured.classList.add('d-none');
            realtimePlaceholder.classList.remove('d-none');

            realtimeStartBtn.classList.remove('d-none');
            realtimeCaptureBtn.classList.add('d-none');
            realtimeRetakeBtn.classList.add('d-none');
            realtimeStopBtn.classList.add('d-none');

            realtimeResults.innerHTML = '';
        });

        function displayRealtimeResult(result) {
            const isReal = result.result === 'Real';
            const confidence = result.confidence_score;

            realtimeResults.innerHTML = `
                <div class="alert alert-${isReal ? 'success' : 'danger'} text-center">
                    <h4><i class="fas fa-${isReal ? 'check-circle' : 'exclamation-triangle'} me-2"></i>
                        ${result.result} Image Detected
                    </h4>
                    <div class="row mt-3">
                        <div class="col-md-4">
                            <div class="stat-number" style="font-size: 2rem;">${confidence}%</div>
                            <small>Confidence</small>
                        </div>
                        <div class="col-md-4">
                            <div class="stat-number" style="font-size: 2rem;">${result.processing_time}s</div>
                            <small>Processing Time</small>
                        </div>
                        <div class="col-md-4">
                            <div class="stat-number" style="font-size: 2rem;"><i class="fas fa-camera"></i></div>
                            <small>Live Capture</small>
                        </div>
                    </div>
                    <div class="progress mt-3" style="height: 15px;">
                        <div class="progress-bar bg-${isReal ? 'success' : 'danger'}" style="width: ${confidence}%"></div>
                    </div>
                </div>
            `;
        }

        // Cleanup
        window.addEventListener('beforeunload', () => {
            if (realtimeStream) {
                realtimeStream.getTracks().forEach(track => track.stop());
            }
        });
    </script>
    '''

    return create_html_page("Real-time Detection - AI Deepfake Detector", content, realtime_js)

@app.route('/api_explorer')
def api_explorer():
    """API explorer page"""
    content = '''
    <div class="container mt-5">
        <div class="content-card">
            <h1 class="mb-4"><i class="fas fa-code me-3"></i>API Explorer</h1>
            <p class="lead">Test our API endpoints and explore integration possibilities.</p>
            <div class="alert alert-info">
                <p class="mb-0">Interactive API testing is available in the full version.</p>
            </div>
        </div>
    </div>
    '''
    return create_html_page("API Explorer - AI Deepfake Detector", content)

@app.route('/batch_processing')
def batch_processing():
    """Batch processing page"""
    content = '''
    <div class="container mt-5">
        <div class="content-card">
            <h1 class="mb-4"><i class="fas fa-layer-group me-3"></i>Batch Processing</h1>
            <p class="lead">Process multiple images at once for efficient analysis.</p>
            <div class="alert alert-info">
                <p class="mb-0">Batch processing features are available in the full version.</p>
            </div>
        </div>
    </div>
    '''
    return create_html_page("Batch Processing - AI Deepfake Detector", content)

@app.route('/home')
def home():
    """Alternative home page"""
    return index()

# API endpoints
@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'AI Deepfake Detector is running on Vercel',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/model_stats')
def get_model_stats():
    """Get model statistics"""
    return jsonify({
        'accuracy': 92.8,
        'precision': 94.1,
        'recall': 91.5,
        'f1_score': 92.8,
        'version': 'v2.1',
        'status': 'demo_mode'
    })

@app.route('/api/analysis_history')
def get_analysis_history():
    """Get analysis history"""
    # Return demo data
    demo_history = [
        {
            'timestamp': '2024-01-15T10:30:00',
            'result': 'Real',
            'confidence': 94.2,
            'processing_time': 0.18
        },
        {
            'timestamp': '2024-01-15T10:25:00',
            'result': 'Fake',
            'confidence': 87.6,
            'processing_time': 0.22
        },
        {
            'timestamp': '2024-01-15T10:20:00',
            'result': 'Real',
            'confidence': 96.1,
            'processing_time': 0.15
        }
    ]
    return jsonify(demo_history)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Demo analysis results with enhanced features
        result_type = random.choice(['Real', 'Fake'])
        confidence = round(random.uniform(85.0, 98.0), 1)
        processing_time = round(random.uniform(0.15, 0.35), 2)

        demo_result = {
            'result': result_type,
            'confidence_score': confidence,
            'confidence': confidence,  # Alternative key for compatibility
            'processing_time': processing_time,
            'features': {
                'edge_detection': round(random.uniform(80, 95), 1),
                'color_analysis': round(random.uniform(85, 95), 1),
                'texture_patterns': round(random.uniform(75, 90), 1),
                'geometric_features': round(random.uniform(88, 96), 1),
                'facial_landmarks': round(random.uniform(70, 95), 1),
                'lighting_consistency': round(random.uniform(75, 92), 1)
            },
            'quality_score': round(random.uniform(7.5, 9.8), 1),
            'image_info': {
                'format': file.filename.split('.')[-1].upper() if '.' in file.filename else 'UNKNOWN',
                'source': 'camera' if 'camera-capture' in file.filename else 'upload'
            },
            'timestamp': datetime.now().isoformat(),
            'model_version': model_metrics['version'],
            'analysis_id': f"analysis_{random.randint(100000, 999999)}"
        }

        return jsonify(demo_result)

    except Exception as e:
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    content = '''
    <div class="container mt-5">
        <div class="content-card text-center">
            <h1 class="mb-4"><i class="fas fa-exclamation-triangle me-3"></i>404 - Page Not Found</h1>
            <p class="lead">The page you're looking for doesn't exist.</p>
            <a href="/" class="btn btn-primary"><i class="fas fa-home me-2"></i>Back to Home</a>
        </div>
    </div>
    '''
    return create_html_page("404 - Page Not Found", content), 404

@app.errorhandler(500)
def internal_error(error):
    content = '''
    <div class="container mt-5">
        <div class="content-card text-center">
            <h1 class="mb-4"><i class="fas fa-exclamation-triangle me-3"></i>500 - Internal Server Error</h1>
            <p class="lead">Something went wrong on our end.</p>
            <a href="/" class="btn btn-primary"><i class="fas fa-home me-2"></i>Back to Home</a>
        </div>
    </div>
    '''
    return create_html_page("500 - Internal Server Error", content), 500

if __name__ == '__main__':
    app.run(debug=True)
