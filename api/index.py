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
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --success-color: #28a745;
            --danger-color: #dc3545;
        }}
        body {{ background: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .navbar {{ background: var(--primary-gradient) !important; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .navbar-brand {{ font-weight: bold; font-size: 1.5rem; }}
        .hero-section {{ background: var(--primary-gradient); color: white; padding: 100px 0; text-align: center; }}
        .feature-card, .stat-card, .content-card {{ background: white; border-radius: 15px; padding: 30px; margin: 20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.1); transition: transform 0.3s ease; }}
        .feature-card:hover, .stat-card:hover {{ transform: translateY(-5px); }}
        .btn-primary {{ background: var(--primary-gradient); border: none; padding: 12px 30px; border-radius: 25px; font-weight: 600; }}
        .stat-number {{ font-size: 2.5rem; font-weight: bold; background: var(--primary-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }}
        .gallery-item {{ background: white; border-radius: 15px; padding: 25px; margin: 20px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.1); }}
        .fake-badge {{ background: var(--danger-color); color: white; padding: 8px 15px; border-radius: 20px; font-size: 12px; font-weight: bold; }}
        .real-badge {{ background: var(--success-color); color: white; padding: 8px 15px; border-radius: 20px; font-size: 12px; font-weight: bold; }}
        .image-placeholder {{ height: 200px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px; margin: 15px 0; }}
        .fake-placeholder {{ background: linear-gradient(45deg, #ff6b6b, #ee5a24); }}
        .real-placeholder {{ background: linear-gradient(45deg, #26de81, #20bf6b); }}
        .footer {{ background: #343a40; color: white; padding: 40px 0; margin-top: 50px; }}
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-robot me-2"></i>AI Deepfake Detector</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <div class="navbar-nav ms-auto">
                    <a class="nav-link" href="/"><i class="fas fa-home me-1"></i>Home</a>
                    <a class="nav-link" href="/gallery"><i class="fas fa-images me-1"></i>Gallery</a>
                    <a class="nav-link" href="/statistics"><i class="fas fa-chart-bar me-1"></i>Statistics</a>
                    <a class="nav-link" href="/training"><i class="fas fa-cogs me-1"></i>Training</a>
                    <a class="nav-link" href="/about"><i class="fas fa-info-circle me-1"></i>About</a>
                </div>
            </div>
        </div>
    </nav>

    <main>{content}</main>

    <footer class="footer">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <h5><i class="fas fa-robot me-2"></i>AI Deepfake Detector</h5>
                    <p>Advanced machine learning technology for deepfake detection.</p>
                </div>
                <div class="col-md-6 text-md-end">
                    <p>&copy; 2024 AI Deepfake Detector. Demo Version.</p>
                    <p><small>Running on Vercel with Flask</small></p>
                </div>
            </div>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
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
            <h1 class="display-4 mb-4"><i class="fas fa-shield-alt me-3"></i>AI Deepfake Detector</h1>
            <p class="lead mb-5">Advanced machine learning technology to detect deepfake images with {model_metrics['accuracy']}% accuracy</p>

            {error_html}

            <div class="row justify-content-center">
                <div class="col-md-8">
                    <div class="feature-card">
                        <h3 class="mb-4">Upload Image for Analysis</h3>
                        <form method="POST" enctype="multipart/form-data">
                            <div class="mb-3">
                                <input type="file" class="form-control" name="file" accept="image/*" required>
                            </div>
                            <button type="submit" class="btn btn-primary btn-lg">
                                <i class="fas fa-search me-2"></i>Analyze Image
                            </button>
                        </form>
                        {result_html}
                    </div>
                </div>
            </div>

            <div class="row mt-5">
                <div class="col-md-4">
                    <div class="feature-card">
                        <i class="fas fa-eye fa-3x text-primary mb-3"></i>
                        <h4>Real-time Detection</h4>
                        <p>Upload images and get instant deepfake analysis with detailed confidence scores.</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="feature-card">
                        <i class="fas fa-chart-line fa-3x text-success mb-3"></i>
                        <h4>{model_metrics['accuracy']}% Accuracy</h4>
                        <p>State-of-the-art machine learning model trained on millions of images.</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="feature-card">
                        <i class="fas fa-images fa-3x text-warning mb-3"></i>
                        <h4>Celebrity Gallery</h4>
                        <p>Explore examples of detected deepfakes of famous personalities.</p>
                        <a href="/gallery" class="btn btn-outline-primary mt-2">View Gallery</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
    '''

    return create_html_page("AI Deepfake Detector - Advanced ML Detection", content)

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
        <div class="content-card">
            <h1 class="mb-4"><i class="fas fa-video me-3"></i>Real-time Detection</h1>
            <p class="lead">Live deepfake detection interface for real-time analysis.</p>
            <div class="alert alert-info">
                <p class="mb-0">Real-time detection features are available in the full version. This demo shows static analysis only.</p>
            </div>
        </div>
    </div>
    '''
    return create_html_page("Real-time Detection - AI Deepfake Detector", content)

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
    
    # Demo analysis results
    demo_result = {
        'result': random.choice(['Real', 'Fake']),
        'confidence': round(random.uniform(85.0, 98.0), 1),
        'processing_time': round(random.uniform(0.15, 0.35), 2),
        'features': {
            'edge_detection': round(random.uniform(80, 95), 1),
            'color_analysis': round(random.uniform(85, 95), 1),
            'texture_patterns': round(random.uniform(75, 90), 1),
            'geometric_features': round(random.uniform(88, 96), 1)
        },
        'quality_score': round(random.uniform(7.5, 9.8), 1),
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(demo_result)

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
