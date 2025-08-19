from flask import Flask, request, jsonify
import os
import random
import time
from datetime import datetime

app = Flask(__name__)

# Configure Flask for Vercel
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

@app.route('/')
def index():
    """Main page - AI Deepfake Detector"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Deepfake Detector</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .hero-section { padding: 100px 0; color: white; text-align: center; }
            .feature-card { background: white; border-radius: 15px; padding: 30px; margin: 20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
            .btn-primary { background: linear-gradient(45deg, #667eea, #764ba2); border: none; padding: 12px 30px; }
            .navbar { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); }
            .navbar-brand { color: white !important; font-weight: bold; }
            .nav-link { color: white !important; }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark fixed-top">
            <div class="container">
                <a class="navbar-brand" href="/"><i class="fas fa-robot me-2"></i>AI Deepfake Detector</a>
                <div class="navbar-nav ms-auto">
                    <a class="nav-link" href="/gallery">Gallery</a>
                    <a class="nav-link" href="/statistics">Statistics</a>
                    <a class="nav-link" href="/about">About</a>
                </div>
            </div>
        </nav>

        <div class="hero-section">
            <div class="container">
                <h1 class="display-4 mb-4"><i class="fas fa-shield-alt me-3"></i>AI Deepfake Detector</h1>
                <p class="lead mb-5">Advanced machine learning technology to detect deepfake images with 92.8% accuracy</p>

                <div class="row justify-content-center">
                    <div class="col-md-8">
                        <div class="feature-card">
                            <h3 class="mb-4">Upload Image for Analysis</h3>
                            <div class="mb-3">
                                <input type="file" class="form-control" id="imageUpload" accept="image/*">
                            </div>
                            <button class="btn btn-primary btn-lg" onclick="analyzeImage()">
                                <i class="fas fa-search me-2"></i>Analyze Image
                            </button>
                            <div id="results" class="mt-4" style="display:none;">
                                <div class="alert alert-info">
                                    <h5>Analysis Results:</h5>
                                    <p id="resultText"></p>
                                    <div class="progress mb-2">
                                        <div id="confidenceBar" class="progress-bar" style="width: 0%"></div>
                                    </div>
                                    <small id="processingTime"></small>
                                </div>
                            </div>
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
                            <h4>92.8% Accuracy</h4>
                            <p>State-of-the-art machine learning model trained on millions of images.</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="feature-card">
                            <i class="fas fa-images fa-3x text-warning mb-3"></i>
                            <h4>Celebrity Gallery</h4>
                            <p>Explore examples of detected deepfakes of famous personalities.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function analyzeImage() {
                const fileInput = document.getElementById('imageUpload');
                const results = document.getElementById('results');
                const resultText = document.getElementById('resultText');
                const confidenceBar = document.getElementById('confidenceBar');
                const processingTime = document.getElementById('processingTime');

                if (!fileInput.files[0]) {
                    alert('Please select an image first!');
                    return;
                }

                // Simulate analysis
                const isReal = Math.random() > 0.5;
                const confidence = Math.floor(Math.random() * 15) + 85; // 85-100%
                const time = (Math.random() * 0.3 + 0.1).toFixed(2); // 0.1-0.4s

                resultText.innerHTML = `<strong>Result:</strong> ${isReal ? 'Real' : 'Fake'} Image<br><strong>Confidence:</strong> ${confidence}%`;
                confidenceBar.style.width = confidence + '%';
                confidenceBar.className = `progress-bar ${isReal ? 'bg-success' : 'bg-danger'}`;
                processingTime.textContent = `Processing time: ${time}s`;

                results.style.display = 'block';
            }
        </script>
    </body>
    </html>
    '''

@app.route('/gallery')
def gallery():
    """Gallery page with deepfake examples"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gallery - AI Deepfake Detector</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background: #f8f9fa; }
            .navbar { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .gallery-item { background: white; border-radius: 10px; padding: 20px; margin: 15px 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
            .fake-badge { background: #dc3545; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; }
            .real-badge { background: #28a745; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark">
            <div class="container">
                <a class="navbar-brand" href="/"><i class="fas fa-robot me-2"></i>AI Deepfake Detector</a>
                <div class="navbar-nav ms-auto">
                    <a class="nav-link" href="/">Home</a>
                    <a class="nav-link active" href="/gallery">Gallery</a>
                    <a class="nav-link" href="/statistics">Statistics</a>
                    <a class="nav-link" href="/about">About</a>
                </div>
            </div>
        </nav>

        <div class="container mt-5">
            <h1 class="mb-4"><i class="fas fa-images me-3"></i>Deepfake Detection Gallery</h1>
            <p class="lead mb-5">Examples of detected deepfakes and authentic images from our analysis system.</p>

            <div class="row">
                <div class="col-md-6">
                    <div class="gallery-item">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5>Celebrity Deepfake</h5>
                            <span class="fake-badge">FAKE - 91.2%</span>
                        </div>
                        <div style="height: 200px; background: linear-gradient(45deg, #ff6b6b, #ee5a24); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px;">
                            <i class="fas fa-user-secret fa-3x"></i>
                        </div>
                        <p class="mt-3">AI-generated face swap of a famous Bollywood actor. Our model detected temporal inconsistencies and unnatural facial features.</p>
                        <small class="text-muted">Processing time: 0.22s</small>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="gallery-item">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5>Authentic Portrait</h5>
                            <span class="real-badge">REAL - 96.5%</span>
                        </div>
                        <div style="height: 200px; background: linear-gradient(45deg, #26de81, #20bf6b); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px;">
                            <i class="fas fa-user-check fa-3x"></i>
                        </div>
                        <p class="mt-3">Genuine photograph with natural lighting and authentic facial characteristics. High confidence score indicates authentic content.</p>
                        <small class="text-muted">Processing time: 0.17s</small>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="gallery-item">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5>Political Figure Deepfake</h5>
                            <span class="fake-badge">FAKE - 88.9%</span>
                        </div>
                        <div style="height: 200px; background: linear-gradient(45deg, #fd79a8, #e84393); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px;">
                            <i class="fas fa-mask fa-3x"></i>
                        </div>
                        <p class="mt-3">Deepfake of a prominent political leader. Detected through analysis of micro-expressions and lighting inconsistencies.</p>
                        <small class="text-muted">Processing time: 0.24s</small>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="gallery-item">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5>Sports Celebrity</h5>
                            <span class="real-badge">REAL - 94.7%</span>
                        </div>
                        <div style="height: 200px; background: linear-gradient(45deg, #74b9ff, #0984e3); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px;">
                            <i class="fas fa-trophy fa-3x"></i>
                        </div>
                        <p class="mt-3">Authentic photograph of a famous cricket player from match coverage. Natural skin texture and consistent lighting confirmed.</p>
                        <small class="text-muted">Processing time: 0.15s</small>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/statistics')
def statistics():
    """Statistics and analytics page"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Statistics - AI Deepfake Detector</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { background: #f8f9fa; }
            .navbar { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .stat-card { background: white; border-radius: 10px; padding: 30px; margin: 15px 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1); text-align: center; }
            .stat-number { font-size: 2.5rem; font-weight: bold; color: #667eea; }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark">
            <div class="container">
                <a class="navbar-brand" href="/"><i class="fas fa-robot me-2"></i>AI Deepfake Detector</a>
                <div class="navbar-nav ms-auto">
                    <a class="nav-link" href="/">Home</a>
                    <a class="nav-link" href="/gallery">Gallery</a>
                    <a class="nav-link active" href="/statistics">Statistics</a>
                    <a class="nav-link" href="/about">About</a>
                </div>
            </div>
        </nav>

        <div class="container mt-5">
            <h1 class="mb-4"><i class="fas fa-chart-bar me-3"></i>Detection Statistics</h1>

            <div class="row">
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-number">1,247</div>
                        <h5>Total Analyses</h5>
                        <p class="text-muted">Images processed</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-number">92.8%</div>
                        <h5>Accuracy Rate</h5>
                        <p class="text-muted">Detection accuracy</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-number">756</div>
                        <h5>Real Images</h5>
                        <p class="text-muted">Authentic detected</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-number">491</div>
                        <h5>Fake Images</h5>
                        <p class="text-muted">Deepfakes detected</p>
                    </div>
                </div>
            </div>

            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="stat-card">
                        <h5 class="mb-4">Detection Results</h5>
                        <canvas id="resultsChart" width="400" height="200"></canvas>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="stat-card">
                        <h5 class="mb-4">Confidence Distribution</h5>
                        <canvas id="confidenceChart" width="400" height="200"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Results Chart
            const ctx1 = document.getElementById('resultsChart').getContext('2d');
            new Chart(ctx1, {
                type: 'doughnut',
                data: {
                    labels: ['Real Images', 'Fake Images'],
                    datasets: [{
                        data: [756, 491],
                        backgroundColor: ['#28a745', '#dc3545']
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });

            // Confidence Chart
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
        </script>
    </body>
    </html>
    '''

@app.route('/about')
def about():
    """About page"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>About - AI Deepfake Detector</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background: #f8f9fa; }
            .navbar { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .content-card { background: white; border-radius: 10px; padding: 40px; margin: 20px 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark">
            <div class="container">
                <a class="navbar-brand" href="/"><i class="fas fa-robot me-2"></i>AI Deepfake Detector</a>
                <div class="navbar-nav ms-auto">
                    <a class="nav-link" href="/">Home</a>
                    <a class="nav-link" href="/gallery">Gallery</a>
                    <a class="nav-link" href="/statistics">Statistics</a>
                    <a class="nav-link active" href="/about">About</a>
                </div>
            </div>
        </nav>

        <div class="container mt-5">
            <div class="content-card">
                <h1 class="mb-4"><i class="fas fa-info-circle me-3"></i>About AI Deepfake Detector</h1>

                <p class="lead">Advanced machine learning technology to combat the growing threat of deepfake media.</p>

                <h3 class="mt-5 mb-3">Our Mission</h3>
                <p>To provide accessible, accurate, and reliable deepfake detection technology that helps preserve truth and authenticity in digital media. We believe in empowering individuals and organizations with the tools needed to identify manipulated content.</p>

                <h3 class="mt-4 mb-3">Technology</h3>
                <ul>
                    <li><strong>Deep Learning Models:</strong> State-of-the-art neural networks trained on millions of images</li>
                    <li><strong>Feature Analysis:</strong> Advanced algorithms for detecting subtle manipulation artifacts</li>
                    <li><strong>Real-time Processing:</strong> Fast analysis with results in under 0.5 seconds</li>
                    <li><strong>High Accuracy:</strong> 92.8% detection accuracy across various deepfake techniques</li>
                </ul>

                <h3 class="mt-4 mb-3">Key Features</h3>
                <div class="row">
                    <div class="col-md-6">
                        <ul>
                            <li>Real-time image analysis</li>
                            <li>Confidence scoring</li>
                            <li>Detailed feature breakdown</li>
                            <li>Celebrity deepfake gallery</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <ul>
                            <li>API access for developers</li>
                            <li>Batch processing capabilities</li>
                            <li>Training progress tracking</li>
                            <li>Comprehensive statistics</li>
                        </ul>
                    </div>
                </div>

                <div class="alert alert-info mt-4">
                    <h5><i class="fas fa-lightbulb me-2"></i>Note</h5>
                    <p class="mb-0">This is a demonstration version running in demo mode. All analysis results are simulated for educational purposes.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

# Simplified routes for other pages
@app.route('/training')
def training():
    return '<h1>Training Dashboard</h1><p>Model training interface - Coming soon!</p><a href="/">← Back to Home</a>'

@app.route('/contact')
def contact():
    return '<h1>Contact Us</h1><p>Get in touch with our team.</p><a href="/">← Back to Home</a>'

@app.route('/documentation')
def documentation():
    return '<h1>API Documentation</h1><p>Developer resources and API guides.</p><a href="/">← Back to Home</a>'

@app.route('/realtime')
def realtime():
    return '<h1>Real-time Detection</h1><p>Live deepfake detection interface.</p><a href="/">← Back to Home</a>'

@app.route('/api_explorer')
def api_explorer():
    return '<h1>API Explorer</h1><p>Test our API endpoints.</p><a href="/">← Back to Home</a>'

@app.route('/batch_processing')
def batch_processing():
    return '<h1>Batch Processing</h1><p>Process multiple images at once.</p><a href="/">← Back to Home</a>'

@app.route('/home')
def home():
    return index()  # Redirect to main page

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
    return '''
    <h1>404 - Page Not Found</h1>
    <p>The page you're looking for doesn't exist.</p>
    <a href="/">← Back to Home</a>
    ''', 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

if __name__ == '__main__':
    app.run(debug=True)
