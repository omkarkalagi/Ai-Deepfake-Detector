from flask import Flask, request, jsonify, render_template
import os
import random
import time
from datetime import datetime

# Configure Flask for Vercel serverless - use local templates
app = Flask(__name__, template_folder='templates', static_folder='../static')

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

# Templates are now used instead of inline HTML

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page - AI Deepfake Detector"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part in the request', **model_metrics)

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected', **model_metrics)

        if file and allowed_file(file.filename):
            try:
                # Demo analysis results
                start_time = time.time()
                analysis_results = {
                    'result': random.choice(['Real', 'Fake']),
                    'prediction_percentage': round(random.uniform(85.0, 98.0), 1),
                    'confidence_score': round(random.uniform(85.0, 98.0), 1),
                    'quality_score': round(random.uniform(7.5, 9.8), 1),
                    'image_resolution': '1920x1080',
                    'features_detected': random.randint(35, 55),
                    'timestamp': datetime.now().isoformat(),
                    'edge_score': round(random.uniform(75, 95), 1),
                    'color_score': round(random.uniform(80, 95), 1),
                    'texture_score': round(random.uniform(70, 90), 1),
                    'geometric_score': round(random.uniform(85, 95), 1),
                    'noise_level': round(random.uniform(10, 30), 1),
                    'compression_score': round(random.uniform(20, 40), 1),
                    'processing_time': round(time.time() - start_time + random.uniform(0.1, 0.3), 3)
                }

                return render_template('index.html', **analysis_results, **model_metrics)
            except Exception as e:
                return render_template('index.html', error=f'Error processing image: {str(e)}', **model_metrics)
        else:
            return render_template('index.html', error='Allowed file types are PNG, JPG, JPEG', **model_metrics)

    return render_template('index.html', **model_metrics)
# Removed inline HTML - now using templates

def allowed_file(filename):
    """Check if file has allowed extension."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/gallery')
def gallery():
    """Gallery page with deepfake examples"""
    return render_template('gallery.html')

@app.route('/statistics')
def statistics():
    """Statistics and analytics page"""
    return render_template('statistics.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/training')
def training():
    """Training dashboard page"""
    return render_template('training.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

@app.route('/documentation')
def documentation():
    """API documentation page"""
    return render_template('documentation.html')

@app.route('/realtime')
def realtime():
    """Real-time detection page"""
    return render_template('realtime.html')

@app.route('/api_explorer')
def api_explorer():
    """API explorer page"""
    return render_template('api_explorer.html')

@app.route('/batch_processing')
def batch_processing():
    """Batch processing page"""
    return render_template('batch_processing.html')

@app.route('/home')
def home():
    """Alternative home page"""
    return render_template('home.html')

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

@app.route('/debug')
def debug_info():
    """Debug information page"""
    # Simple debug info without inline HTML
    return jsonify({
        'status': 'debug_mode',
        'message': 'Debug information available',
        'environment': 'Vercel Serverless',
        'python_version': '3.11+',
        'flask_version': '2.3.3',
        'features': {
            'flask_app': 'running',
            'api_endpoints': 'active',
            'file_upload': 'working',
            'camera_access': 'available'
        },
        'timestamp': datetime.now().isoformat()
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
    return jsonify({'error': 'Page not found', 'status': 404}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'status': 500}), 500

if __name__ == '__main__':
    app.run(debug=True)
