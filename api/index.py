from flask import Flask, render_template, request, jsonify
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
    return render_template('index.html')

@app.route('/gallery')
def gallery():
    """Gallery page with deepfake examples"""
    return render_template('gallery.html')

@app.route('/training')
def training():
    """Training dashboard page"""
    return render_template('training.html')

@app.route('/statistics')
def statistics():
    """Statistics and analytics page"""
    return render_template('statistics.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

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
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
