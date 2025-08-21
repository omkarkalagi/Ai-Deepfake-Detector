
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import random
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from create_demo_model import DemoModel

class AdvancedDeepfakeDetector:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        # Ensure upload folder exists
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Initialize the demo model
        self.model = DemoModel()
        self.model_metrics = self.model.model_config['performance']
        self.setup_routes()

    def setup_routes(self):
        self.app.add_url_rule('/', 'home', self.home)
        self.app.add_url_rule('/health', 'health', self.health)
        self.app.add_url_rule('/api/health', 'api_health', self.health)  # Additional health endpoint
        self.app.add_url_rule('/upload', 'upload_file', self.upload_file, methods=['POST'])
        self.app.add_url_rule('/realtime', 'realtime', self.realtime)
        self.app.add_url_rule('/batch', 'batch', self.batch)
        self.app.add_url_rule('/api', 'api', self.api)
        self.app.add_url_rule('/documentation', 'documentation', self.documentation)
        self.app.add_url_rule('/training', 'training', self.training)
        self.app.add_url_rule('/statistics', 'statistics', self.statistics)
        self.app.add_url_rule('/gallery', 'gallery', self.gallery)
        self.app.add_url_rule('/about', 'about', self.about)
        self.app.add_url_rule('/contact', 'contact', self.contact)

    def home(self):
        return render_template('index.html', **self.model_metrics)

    def upload_file(self):
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                # Use the demo model to analyze the image
                result = self.model.predict(file_path)
                
                # Clean up the uploaded file
                os.remove(file_path)
                
                # Format the response
                response = {
                    'result': 'Fake' if result['is_fake'] else 'Real',
                    'confidence_score': result['confidence'],
                    'is_edited': result['analysis']['manipulation_score'] > 0.3,
                    'editing_percentage': result['analysis']['manipulation_score'] * 100,
                    'quality_score': (10 - result['analysis']['noise_level'] * 10),
                    'image_resolution': '128x128',  # Fixed for demo
                    'features_detected': int(40 + result['analysis']['sharpness'] / 10),
                    'analysis_details': result['analysis']
                }
                return jsonify(response)
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'error': str(e)})
            analysis_result = self.comprehensive_analysis(file_path)
            return jsonify(analysis_result)

    def health(self):
        """Health check endpoint."""
        return jsonify({"status": "healthy"}), 200
        
    def realtime(self):
        return render_template('realtime.html')

    def batch(self):
        return render_template('batch.html')

    def api(self):
        return render_template('api.html')

    def documentation(self):
        return render_template('documentation.html')

    def training(self):
        return render_template('training.html')

    def statistics(self):
        return render_template('statistics.html')

    def gallery(self):
        return render_template('gallery.html')

    def about(self):
        return render_template('about.html')

    def contact(self):
        return render_template('contact.html')

    def comprehensive_analysis(self, file_path):
        # Simulated analysis for demo purposes
        prediction = random.random()
        prediction_percentage = prediction * 100
        result = 'Fake' if prediction >= 0.5 else 'Real'
        confidence_score = prediction_percentage if result == 'Fake' else 100 - prediction_percentage
        
        return {
            'result': result,
            'prediction_percentage': round(prediction_percentage, 1),
            'confidence_score': round(confidence_score, 1),
            'is_edited': random.choice([True, False]),
            'editing_percentage': round(random.uniform(0, 100), 1),
            'quality_score': round(random.uniform(0, 10), 1),
            'image_resolution': f"{random.randint(500, 4000)}x{random.randint(500, 4000)}",
            'features_detected': random.randint(35, 55),
            'timestamp': datetime.now().isoformat()
        }

detector = AdvancedDeepfakeDetector()
app = detector.app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
