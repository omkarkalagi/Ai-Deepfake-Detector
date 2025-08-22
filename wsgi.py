import os
from app.app import AdvancedDeepfakeDetector

try:
    detector = AdvancedDeepfakeDetector()
    app = detector.app
    
    # Configure for production
    app.config['ENV'] = os.getenv('FLASK_ENV', 'production')
    app.config['DEBUG'] = False
    
    if __name__ == '__main__':
        port = int(os.getenv('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
except Exception as e:
    print(f"Error initializing application: {str(e)}")
    raise
