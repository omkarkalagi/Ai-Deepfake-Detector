from app.app import AdvancedDeepfakeDetector

detector = AdvancedDeepfakeDetector()
app = detector.app

if __name__ == '__main__':
    app.run()
