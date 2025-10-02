# 🚀 How to Run the AI Deepfake Detector

## Quick Start (Simplest Method)

### Option 1: Using the Launcher Script (Recommended)
```bash
python run.py
```

### Option 2: Direct Run
```bash
python api/index.py
```

## 📋 Prerequisites

- **Python**: 3.8 or higher (You have Python 3.13.5 ✅)
- **Dependencies**: Already installed ✅
  - Flask 3.1.1
  - TensorFlow 2.20.0
  - OpenCV 4.12.0
  - NumPy, Pillow, etc.

## 🎯 Step-by-Step Guide

### 1. Open Terminal
Navigate to the project directory:
```bash
cd C:\Projects\deepfake-detector
```

### 2. Run the Application
Choose one of these commands:

**Method A - Using launcher:**
```bash
python run.py
```

**Method B - Direct run:**
```bash
python api/index.py
```

**Method C - Using Python module:**
```bash
python -m flask --app api.index run
```

### 3. Access the Application
Once the server starts, open your browser and go to:
```
http://localhost:5000
```

### 4. Stop the Server
Press `Ctrl+C` in the terminal to stop the server.

## 🌐 Available Pages

Once running, you can access:

- **Home (Detection)**: http://localhost:5000/
- **Training**: http://localhost:5000/training
- **Statistics**: http://localhost:5000/statistics
- **Real-time Detection**: http://localhost:5000/realtime
- **About**: http://localhost:5000/about
- **Contact**: http://localhost:5000/contact

## 🔍 Troubleshooting

### Issue: Port Already in Use
If port 5000 is busy, the app will automatically try other ports. Check the terminal output for the actual port.

Or manually specify a different port:
```bash
$env:PORT=8080; python api/index.py
```

### Issue: Import Errors
If you see import errors, install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Model Not Found
The app works in demo mode without trained models. If you want to use actual models:
1. Train a model using the training page
2. Or place a model file named `deepfake_detector_model.keras` in the root directory

### Issue: Camera Not Working (Real-time Page)
- Ensure your browser has camera permissions
- Use HTTPS or localhost (required by modern browsers)
- Check if another application is using the camera

## 📊 Features Available

### ✅ Currently Working:
- Image upload and analysis
- Demo predictions (simulated with realistic scores)
- All page navigation
- Contact form
- Statistics dashboard
- Training interface

### 🔄 Requires Training:
- Actual ML model predictions (currently in demo mode)
- Real-time camera detection with trained model
- Gallery detection accuracy (requires trained model)

## 🎓 Training a Model

1. Navigate to: http://localhost:5000/training
2. Configure training parameters:
   - Epochs: 10-50 (start with 10)
   - Batch size: 32
   - Learning rate: 0.0001
3. Click "Start Training"
4. Wait for training to complete
5. Model will be saved automatically

## 🔧 Environment Variables

Optional environment variables:
```bash
PORT=5000                    # Server port (default: 5000)
UPLOAD_FOLDER=/tmp/uploads   # Upload directory
MAX_CONTENT_LENGTH=10MB      # Max file size
```

## 💡 Tips

1. **First Run**: The app works immediately in demo mode
2. **Production**: For deployment, use Vercel or Railway (configs included)
3. **Development**: Flask debug mode is disabled by default for security
4. **Performance**: First prediction may be slow due to model loading

## 📞 Support

- **Developer**: Omkar Kalagi
- **Email**: omkardigambar4@gmail.com
- **GitHub**: [@omkarkalagi](https://github.com/omkarkalagi)

---

**Made with ❤️ by Omkar Kalagi**
