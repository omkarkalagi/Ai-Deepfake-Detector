# 🚀 Deployment Guide - AI Deepfake Detector

This guide will help you deploy the AI Deepfake Detector to Vercel.

## 📋 Prerequisites

Before deploying, ensure you have:
- ✅ A GitHub account
- ✅ A Vercel account (free tier works fine)
- ✅ Git installed on your computer
- ✅ Your code pushed to GitHub

## 🌐 Deploy to Vercel (Recommended)

### Method 1: Deploy via Vercel Dashboard (Easiest)

1. **Push your code to GitHub**
   ```bash
   cd C:\Projects\deepfake-detector
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Go to Vercel**
   - Visit [vercel.com](https://vercel.com)
   - Click "Sign Up" or "Log In"
   - Choose "Continue with GitHub"

3. **Import Your Project**
   - Click "Add New..." → "Project"
   - Select your GitHub repository: `Ai-Deepfake-Detector`
   - Click "Import"

4. **Configure Project**
   - **Framework Preset**: Other
   - **Root Directory**: `./` (leave as default)
   - **Build Command**: Leave empty
   - **Output Directory**: Leave empty
   - Click "Deploy"

5. **Wait for Deployment**
   - Vercel will automatically detect the `vercel.json` configuration
   - Deployment typically takes 2-3 minutes
   - You'll get a live URL like: `https://your-project.vercel.app`

### Method 2: Deploy via Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   cd C:\Projects\deepfake-detector
   vercel --prod
   ```

4. **Follow the prompts**
   - Set up and deploy: Yes
   - Which scope: Your account
   - Link to existing project: No
   - Project name: ai-deepfake-detector
   - Directory: ./
   - Override settings: No

## ⚙️ Configuration Details

### vercel.json Explained

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/static/(.*)",
      "dest": "/static/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/api/index.py"
    }
  ],
  "functions": {
    "api/index.py": {
      "maxDuration": 60,
      "memory": 3008
    }
  }
}
```

**Key Settings:**
- `maxDuration: 60` - Functions can run up to 60 seconds
- `memory: 3008` - Maximum memory allocation (3GB)
- Routes all requests through `api/index.py`
- Serves static files from `/static` directory

### requirements.txt

Vercel automatically installs dependencies from `requirements.txt`:
```
Flask>=3.0.0
Flask-CORS>=4.0.0
Pillow>=10.4.0
numpy>=1.26.0
opencv-python-headless>=4.9.0
tensorflow>=2.15.0
Werkzeug>=3.0.0
requests>=2.31.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

## 🔧 Environment Variables (Optional)

If you need to set environment variables:

1. Go to your Vercel project dashboard
2. Click "Settings" → "Environment Variables"
3. Add variables:
   - `PORT` (default: 5000)
   - `MAX_CONTENT_LENGTH` (default: 10MB)
   - Any other custom variables

## 🐛 Troubleshooting

### Issue: Build Fails

**Solution:**
- Check that `requirements.txt` is in the root directory
- Ensure `api/index.py` exists
- Verify `vercel.json` syntax is correct

### Issue: 404 Errors

**Solution:**
- Check routes in `vercel.json`
- Ensure all template files are in `templates/` folder
- Verify static files are in `static/` folder

### Issue: Function Timeout

**Solution:**
- Increase `maxDuration` in `vercel.json` (max 60s on free tier)
- Optimize image processing code
- Consider using smaller models

### Issue: Memory Limit Exceeded

**Solution:**
- Increase `memory` in `vercel.json` (max 3008MB on free tier)
- Optimize TensorFlow model loading
- Use `opencv-python-headless` instead of `opencv-python`

### Issue: Static Files Not Loading

**Solution:**
- Ensure static files are in `static/` directory
- Check route configuration in `vercel.json`
- Verify file paths in templates use `url_for('static', filename='...')`

## 📊 Monitoring Your Deployment

### View Logs
1. Go to Vercel dashboard
2. Select your project
3. Click on a deployment
4. View "Functions" tab for logs

### Check Performance
- Monitor response times in Vercel dashboard
- Check function execution duration
- Review error rates

## 🔄 Updating Your Deployment

### Automatic Deployments
Vercel automatically deploys when you push to GitHub:
```bash
git add .
git commit -m "Update feature"
git push origin main
```

### Manual Deployments
Using Vercel CLI:
```bash
vercel --prod
```

## 🌟 Post-Deployment Checklist

- [ ] Test all pages (Home, Training, Statistics, About, Contact)
- [ ] Upload and analyze a test image
- [ ] Test training page functionality
- [ ] Verify all static assets load correctly
- [ ] Check mobile responsiveness
- [ ] Test API endpoints
- [ ] Monitor initial performance

## 📱 Custom Domain (Optional)

1. Go to Vercel project settings
2. Click "Domains"
3. Add your custom domain
4. Follow DNS configuration instructions
5. Wait for SSL certificate provisioning

## 🔒 Security Considerations

- ✅ HTTPS is enabled by default on Vercel
- ✅ Environment variables are encrypted
- ✅ File uploads are validated
- ✅ CORS is configured properly
- ⚠️ Consider adding rate limiting for production
- ⚠️ Implement user authentication if needed

## 💡 Performance Tips

1. **Optimize Images**: Compress images in `static/` folder
2. **Cache Static Assets**: Vercel automatically caches static files
3. **Lazy Load Models**: Load TensorFlow models only when needed
4. **Use CDN**: Vercel's edge network serves content globally
5. **Monitor Usage**: Check Vercel analytics regularly

## 📞 Support

If you encounter issues:
- Check [Vercel Documentation](https://vercel.com/docs)
- Review [Flask Deployment Guide](https://flask.palletsprojects.com/en/latest/deploying/)
- Contact: omkardigambar4@gmail.com

## 🎉 Success!

Once deployed, your app will be live at:
- **Production URL**: `https://your-project.vercel.app`
- **Custom Domain**: `https://your-domain.com` (if configured)

Share your deployment and start detecting deepfakes! 🚀

---

**Made with ❤️ by Omkar Kalagi**