# 🚀 Deployment Guide - AI Deepfake Detector

This guide will help you deploy your AI Deepfake Detector to Vercel and GitHub.

## 📋 Prerequisites

- GitHub account
- Vercel account (free)
- Git installed on your computer

## 🔧 Step 1: Create GitHub Repository

### Option A: Using GitHub Web Interface

1. **Go to GitHub.com** and sign in
2. **Click "New Repository"** (green button)
3. **Repository Settings:**
   - Repository name: `ai-deepfake-detector`
   - Description: `Advanced AI Deepfake Detector with real-time analysis and interactive visualizations`
   - Visibility: Public (recommended for Vercel free tier)
   - Initialize: Don't initialize (we already have files)

4. **Create Repository**

### Option B: Using GitHub CLI (if installed)
```bash
gh repo create ai-deepfake-detector --public --description "Advanced AI Deepfake Detector"
```

## 📤 Step 2: Push Code to GitHub

1. **Add GitHub remote** (replace `yourusername` with your GitHub username):
```bash
git remote add origin https://github.com/yourusername/ai-deepfake-detector.git
```

2. **Push to GitHub**:
```bash
git branch -M main
git push -u origin main
```

## 🌐 Step 3: Deploy to Vercel

### Quick Deploy (Recommended)
1. **Click the Deploy Button** in your GitHub repository README
2. **Or use this direct link**: [Deploy to Vercel](https://vercel.com/new)

### Manual Deployment

1. **Go to Vercel.com** and sign in with GitHub
2. **Click "New Project"**
3. **Import Git Repository:**
   - Select your `ai-deepfake-detector` repository
   - Click "Import"

4. **Configure Project:**
   - Framework Preset: `Other`
   - Root Directory: `./` (default)
   - Build Command: (leave empty)
   - Output Directory: (leave empty)
   - Install Command: `pip install -r requirements.txt`

5. **Environment Variables** (optional):
   ```
   FLASK_ENV=production
   ```

6. **Deploy:**
   - Click "Deploy"
   - Wait for deployment to complete (2-3 minutes)
   - Your app will be live at `https://your-app-name.vercel.app`

## ✅ Step 4: Verify Deployment

1. **Check your live URL**
2. **Test main features:**
   - Upload and analyze an image
   - Check gallery page
   - Verify training dashboard
   - Test statistics page

## 🔧 Configuration Files Explained

### `vercel.json`
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

### `requirements.txt`
- Contains all Python dependencies
- Optimized for Vercel deployment
- Uses `opencv-python-headless` for serverless compatibility

### `runtime.txt`
- Specifies Python version: `python-3.11.5`
- Ensures consistent runtime environment

## 🚨 Troubleshooting

### Common Issues:

1. **Build Fails - Dependencies**
   - Check `requirements.txt` for correct versions
   - Ensure `opencv-python-headless` is used (not `opencv-python`)

2. **Function Timeout**
   - Large model files may cause timeouts
   - Consider model optimization or caching

3. **Static Files Not Loading**
   - Verify file paths in templates
   - Check Vercel static file handling

4. **Memory Issues**
   - TensorFlow models can be memory-intensive
   - Consider model quantization for production

### Solutions:

1. **Check Vercel Logs:**
   - Go to Vercel Dashboard
   - Select your project
   - Check "Functions" tab for error logs

2. **Local Testing:**
   ```bash
   vercel dev
   ```

3. **Environment Variables:**
   - Add in Vercel Dashboard under "Settings" → "Environment Variables"

## 🔄 Continuous Deployment

Once connected to GitHub:
- **Automatic Deployments**: Every push to `main` branch triggers deployment
- **Preview Deployments**: Pull requests get preview URLs
- **Rollback**: Easy rollback to previous versions

## 📊 Performance Optimization

### For Production:
1. **Model Optimization:**
   - Use TensorFlow Lite for smaller models
   - Implement model caching
   - Consider edge deployment

2. **Static Assets:**
   - Optimize images and CSS
   - Use CDN for large assets
   - Implement proper caching headers

3. **Database:**
   - Consider adding database for analytics
   - Use Vercel KV for session storage

## 🔐 Security Considerations

1. **Environment Variables:**
   - Store sensitive data in Vercel environment variables
   - Never commit API keys or secrets

2. **File Uploads:**
   - Implement file size limits
   - Validate file types
   - Consider virus scanning

3. **Rate Limiting:**
   - Implement request rate limiting
   - Monitor usage patterns

## 📈 Monitoring & Analytics

1. **Vercel Analytics:**
   - Enable in project settings
   - Monitor performance and usage

2. **Error Tracking:**
   - Consider Sentry integration
   - Monitor application errors

3. **Custom Analytics:**
   - Track detection accuracy
   - Monitor user engagement

## 🎯 Next Steps

1. **Custom Domain:**
   - Add your custom domain in Vercel settings
   - Configure DNS records

2. **API Integration:**
   - Expose REST API endpoints
   - Add API documentation

3. **Mobile App:**
   - Create React Native or Flutter app
   - Use your Vercel deployment as backend

## 📞 Support

If you encounter issues:
1. Check Vercel documentation
2. Review GitHub repository issues
3. Contact support through Vercel dashboard

---

**🎉 Congratulations! Your AI Deepfake Detector is now live on the web!**

**🔗 Share your deployment:**
- GitHub: `https://github.com/yourusername/ai-deepfake-detector`
- Live App: `https://your-app-name.vercel.app`
