// Performance optimizations
document.addEventListener('DOMContentLoaded', function() {
    // Lazy load images
    const images = document.querySelectorAll('img[data-src]');
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                imageObserver.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
    
    // Preload critical resources
    const criticalResources = [
        '/static/theme-manager.js',
        '/static/advanced-loader.js',
        '/static/interactive-enhancements.js'
    ];
    
    criticalResources.forEach(resource => {
        const link = document.createElement('link');
        link.rel = 'preload';
        link.as = 'script';
        link.href = resource;
        document.head.appendChild(link);
    });
});

/**
 * Advanced Loading Animations System
 * Creates beautiful loading experiences for the AI Deepfake Detector
 */

class AdvancedLoader {
    constructor() {
        this.isLoading = false;
        this.loadingOverlay = null;
        this.init();
    }

    init() {
        this.createLoadingOverlay();
        this.setupPageTransitions();
        this.setupFormSubmissions();
        this.setupImageUploads();
        this.showInitialLoader();
    }

    createLoadingOverlay() {
        // Create main loading overlay
        this.loadingOverlay = document.createElement('div');
        this.loadingOverlay.id = 'advanced-loader';
        this.loadingOverlay.innerHTML = `
            <div class="loader-container">
                <div class="ai-brain-loader">
                    <div class="brain-core">
                        <div class="neural-network">
                            <div class="neuron"></div>
                            <div class="neuron"></div>
                            <div class="neuron"></div>
                            <div class="neuron"></div>
                            <div class="neuron"></div>
                        </div>
                        <div class="brain-waves">
                            <div class="wave"></div>
                            <div class="wave"></div>
                            <div class="wave"></div>
                        </div>
                    </div>
                </div>
                <div class="loader-text">
                    <h3 id="loader-title">AI Deepfake Detector</h3>
                    <p id="loader-subtitle">Initializing Neural Networks...</p>
                    <div class="progress-bar">
                        <div class="progress-fill"></div>
                    </div>
                    <div class="loader-stats">
                        <span id="loader-percentage">0%</span>
                        <span id="loader-status">Loading</span>
                    </div>
                </div>
            </div>
            <div class="loader-particles">
                ${this.generateParticles(20)}
            </div>
        `;

        // Add CSS styles
        const style = document.createElement('style');
        style.textContent = `
            #advanced-loader {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
                opacity: 1;
                transition: opacity 0.5s ease-out;
            }

            #advanced-loader.fade-out {
                opacity: 0;
                pointer-events: none;
            }

            .loader-container {
                text-align: center;
                color: white;
                position: relative;
                z-index: 2;
            }

            .ai-brain-loader {
                width: 120px;
                height: 120px;
                margin: 0 auto 2rem;
                position: relative;
            }

            .brain-core {
                width: 100%;
                height: 100%;
                border: 3px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                position: relative;
                animation: brainRotate 3s linear infinite;
            }

            @keyframes brainRotate {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .neural-network {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 80px;
                height: 80px;
            }

            .neuron {
                position: absolute;
                width: 12px;
                height: 12px;
                background: #fff;
                border-radius: 50%;
                animation: neuronPulse 2s ease-in-out infinite;
            }

            .neuron:nth-child(1) {
                top: 0;
                left: 50%;
                transform: translateX(-50%);
                animation-delay: 0s;
            }

            .neuron:nth-child(2) {
                top: 20px;
                right: 10px;
                animation-delay: 0.4s;
            }

            .neuron:nth-child(3) {
                bottom: 20px;
                right: 10px;
                animation-delay: 0.8s;
            }

            .neuron:nth-child(4) {
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                animation-delay: 1.2s;
            }

            .neuron:nth-child(5) {
                top: 20px;
                left: 10px;
                animation-delay: 1.6s;
            }

            @keyframes neuronPulse {
                0%, 100% {
                    opacity: 0.3;
                    transform: scale(1);
                }
                50% {
                    opacity: 1;
                    transform: scale(1.5);
                    box-shadow: 0 0 20px rgba(255, 255, 255, 0.8);
                }
            }

            .brain-waves {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 140px;
                height: 140px;
            }

            .wave {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                border: 2px solid rgba(255, 255, 255, 0.2);
                border-radius: 50%;
                animation: waveExpand 3s ease-out infinite;
            }

            .wave:nth-child(2) {
                animation-delay: 1s;
            }

            .wave:nth-child(3) {
                animation-delay: 2s;
            }

            @keyframes waveExpand {
                0% {
                    transform: scale(0.8);
                    opacity: 1;
                }
                100% {
                    transform: scale(1.5);
                    opacity: 0;
                }
            }

            .loader-text h3 {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            }

            .loader-text p {
                font-size: 1.1rem;
                margin-bottom: 1.5rem;
                opacity: 0.9;
            }

            .progress-bar {
                width: 300px;
                height: 6px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 3px;
                margin: 0 auto 1rem;
                overflow: hidden;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #fff, #f39c12, #fff);
                border-radius: 3px;
                width: 0%;
                animation: progressFill 3s ease-out forwards;
            }

            @keyframes progressFill {
                0% { width: 0%; }
                100% { width: 100%; }
            }

            .loader-stats {
                display: flex;
                justify-content: space-between;
                width: 300px;
                margin: 0 auto;
                font-size: 0.9rem;
                opacity: 0.8;
            }

            .loader-particles {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 1;
            }

            .particle {
                position: absolute;
                width: 4px;
                height: 4px;
                background: rgba(255, 255, 255, 0.6);
                border-radius: 50%;
                animation: particleFloat 6s linear infinite;
            }

            @keyframes particleFloat {
                0% {
                    transform: translateY(100vh) rotate(0deg);
                    opacity: 0;
                }
                10% {
                    opacity: 1;
                }
                90% {
                    opacity: 1;
                }
                100% {
                    transform: translateY(-100px) rotate(360deg);
                    opacity: 0;
                }
            }

            /* Mobile responsiveness */
            @media (max-width: 768px) {
                .ai-brain-loader {
                    width: 80px;
                    height: 80px;
                }

                .neural-network {
                    width: 60px;
                    height: 60px;
                }

                .brain-waves {
                    width: 100px;
                    height: 100px;
                }

                .progress-bar,
                .loader-stats {
                    width: 250px;
                }

                .loader-text h3 {
                    font-size: 1.5rem;
                }
            }
        `;

        document.head.appendChild(style);
        document.body.appendChild(this.loadingOverlay);
    }

    generateParticles(count) {
        let particles = '';
        for (let i = 0; i < count; i++) {
            const left = Math.random() * 100;
            const delay = Math.random() * 6;
            const duration = 6 + Math.random() * 4;
            particles += `<div class="particle" style="left: ${left}%; animation-delay: ${delay}s; animation-duration: ${duration}s;"></div>`;
        }
        return particles;
    }

    showInitialLoader() {
        // Show loader on page load
        this.show('Initializing AI Systems...', 'Loading neural networks and models...');
        
        // Simulate loading progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress >= 100) {
                progress = 100;
                clearInterval(progressInterval);
                setTimeout(() => this.hide(), 500);
            }
            this.updateProgress(progress);
        }, 200);
    }

    show(title = 'Processing...', subtitle = 'Please wait...') {
        if (this.isLoading) return;
        
        this.isLoading = true;
        document.getElementById('loader-title').textContent = title;
        document.getElementById('loader-subtitle').textContent = subtitle;
        this.loadingOverlay.classList.remove('fade-out');
        this.loadingOverlay.style.display = 'flex';
        
        // Reset progress
        this.updateProgress(0);
    }

    hide() {
        if (!this.isLoading) return;
        
        this.isLoading = false;
        this.loadingOverlay.classList.add('fade-out');
        
        setTimeout(() => {
            this.loadingOverlay.style.display = 'none';
        }, 500);
    }

    updateProgress(percentage) {
        const progressFill = document.querySelector('.progress-fill');
        const percentageSpan = document.getElementById('loader-percentage');
        const statusSpan = document.getElementById('loader-status');
        
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }
        
        if (percentageSpan) {
            percentageSpan.textContent = `${Math.round(percentage)}%`;
        }
        
        if (statusSpan) {
            if (percentage < 30) {
                statusSpan.textContent = 'Initializing...';
            } else if (percentage < 60) {
                statusSpan.textContent = 'Loading Models...';
            } else if (percentage < 90) {
                statusSpan.textContent = 'Optimizing...';
            } else {
                statusSpan.textContent = 'Ready!';
            }
        }
    }

    setupPageTransitions() {
        // Add loading for navigation links
        document.addEventListener('click', (e) => {
            const link = e.target.closest('a[href]');
            if (link && !link.href.includes('#') && !link.href.includes('tel:') && !link.href.includes('mailto:')) {
                const href = link.getAttribute('href');
                if (href && !href.startsWith('http') && !href.startsWith('//')) {
                    this.show('Navigating...', 'Loading page content...');
                }
            }
        });
    }

    setupFormSubmissions() {
        // Add loading for form submissions
        document.addEventListener('submit', (e) => {
            const form = e.target;
            if (form.tagName === 'FORM') {
                this.show('Processing...', 'Analyzing your request...');
            }
        });
    }

    setupImageUploads() {
        // Add loading for image uploads
        document.addEventListener('change', (e) => {
            if (e.target.type === 'file' && e.target.accept && e.target.accept.includes('image')) {
                if (e.target.files.length > 0) {
                    this.show('Uploading Image...', 'Preparing for analysis...');
                    
                    // Simulate upload progress
                    let progress = 0;
                    const uploadInterval = setInterval(() => {
                        progress += Math.random() * 20;
                        if (progress >= 100) {
                            progress = 100;
                            clearInterval(uploadInterval);
                            setTimeout(() => this.hide(), 1000);
                        }
                        this.updateProgress(progress);
                    }, 300);
                }
            }
        });
    }

    // Public methods for manual control
    showAnalyzing() {
        this.show('Analyzing Image...', 'AI is detecting deepfake patterns...');
    }

    showTraining() {
        this.show('Training Model...', 'Optimizing neural network parameters...');
    }

    showBatchProcessing() {
        this.show('Batch Processing...', 'Analyzing multiple images...');
    }
}

// Initialize the advanced loader when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.advancedLoader = new AdvancedLoader();
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdvancedLoader;
}
