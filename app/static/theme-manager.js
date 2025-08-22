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
 * Universal Theme Manager for AI Deepfake Detector
 * Handles light/dark theme switching across all pages
 */

class UniversalThemeManager {
    constructor() {
        this.currentTheme = localStorage.getItem('deepfake-detector-theme') || 'light';
        this.init();
    }

    init() {
        this.applyTheme(this.currentTheme);
        this.setupEventListeners();
        this.addThemeStyles();
    }

    setupEventListeners() {
        // Wait for DOM to be ready
        document.addEventListener('DOMContentLoaded', () => {
            const themeToggle = document.getElementById('themeToggle');
            if (themeToggle) {
                themeToggle.addEventListener('click', () => this.toggleTheme());
            }
        });
    }

    toggleTheme() {
        this.currentTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.applyTheme(this.currentTheme);
        localStorage.setItem('deepfake-detector-theme', this.currentTheme);
        
        // Dispatch custom event for other components to listen
        window.dispatchEvent(new CustomEvent('themeChanged', { 
            detail: { theme: this.currentTheme } 
        }));
    }

    applyTheme(theme) {
        const body = document.body;
        const themeIcon = document.getElementById('themeIcon');
        
        if (theme === 'dark') {
            body.classList.add('dark-theme');
            if (themeIcon) {
                themeIcon.className = 'fas fa-sun';
            }
        } else {
            body.classList.remove('dark-theme');
            if (themeIcon) {
                themeIcon.className = 'fas fa-moon';
            }
        }

        // Update charts if they exist
        this.updateChartColors(theme);
    }

    addThemeStyles() {
        // Add universal theme styles if not already present
        if (!document.getElementById('universal-theme-styles')) {
            const style = document.createElement('style');
            style.id = 'universal-theme-styles';
            style.textContent = `
                /* Universal Theme Styles */
                body {
                    transition: all 0.3s ease;
                }

                body.dark-theme {
                    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                    color: #ecf0f1;
                }

                .dark-theme .navbar {
                    background: rgba(44, 62, 80, 0.95) !important;
                    box-shadow: 0 2px 20px rgba(0, 0, 0, 0.3);
                }

                .dark-theme .main-container {
                    background: rgba(44, 62, 80, 0.95);
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                }

                .dark-theme .navbar-brand,
                .dark-theme .nav-link {
                    color: #ecf0f1 !important;
                }

                .dark-theme .nav-link:hover {
                    color: #3498db !important;
                }

                .dark-theme .nav-link.active {
                    color: #e74c3c !important;
                }

                .dark-theme .upload-section,
                .dark-theme .contact-form,
                .dark-theme .api-section,
                .dark-theme .control-panel,
                .dark-theme .stats-panel,
                .dark-theme .progress-container,
                .dark-theme .results-section,
                .dark-theme .chart-container,
                .dark-theme .file-list,
                .dark-theme .filter-section,
                .dark-theme .contact-section,
                .dark-theme .sdk-section {
                    background: rgba(44, 62, 80, 0.9);
                    color: #ecf0f1;
                }

                .dark-theme .result-card,
                .dark-theme .feature-item,
                .dark-theme .model-info,
                .dark-theme .endpoint-card,
                .dark-theme .gallery-item,
                .dark-theme .contact-info,
                .dark-theme .faq-item,
                .dark-theme .team-member,
                .dark-theme .alert-panel,
                .dark-theme .settings-panel {
                    background: rgba(44, 62, 80, 0.8);
                    color: #ecf0f1;
                }

                .dark-theme .form-control,
                .dark-theme .form-select {
                    background: rgba(52, 73, 94, 0.8);
                    border-color: #5d6d7e;
                    color: #ecf0f1;
                }

                .dark-theme .form-control:focus,
                .dark-theme .form-select:focus {
                    background: rgba(52, 73, 94, 0.9);
                    border-color: #3498db;
                    color: #ecf0f1;
                    box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
                }

                .dark-theme .btn-outline-primary {
                    color: #3498db;
                    border-color: #3498db;
                }

                .dark-theme .btn-outline-primary:hover {
                    background: #3498db;
                    border-color: #3498db;
                }

                .dark-theme .table {
                    color: #ecf0f1;
                }

                .dark-theme .table th {
                    border-color: #5d6d7e;
                    background: rgba(52, 73, 94, 0.5);
                }

                .dark-theme .table td {
                    border-color: #5d6d7e;
                }

                .dark-theme .dropdown-menu {
                    background: rgba(44, 62, 80, 0.95);
                    border-color: #5d6d7e;
                }

                .dark-theme .dropdown-item {
                    color: #ecf0f1;
                }

                .dark-theme .dropdown-item:hover {
                    background: rgba(52, 152, 219, 0.2);
                    color: #3498db;
                }

                .dark-theme .alert {
                    background: rgba(44, 62, 80, 0.9);
                    border-color: #5d6d7e;
                    color: #ecf0f1;
                }

                .dark-theme .modal-content {
                    background: rgba(44, 62, 80, 0.95);
                    color: #ecf0f1;
                }

                .dark-theme .modal-header {
                    border-color: #5d6d7e;
                }

                .dark-theme .modal-footer {
                    border-color: #5d6d7e;
                }

                .dark-theme .navbar-collapse {
                    background: rgba(44, 62, 80, 0.95) !important;
                }

                /* Theme toggle button */
                .theme-toggle {
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    border: none;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    color: white;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    margin-left: 15px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .theme-toggle:hover {
                    transform: scale(1.1);
                    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
                }

                .dark-theme .theme-toggle {
                    background: linear-gradient(135deg, #f39c12, #e67e22);
                }

                /* Mobile responsive theme toggle */
                @media (max-width: 768px) {
                    .theme-toggle {
                        margin: 10px auto;
                        position: relative;
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }

    updateChartColors(theme) {
        const isDark = theme === 'dark';
        const textColor = isDark ? '#ecf0f1' : '#333';
        const gridColor = isDark ? '#5d6d7e' : '#e9ecef';

        // Dispatch event for charts to update
        window.dispatchEvent(new CustomEvent('updateChartColors', {
            detail: { 
                theme: theme,
                textColor: textColor,
                gridColor: gridColor,
                isDark: isDark
            }
        }));
    }

    // Method to get current theme
    getCurrentTheme() {
        return this.currentTheme;
    }

    // Method to set theme programmatically
    setTheme(theme) {
        if (theme === 'light' || theme === 'dark') {
            this.currentTheme = theme;
            this.applyTheme(theme);
            localStorage.setItem('deepfake-detector-theme', theme);
        }
    }
}

// Initialize theme manager
const themeManager = new UniversalThemeManager();

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UniversalThemeManager;
}

// Make available globally
window.ThemeManager = themeManager;
