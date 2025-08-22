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
 * Interactive Enhancements for AI Deepfake Detector
 * Creates engaging user interactions and animations
 */

class InteractiveEnhancements {
    constructor() {
        this.init();
    }

    init() {
        this.setupParallaxEffects();
        this.setupHoverAnimations();
        this.setupScrollAnimations();
        this.setupInteractiveCards();
        this.setupFloatingElements();
        this.setupMouseTracker();
        this.setupTypingAnimations();
        this.setupCounterAnimations();
        this.setupImageHoverEffects();
        this.setupButtonRippleEffects();
    }

    setupParallaxEffects() {
        // Create floating background elements
        const floatingElements = document.createElement('div');
        floatingElements.className = 'floating-bg-elements';
        floatingElements.innerHTML = `
            <div class="floating-shape shape-1"></div>
            <div class="floating-shape shape-2"></div>
            <div class="floating-shape shape-3"></div>
            <div class="floating-shape shape-4"></div>
            <div class="floating-shape shape-5"></div>
        `;
        document.body.appendChild(floatingElements);

        // Parallax scroll effect
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const shapes = document.querySelectorAll('.floating-shape');
            
            shapes.forEach((shape, index) => {
                const speed = 0.5 + (index * 0.1);
                const yPos = -(scrolled * speed);
                shape.style.transform = `translateY(${yPos}px) rotate(${scrolled * 0.1}deg)`;
            });
        });
    }

    setupHoverAnimations() {
        // Enhanced hover effects for cards and buttons
        const interactiveElements = document.querySelectorAll('.card, .btn, .nav-link, .gallery-item');
        
        interactiveElements.forEach(element => {
            element.addEventListener('mouseenter', (e) => {
                e.target.style.transform = 'translateY(-5px) scale(1.02)';
                e.target.style.boxShadow = '0 15px 35px rgba(0, 0, 0, 0.2)';
                e.target.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
            });

            element.addEventListener('mouseleave', (e) => {
                e.target.style.transform = 'translateY(0) scale(1)';
                e.target.style.boxShadow = '';
            });
        });
    }

    setupScrollAnimations() {
        // Intersection Observer for scroll animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, observerOptions);

        // Observe elements for animation
        const animateElements = document.querySelectorAll('.card, .stat-card, .team-member, .gallery-item');
        animateElements.forEach(el => {
            el.classList.add('animate-on-scroll');
            observer.observe(el);
        });
    }

    setupInteractiveCards() {
        // 3D tilt effect for cards
        const cards = document.querySelectorAll('.card, .stat-card, .team-member');
        
        cards.forEach(card => {
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                
                const rotateX = (y - centerY) / 10;
                const rotateY = (centerX - x) / 10;
                
                card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(10px)`;
            });

            card.addEventListener('mouseleave', () => {
                card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateZ(0)';
            });
        });
    }

    setupFloatingElements() {
        // Create floating AI-themed icons
        const floatingIcons = ['🧠', '🤖', '👁️', '⚡', '🔬', '💡'];
        
        floatingIcons.forEach((icon, index) => {
            const floatingIcon = document.createElement('div');
            floatingIcon.className = 'floating-icon';
            floatingIcon.textContent = icon;
            floatingIcon.style.cssText = `
                position: fixed;
                font-size: 2rem;
                opacity: 0.1;
                pointer-events: none;
                z-index: 1;
                animation: float ${5 + index}s ease-in-out infinite;
                animation-delay: ${index * 0.5}s;
                left: ${Math.random() * 100}%;
                top: ${Math.random() * 100}%;
            `;
            document.body.appendChild(floatingIcon);
        });
    }

    setupMouseTracker() {
        // Create mouse follower effect
        const mouseFollower = document.createElement('div');
        mouseFollower.className = 'mouse-follower';
        mouseFollower.style.cssText = `
            position: fixed;
            width: 20px;
            height: 20px;
            background: radial-gradient(circle, rgba(52, 152, 219, 0.3), transparent);
            border-radius: 50%;
            pointer-events: none;
            z-index: 9999;
            transition: transform 0.1s ease;
        `;
        document.body.appendChild(mouseFollower);

        document.addEventListener('mousemove', (e) => {
            mouseFollower.style.left = e.clientX - 10 + 'px';
            mouseFollower.style.top = e.clientY - 10 + 'px';
        });

        // Enhance mouse follower on interactive elements
        const interactiveElements = document.querySelectorAll('a, button, .card, input');
        interactiveElements.forEach(element => {
            element.addEventListener('mouseenter', () => {
                mouseFollower.style.transform = 'scale(2)';
                mouseFollower.style.background = 'radial-gradient(circle, rgba(231, 76, 60, 0.3), transparent)';
            });

            element.addEventListener('mouseleave', () => {
                mouseFollower.style.transform = 'scale(1)';
                mouseFollower.style.background = 'radial-gradient(circle, rgba(52, 152, 219, 0.3), transparent)';
            });
        });
    }

    setupTypingAnimations() {
        // Typing animation for text elements
        const typingElements = document.querySelectorAll('[data-typing]');
        
        typingElements.forEach(element => {
            const text = element.textContent;
            element.textContent = '';
            element.style.borderRight = '2px solid #3498db';
            
            let i = 0;
            const typeWriter = () => {
                if (i < text.length) {
                    element.textContent += text.charAt(i);
                    i++;
                    setTimeout(typeWriter, 100);
                } else {
                    element.style.borderRight = 'none';
                }
            };
            
            // Start typing when element is visible
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        setTimeout(typeWriter, 500);
                        observer.unobserve(element);
                    }
                });
            });
            
            observer.observe(element);
        });
    }

    setupCounterAnimations() {
        // Animated counters for statistics
        const counters = document.querySelectorAll('[data-counter]');
        
        counters.forEach(counter => {
            const target = parseInt(counter.getAttribute('data-counter'));
            const duration = 2000;
            const increment = target / (duration / 16);
            let current = 0;
            
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const updateCounter = () => {
                            current += increment;
                            if (current < target) {
                                counter.textContent = Math.floor(current);
                                requestAnimationFrame(updateCounter);
                            } else {
                                counter.textContent = target;
                            }
                        };
                        updateCounter();
                        observer.unobserve(counter);
                    }
                });
            });
            
            observer.observe(counter);
        });
    }

    setupImageHoverEffects() {
        // Enhanced image hover effects
        const images = document.querySelectorAll('img');
        
        images.forEach(img => {
            img.addEventListener('mouseenter', () => {
                img.style.transform = 'scale(1.05)';
                img.style.filter = 'brightness(1.1) contrast(1.1)';
                img.style.transition = 'all 0.3s ease';
            });

            img.addEventListener('mouseleave', () => {
                img.style.transform = 'scale(1)';
                img.style.filter = 'brightness(1) contrast(1)';
            });
        });
    }

    setupButtonRippleEffects() {
        // Ripple effect for buttons
        const buttons = document.querySelectorAll('button, .btn');
        
        buttons.forEach(button => {
            button.addEventListener('click', (e) => {
                const ripple = document.createElement('span');
                const rect = button.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;
                
                ripple.style.cssText = `
                    position: absolute;
                    width: ${size}px;
                    height: ${size}px;
                    left: ${x}px;
                    top: ${y}px;
                    background: rgba(255, 255, 255, 0.3);
                    border-radius: 50%;
                    transform: scale(0);
                    animation: ripple 0.6s linear;
                    pointer-events: none;
                `;
                
                button.style.position = 'relative';
                button.style.overflow = 'hidden';
                button.appendChild(ripple);
                
                setTimeout(() => {
                    ripple.remove();
                }, 600);
            });
        });
    }

    // Add dynamic CSS animations
    addDynamicStyles() {
        const style = document.createElement('style');
        style.textContent = `
            @keyframes float {
                0%, 100% { transform: translateY(0px) rotate(0deg); }
                50% { transform: translateY(-20px) rotate(180deg); }
            }

            @keyframes ripple {
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }

            .floating-bg-elements {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: -1;
            }

            .floating-shape {
                position: absolute;
                opacity: 0.05;
                animation: float 10s ease-in-out infinite;
            }

            .shape-1 {
                width: 100px;
                height: 100px;
                background: linear-gradient(45deg, #3498db, #9b59b6);
                border-radius: 50%;
                top: 10%;
                left: 10%;
            }

            .shape-2 {
                width: 80px;
                height: 80px;
                background: linear-gradient(45deg, #e74c3c, #f39c12);
                border-radius: 20%;
                top: 20%;
                right: 15%;
                animation-delay: -2s;
            }

            .shape-3 {
                width: 120px;
                height: 120px;
                background: linear-gradient(45deg, #27ae60, #2ecc71);
                border-radius: 30%;
                bottom: 20%;
                left: 20%;
                animation-delay: -4s;
            }

            .shape-4 {
                width: 60px;
                height: 60px;
                background: linear-gradient(45deg, #f39c12, #f1c40f);
                border-radius: 50%;
                top: 60%;
                right: 30%;
                animation-delay: -6s;
            }

            .shape-5 {
                width: 90px;
                height: 90px;
                background: linear-gradient(45deg, #9b59b6, #8e44ad);
                border-radius: 40%;
                bottom: 10%;
                right: 10%;
                animation-delay: -8s;
            }

            .animate-on-scroll {
                opacity: 0;
                transform: translateY(30px);
                transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            }

            .animate-on-scroll.animate-in {
                opacity: 1;
                transform: translateY(0);
            }

            .floating-icon {
                animation: float 6s ease-in-out infinite;
            }

            /* Enhanced hover states */
            .card:hover,
            .stat-card:hover,
            .team-member:hover {
                transform: translateY(-10px) scale(1.02);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
            }

            /* Smooth transitions for all interactive elements */
            * {
                transition: transform 0.3s ease, box-shadow 0.3s ease, filter 0.3s ease;
            }

            /* Glowing effect for important elements */
            .btn-primary:hover,
            .theme-toggle:hover {
                box-shadow: 0 0 20px rgba(52, 152, 219, 0.5);
            }

            /* Pulse animation for attention-grabbing elements */
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }

            .pulse {
                animation: pulse 2s ease-in-out infinite;
            }
        `;
        document.head.appendChild(style);
    }
}

// Initialize interactive enhancements when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const enhancements = new InteractiveEnhancements();
    enhancements.addDynamicStyles();
    
    // Add data attributes for animations
    const statsElements = document.querySelectorAll('.stat-card h3');
    statsElements.forEach((el, index) => {
        const value = el.textContent.replace(/[^\d]/g, '');
        if (value) {
            el.setAttribute('data-counter', value);
            el.textContent = '0';
        }
    });
    
    // Add typing animation to main headings
    const mainHeadings = document.querySelectorAll('h1.display-4');
    mainHeadings.forEach(heading => {
        heading.setAttribute('data-typing', 'true');
    });
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InteractiveEnhancements;
}
