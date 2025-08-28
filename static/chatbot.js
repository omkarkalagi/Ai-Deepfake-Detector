/**
 * AI Deepfake Detector Chatbot
 * Fully functional chatbot with intelligent responses
 */

class DeepfakeChatbot {
    constructor() {
        this.isOpen = false;
        this.messages = [];
        this.responses = this.initializeResponses();
        this.init();
    }

    init() {
        this.createChatbotHTML();
        this.setupEventListeners();
        this.addWelcomeMessage();
    }

    initializeResponses() {
        return {
            greetings: [
                "Hello! I'm the AI Deepfake Detector assistant. How can I help you today?",
                "Hi there! I'm here to help you with deepfake detection. What would you like to know?",
                "Welcome! I can assist you with questions about our AI deepfake detection system."
            ],
            deepfake: [
                "Deepfakes are AI-generated synthetic media where a person appears to say or do things they never did. Our system can detect them with 92.7% accuracy!",
                "Our AI model analyzes facial features, lighting inconsistencies, and temporal artifacts to identify deepfakes. Would you like to try uploading an image?",
                "Deepfake detection is crucial for media authenticity. Our system uses advanced CNN architecture to identify manipulated content."
            ],
            accuracy: [
                "Our current model achieves 92.7% accuracy with 94.2% precision and 91.3% recall on our test dataset.",
                "We've trained our model on over 50,000 samples and continuously improve it with web-scraped data.",
                "The system provides confidence scores and detailed analysis for each detection."
            ],
            howto: [
                "To use our detector: 1) Go to the Home page, 2) Upload an image (PNG, JPG, JPEG), 3) Click 'Analyze Image', 4) View the results with confidence score!",
                "You can also use our API for batch processing or integrate it into your applications. Check out the API documentation!",
                "For real-time detection, visit our Real-time page where you can use your camera for live analysis."
            ],
            api: [
                "Our API is available at /api/detect. You can send POST requests with image files and get JSON responses with detection results.",
                "API features include: confidence scoring, feature extraction, batch processing, and detailed analysis reports.",
                "Check out our API Explorer page for interactive documentation and code examples in multiple languages."
            ],
            team: [
                "Our team is led by Omkar Digambar, Project Lead & AI Research Engineer. We have 9 talented developers working on this project.",
                "The team includes specialists in AI, deep learning, full-stack development, data engineering, DevOps, and security.",
                "You can contact our team lead Omkar at +91 7624828106 or omkardigambar4@gmail.com."
            ],
            contact: [
                "You can reach us at: Phone: +91 7624828106, Email: omkardigambar4@gmail.com",
                "Follow us on social media: LinkedIn, GitHub, YouTube, Instagram (@omkar_kalagi), WhatsApp, and Telegram.",
                "Visit our Contact page for a detailed contact form and team information."
            ],
            features: [
                "Our system features: Real-time detection, Batch processing, API integration, Gallery of examples, Training interface, and Performance statistics.",
                "Advanced features include: Confidence scoring, Feature extraction, Web dataset training, Mobile-responsive design, and Dark/Light themes.",
                "We also provide comprehensive documentation, interactive API explorer, and detailed performance metrics."
            ],
            help: [
                "I can help you with: Deepfake detection questions, How to use the system, API documentation, Team information, Technical support, and Feature explanations.",
                "Try asking me about: 'How does deepfake detection work?', 'What's your accuracy?', 'How to use the API?', or 'Contact information'.",
                "For technical issues, you can also contact our support team directly through the Contact page."
            ]
        };
    }

    createChatbotHTML() {
        const chatbotHTML = `
            <div id="chatbot-container" class="chatbot-container">
                <div id="chatbot-toggle" class="chatbot-toggle">
                    <i class="fas fa-robot"></i>
                    <span class="chatbot-badge">AI</span>
                </div>
                
                <div id="chatbot-window" class="chatbot-window">
                    <div class="chatbot-header">
                        <div class="chatbot-title">
                            <i class="fas fa-robot me-2"></i>
                            <span>AI Assistant</span>
                        </div>
                        <div class="chatbot-status">
                            <span class="status-dot"></span>
                            <span>Online</span>
                        </div>
                        <button id="chatbot-close" class="chatbot-close">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    
                    <div id="chatbot-messages" class="chatbot-messages">
                        <!-- Messages will be added here -->
                    </div>
                    
                    <div class="chatbot-input-container">
                        <input type="text" id="chatbot-input" class="chatbot-input" placeholder="Ask me anything about deepfake detection..." maxlength="500">
                        <button id="chatbot-send" class="chatbot-send">
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                    
                    <div class="chatbot-quick-actions">
                        <button class="quick-action" data-message="How does deepfake detection work?">
                            <i class="fas fa-brain"></i> How it works
                        </button>
                        <button class="quick-action" data-message="What's your accuracy?">
                            <i class="fas fa-chart-line"></i> Accuracy
                        </button>
                        <button class="quick-action" data-message="How to use the API?">
                            <i class="fas fa-code"></i> API Help
                        </button>
                        <button class="quick-action" data-message="Contact information">
                            <i class="fas fa-phone"></i> Contact
                        </button>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', chatbotHTML);
        this.addChatbotStyles();
    }

    addChatbotStyles() {
        const styles = `
            <style>
                .chatbot-container {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    z-index: 10000;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }

                .chatbot-toggle {
                    width: 60px;
                    height: 60px;
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 24px;
                    cursor: pointer;
                    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
                    transition: all 0.3s ease;
                    position: relative;
                    animation: pulse 2s ease-in-out infinite;
                }

                .chatbot-toggle:hover {
                    transform: scale(1.1);
                    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
                }

                .chatbot-badge {
                    position: absolute;
                    top: -5px;
                    right: -5px;
                    background: #e74c3c;
                    color: white;
                    font-size: 10px;
                    font-weight: bold;
                    padding: 2px 6px;
                    border-radius: 10px;
                    animation: bounce 1s ease-in-out infinite;
                }

                @keyframes pulse {
                    0%, 100% { transform: scale(1); }
                    50% { transform: scale(1.05); }
                }

                @keyframes bounce {
                    0%, 100% { transform: translateY(0); }
                    50% { transform: translateY(-3px); }
                }

                .chatbot-window {
                    position: absolute;
                    bottom: 80px;
                    right: 0;
                    width: 350px;
                    height: 500px;
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
                    display: none;
                    flex-direction: column;
                    overflow: hidden;
                    transform: translateY(20px);
                    opacity: 0;
                    transition: all 0.3s ease;
                }

                .chatbot-window.open {
                    display: flex;
                    transform: translateY(0);
                    opacity: 1;
                }

                .chatbot-header {
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                    padding: 15px 20px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }

                .chatbot-title {
                    display: flex;
                    align-items: center;
                    font-weight: 600;
                    font-size: 16px;
                }

                .chatbot-status {
                    display: flex;
                    align-items: center;
                    font-size: 12px;
                    opacity: 0.9;
                }

                .status-dot {
                    width: 8px;
                    height: 8px;
                    background: #2ecc71;
                    border-radius: 50%;
                    margin-right: 5px;
                    animation: blink 2s ease-in-out infinite;
                }

                @keyframes blink {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }

                .chatbot-close {
                    background: none;
                    border: none;
                    color: white;
                    font-size: 16px;
                    cursor: pointer;
                    padding: 5px;
                    border-radius: 50%;
                    transition: all 0.3s ease;
                }

                .chatbot-close:hover {
                    background: rgba(255, 255, 255, 0.2);
                }

                .chatbot-messages {
                    flex: 1;
                    padding: 20px;
                    overflow-y: auto;
                    background: #f8f9fa;
                }

                .message {
                    margin-bottom: 15px;
                    animation: slideIn 0.3s ease;
                }

                @keyframes slideIn {
                    from { transform: translateY(10px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }

                .message.user {
                    text-align: right;
                }

                .message.bot {
                    text-align: left;
                }

                .message-bubble {
                    display: inline-block;
                    max-width: 80%;
                    padding: 12px 16px;
                    border-radius: 18px;
                    font-size: 14px;
                    line-height: 1.4;
                    word-wrap: break-word;
                }

                .message.user .message-bubble {
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    color: white;
                }

                .message.bot .message-bubble {
                    background: white;
                    color: #333;
                    border: 1px solid #e9ecef;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }

                .message-time {
                    font-size: 11px;
                    color: #95a5a6;
                    margin-top: 5px;
                }

                .chatbot-input-container {
                    display: flex;
                    padding: 15px 20px;
                    background: white;
                    border-top: 1px solid #e9ecef;
                }

                .chatbot-input {
                    flex: 1;
                    border: 1px solid #e9ecef;
                    border-radius: 25px;
                    padding: 12px 16px;
                    font-size: 14px;
                    outline: none;
                    transition: all 0.3s ease;
                }

                .chatbot-input:focus {
                    border-color: #667eea;
                    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                }

                .chatbot-send {
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    border: none;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    color: white;
                    margin-left: 10px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .chatbot-send:hover {
                    transform: scale(1.1);
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                }

                .chatbot-quick-actions {
                    padding: 15px 20px;
                    background: white;
                    border-top: 1px solid #e9ecef;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                }

                .quick-action {
                    background: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 15px;
                    padding: 6px 12px;
                    font-size: 12px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    gap: 5px;
                }

                .quick-action:hover {
                    background: #667eea;
                    color: white;
                    border-color: #667eea;
                }

                .typing-indicator {
                    display: flex;
                    align-items: center;
                    gap: 5px;
                    padding: 12px 16px;
                    background: white;
                    border-radius: 18px;
                    border: 1px solid #e9ecef;
                    margin-bottom: 15px;
                }

                .typing-dot {
                    width: 8px;
                    height: 8px;
                    background: #95a5a6;
                    border-radius: 50%;
                    animation: typing 1.4s ease-in-out infinite;
                }

                .typing-dot:nth-child(2) { animation-delay: 0.2s; }
                .typing-dot:nth-child(3) { animation-delay: 0.4s; }

                @keyframes typing {
                    0%, 60%, 100% { transform: translateY(0); }
                    30% { transform: translateY(-10px); }
                }

                /* Dark theme support */
                .dark-theme .chatbot-window {
                    background: #2c3e50;
                }

                .dark-theme .chatbot-messages {
                    background: #34495e;
                }

                .dark-theme .message.bot .message-bubble {
                    background: #2c3e50;
                    color: #ecf0f1;
                    border-color: #5d6d7e;
                }

                .dark-theme .chatbot-input-container,
                .dark-theme .chatbot-quick-actions {
                    background: #2c3e50;
                    border-color: #5d6d7e;
                }

                .dark-theme .chatbot-input {
                    background: #34495e;
                    color: #ecf0f1;
                    border-color: #5d6d7e;
                }

                .dark-theme .quick-action {
                    background: #34495e;
                    color: #ecf0f1;
                    border-color: #5d6d7e;
                }

                /* Mobile responsiveness */
                @media (max-width: 768px) {
                    .chatbot-window {
                        width: 300px;
                        height: 450px;
                        bottom: 70px;
                        right: -10px;
                    }
                    
                    .chatbot-container {
                        bottom: 15px;
                        right: 15px;
                    }
                }
            </style>
        `;

        document.head.insertAdjacentHTML('beforeend', styles);
    }

    setupEventListeners() {
        const toggle = document.getElementById('chatbot-toggle');
        const close = document.getElementById('chatbot-close');
        const send = document.getElementById('chatbot-send');
        const input = document.getElementById('chatbot-input');
        const quickActions = document.querySelectorAll('.quick-action');

        toggle.addEventListener('click', () => this.toggleChatbot());
        close.addEventListener('click', () => this.closeChatbot());
        send.addEventListener('click', () => this.sendMessage());
        
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });

        quickActions.forEach(action => {
            action.addEventListener('click', () => {
                const message = action.getAttribute('data-message');
                this.sendUserMessage(message);
            });
        });
    }

    toggleChatbot() {
        const window = document.getElementById('chatbot-window');
        this.isOpen = !this.isOpen;
        
        if (this.isOpen) {
            window.classList.add('open');
            document.getElementById('chatbot-input').focus();
        } else {
            window.classList.remove('open');
        }
    }

    closeChatbot() {
        const window = document.getElementById('chatbot-window');
        window.classList.remove('open');
        this.isOpen = false;
    }

    addWelcomeMessage() {
        setTimeout(() => {
            this.addBotMessage("👋 Hello! I'm your AI assistant for the Deepfake Detector. I can help you with questions about our system, API, team, and more. How can I assist you today?");
        }, 1000);
    }

    sendMessage() {
        const input = document.getElementById('chatbot-input');
        const message = input.value.trim();
        
        if (message) {
            this.sendUserMessage(message);
            input.value = '';
        }
    }

    sendUserMessage(message) {
        this.addUserMessage(message);
        this.showTypingIndicator();
        
        setTimeout(() => {
            this.hideTypingIndicator();
            const response = this.generateResponse(message);
            this.addBotMessage(response);
        }, 1000 + Math.random() * 1000); // Random delay for realism
    }

    addUserMessage(message) {
        const messagesContainer = document.getElementById('chatbot-messages');
        const messageElement = this.createMessageElement(message, 'user');
        messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }

    addBotMessage(message) {
        const messagesContainer = document.getElementById('chatbot-messages');
        const messageElement = this.createMessageElement(message, 'bot');
        messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }

    createMessageElement(message, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.innerHTML = `
            <div class="message-bubble">${message}</div>
            <div class="message-time">${time}</div>
        `;
        
        return messageDiv;
    }

    showTypingIndicator() {
        const messagesContainer = document.getElementById('chatbot-messages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('chatbot-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    generateResponse(message) {
        const lowerMessage = message.toLowerCase();
        
        // Greeting patterns
        if (this.matchesPattern(lowerMessage, ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'])) {
            return this.getRandomResponse('greetings');
        }
        
        // Deepfake related
        if (this.matchesPattern(lowerMessage, ['deepfake', 'fake', 'synthetic', 'generated', 'manipulated', 'detection', 'detect'])) {
            return this.getRandomResponse('deepfake');
        }
        
        // Accuracy questions
        if (this.matchesPattern(lowerMessage, ['accuracy', 'precise', 'reliable', 'performance', 'how good', 'success rate'])) {
            return this.getRandomResponse('accuracy');
        }
        
        // How to use
        if (this.matchesPattern(lowerMessage, ['how to', 'how do', 'use', 'upload', 'analyze', 'steps', 'guide'])) {
            return this.getRandomResponse('howto');
        }
        
        // API questions
        if (this.matchesPattern(lowerMessage, ['api', 'integrate', 'endpoint', 'json', 'request', 'response', 'code'])) {
            return this.getRandomResponse('api');
        }
        
        // Team questions
        if (this.matchesPattern(lowerMessage, ['team', 'developer', 'omkar', 'who made', 'creator', 'author', 'contact'])) {
            return this.getRandomResponse('team');
        }
        
        // Contact information
        if (this.matchesPattern(lowerMessage, ['contact', 'phone', 'email', 'reach', 'social', 'instagram', 'linkedin'])) {
            return this.getRandomResponse('contact');
        }
        
        // Features
        if (this.matchesPattern(lowerMessage, ['features', 'what can', 'capabilities', 'functions', 'options'])) {
            return this.getRandomResponse('features');
        }
        
        // Help
        if (this.matchesPattern(lowerMessage, ['help', 'support', 'assist', 'what', 'questions'])) {
            return this.getRandomResponse('help');
        }
        
        // Default response
        return "I understand you're asking about our deepfake detection system. Could you be more specific? I can help with questions about how it works, accuracy, API usage, team information, or general support. Try using the quick action buttons below for common questions!";
    }

    matchesPattern(message, patterns) {
        return patterns.some(pattern => message.includes(pattern));
    }

    getRandomResponse(category) {
        const responses = this.responses[category];
        return responses[Math.floor(Math.random() * responses.length)];
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    new DeepfakeChatbot();
});
