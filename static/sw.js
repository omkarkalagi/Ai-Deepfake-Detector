
// Advanced Service Worker for Lightning Fast Loading
const CACHE_NAME = 'deepfake-detector-v1.2';
const STATIC_CACHE = 'static-v1.2';
const DYNAMIC_CACHE = 'dynamic-v1.2';

const STATIC_FILES = [
    '/',
    '/static/theme-manager.js',
    '/static/advanced-loader.js',
    '/static/interactive-enhancements.js',
    '/static/chatbot.js',
    '/static/contact-icon-fix.css',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
];

// Install event
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then(cache => cache.addAll(STATIC_FILES))
            .then(() => self.skipWaiting())
    );
});

// Activate event
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys => {
            return Promise.all(keys
                .filter(key => key !== STATIC_CACHE && key !== DYNAMIC_CACHE)
                .map(key => caches.delete(key))
            );
        }).then(() => self.clients.claim())
    );
});

// Fetch event with advanced caching strategy
self.addEventListener('fetch', event => {
    const { request } = event;
    
    // Skip non-GET requests
    if (request.method !== 'GET') return;
    
    // Handle different types of requests
    if (request.url.includes('/static/') || request.url.includes('cdn.')) {
        // Cache first for static resources
        event.respondWith(cacheFirst(request));
    } else if (request.url.includes('/api/')) {
        // Network first for API calls
        event.respondWith(networkFirst(request));
    } else {
        // Stale while revalidate for pages
        event.respondWith(staleWhileRevalidate(request));
    }
});

// Cache strategies
async function cacheFirst(request) {
    const cached = await caches.match(request);
    return cached || fetch(request).then(response => {
        const cache = caches.open(STATIC_CACHE);
        cache.then(c => c.put(request, response.clone()));
        return response;
    });
}

async function networkFirst(request) {
    try {
        const response = await fetch(request);
        const cache = await caches.open(DYNAMIC_CACHE);
        cache.put(request, response.clone());
        return response;
    } catch (error) {
        return caches.match(request);
    }
}

async function staleWhileRevalidate(request) {
    const cached = await caches.match(request);
    const fetchPromise = fetch(request).then(response => {
        const cache = caches.open(DYNAMIC_CACHE);
        cache.then(c => c.put(request, response.clone()));
        return response;
    });
    
    return cached || fetchPromise;
}
