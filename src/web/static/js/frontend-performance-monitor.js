/**
 * Frontend Performance Monitor
 * Advanced error tracking, performance metrics, and ML-ready data collection
 */

class FrontendPerformanceMonitor {
    constructor() {
        this.errors = [];
        this.metrics = {
            pageLoadTime: 0,
            domInteractive: 0,
            resourceTimings: [],
            userInteractions: [],
            memoryUsage: [],
            networkErrors: [],
            jsErrors: []
        };
        this.sessionId = this.generateSessionId();
        this.startTime = performance.now();
        this.errorPatterns = new Map();
        this.init();
    }

    init() {
        this.setupErrorTracking();
        this.setupPerformanceObserver();
        this.setupNetworkMonitoring();
        this.setupMemoryMonitoring();
        this.setupUserInteractionTracking();
        this.setupMLDataCollection();
        this.startReporting();
    }

    generateSessionId() {
        return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    setupErrorTracking() {
        // Global error handler
        window.addEventListener('error', (event) => {
            this.captureError({
                type: 'javascript',
                message: event.message,
                source: event.filename,
                line: event.lineno,
                column: event.colno,
                stack: event.error?.stack,
                timestamp: Date.now(),
                userAgent: navigator.userAgent,
                url: window.location.href,
                sessionId: this.sessionId
            });
        });

        // Promise rejection handler
        window.addEventListener('unhandledrejection', (event) => {
            this.captureError({
                type: 'promise',
                message: event.reason?.message || event.reason,
                stack: event.reason?.stack,
                timestamp: Date.now(),
                url: window.location.href,
                sessionId: this.sessionId
            });
        });

        // Console error interceptor
        const originalError = console.error;
        console.error = (...args) => {
            this.captureError({
                type: 'console',
                message: args.join(' '),
                timestamp: Date.now(),
                url: window.location.href,
                sessionId: this.sessionId
            });
            originalError.apply(console, args);
        };

        // AJAX error tracking
        this.interceptXHR();
        this.interceptFetch();
    }

    interceptXHR() {
        const originalOpen = XMLHttpRequest.prototype.open;
        const originalSend = XMLHttpRequest.prototype.send;
        
        XMLHttpRequest.prototype.open = function(method, url, ...args) {
            this._monitoring = {
                method,
                url,
                startTime: performance.now()
            };
            return originalOpen.apply(this, [method, url, ...args]);
        };

        XMLHttpRequest.prototype.send = function(body) {
            const monitor = window.performanceMonitor;
            
            this.addEventListener('loadend', () => {
                const duration = performance.now() - this._monitoring.startTime;
                
                if (this.status >= 400) {
                    monitor.captureError({
                        type: 'network',
                        subtype: 'xhr',
                        method: this._monitoring.method,
                        url: this._monitoring.url,
                        status: this.status,
                        statusText: this.statusText,
                        duration,
                        timestamp: Date.now()
                    });
                }
                
                monitor.metrics.networkRequests = monitor.metrics.networkRequests || [];
                monitor.metrics.networkRequests.push({
                    method: this._monitoring.method,
                    url: this._monitoring.url,
                    status: this.status,
                    duration,
                    timestamp: Date.now()
                });
            });
            
            return originalSend.apply(this, [body]);
        };
    }

    interceptFetch() {
        const originalFetch = window.fetch;
        
        window.fetch = async function(...args) {
            const startTime = performance.now();
            const [url, options = {}] = args;
            
            try {
                const response = await originalFetch.apply(this, args);
                const duration = performance.now() - startTime;
                
                if (!response.ok) {
                    window.performanceMonitor.captureError({
                        type: 'network',
                        subtype: 'fetch',
                        url: url.toString(),
                        method: options.method || 'GET',
                        status: response.status,
                        statusText: response.statusText,
                        duration,
                        timestamp: Date.now()
                    });
                }
                
                window.performanceMonitor.metrics.networkRequests = window.performanceMonitor.metrics.networkRequests || [];
                window.performanceMonitor.metrics.networkRequests.push({
                    url: url.toString(),
                    method: options.method || 'GET',
                    status: response.status,
                    duration,
                    timestamp: Date.now()
                });
                
                return response;
            } catch (error) {
                window.performanceMonitor.captureError({
                    type: 'network',
                    subtype: 'fetch',
                    url: url.toString(),
                    method: options.method || 'GET',
                    error: error.message,
                    timestamp: Date.now()
                });
                throw error;
            }
        };
    }

    setupPerformanceObserver() {
        // Navigation timing
        if (window.performance && window.performance.timing) {
            const timing = window.performance.timing;
            this.metrics.pageLoadTime = timing.loadEventEnd - timing.navigationStart;
            this.metrics.domInteractive = timing.domInteractive - timing.navigationStart;
            this.metrics.dnsTime = timing.domainLookupEnd - timing.domainLookupStart;
            this.metrics.tcpTime = timing.connectEnd - timing.connectStart;
            this.metrics.ttfb = timing.responseStart - timing.navigationStart;
        }

        // Resource timing
        if ('PerformanceObserver' in window) {
            // Observe resource timings
            const resourceObserver = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    this.metrics.resourceTimings.push({
                        name: entry.name,
                        type: entry.initiatorType,
                        duration: entry.duration,
                        size: entry.transferSize || 0,
                        timestamp: entry.startTime
                    });
                    
                    // Flag slow resources
                    if (entry.duration > 3000) {
                        this.captureError({
                            type: 'performance',
                            subtype: 'slow-resource',
                            resource: entry.name,
                            duration: entry.duration,
                            timestamp: Date.now()
                        });
                    }
                }
            });
            
            resourceObserver.observe({ entryTypes: ['resource'] });

            // Observe long tasks
            if ('PerformanceLongTaskTiming' in window) {
                const longTaskObserver = new PerformanceObserver((list) => {
                    for (const entry of list.getEntries()) {
                        this.captureError({
                            type: 'performance',
                            subtype: 'long-task',
                            duration: entry.duration,
                            timestamp: entry.startTime
                        });
                    }
                });
                
                try {
                    longTaskObserver.observe({ entryTypes: ['longtask'] });
                } catch (e) {
                    console.warn('Long task observation not supported');
                }
            }

            // Observe layout shifts
            if ('LayoutShift' in window) {
                let cls = 0;
                const clsObserver = new PerformanceObserver((list) => {
                    for (const entry of list.getEntries()) {
                        if (!entry.hadRecentInput) {
                            cls += entry.value;
                        }
                    }
                    this.metrics.cumulativeLayoutShift = cls;
                });
                
                try {
                    clsObserver.observe({ type: 'layout-shift', buffered: true });
                } catch (e) {
                    console.warn('Layout shift observation not supported');
                }
            }
        }
    }

    setupNetworkMonitoring() {
        // Monitor connection changes
        if ('connection' in navigator) {
            const connection = navigator.connection;
            
            const updateConnectionInfo = () => {
                this.metrics.connectionInfo = {
                    effectiveType: connection.effectiveType,
                    downlink: connection.downlink,
                    rtt: connection.rtt,
                    saveData: connection.saveData,
                    timestamp: Date.now()
                };
            };
            
            updateConnectionInfo();
            connection.addEventListener('change', updateConnectionInfo);
        }

        // Monitor online/offline events
        window.addEventListener('online', () => {
            this.captureError({
                type: 'network',
                subtype: 'connection',
                status: 'online',
                timestamp: Date.now()
            });
        });

        window.addEventListener('offline', () => {
            this.captureError({
                type: 'network',
                subtype: 'connection',
                status: 'offline',
                timestamp: Date.now()
            });
        });
    }

    setupMemoryMonitoring() {
        if (performance.memory) {
            setInterval(() => {
                const memory = performance.memory;
                const usage = {
                    usedJSHeapSize: memory.usedJSHeapSize,
                    totalJSHeapSize: memory.totalJSHeapSize,
                    jsHeapSizeLimit: memory.jsHeapSizeLimit,
                    percentUsed: (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100,
                    timestamp: Date.now()
                };
                
                this.metrics.memoryUsage.push(usage);
                
                // Alert on high memory usage
                if (usage.percentUsed > 90) {
                    this.captureError({
                        type: 'performance',
                        subtype: 'high-memory',
                        usage: usage.percentUsed,
                        timestamp: Date.now()
                    });
                }
                
                // Keep only last 100 measurements
                if (this.metrics.memoryUsage.length > 100) {
                    this.metrics.memoryUsage.shift();
                }
            }, 10000); // Every 10 seconds
        }
    }

    setupUserInteractionTracking() {
        // Track clicks
        document.addEventListener('click', (event) => {
            this.metrics.userInteractions.push({
                type: 'click',
                target: event.target.tagName,
                id: event.target.id,
                className: event.target.className,
                timestamp: Date.now(),
                x: event.clientX,
                y: event.clientY
            });
        });

        // Track form submissions
        document.addEventListener('submit', (event) => {
            this.metrics.userInteractions.push({
                type: 'form-submit',
                formId: event.target.id,
                formName: event.target.name,
                timestamp: Date.now()
            });
        });

        // Track page visibility changes
        document.addEventListener('visibilitychange', () => {
            this.metrics.userInteractions.push({
                type: 'visibility-change',
                hidden: document.hidden,
                timestamp: Date.now()
            });
        });

        // Track rage clicks (rapid repeated clicks)
        let clickCount = 0;
        let clickTimer;
        
        document.addEventListener('click', (event) => {
            clickCount++;
            
            if (clickCount >= 3) {
                this.captureError({
                    type: 'user-frustration',
                    subtype: 'rage-click',
                    target: event.target.tagName,
                    count: clickCount,
                    timestamp: Date.now()
                });
            }
            
            clearTimeout(clickTimer);
            clickTimer = setTimeout(() => {
                clickCount = 0;
            }, 1000);
        });
    }

    setupMLDataCollection() {
        // Collect patterns for ML analysis
        this.errorPatternAnalyzer = setInterval(() => {
            this.analyzeErrorPatterns();
        }, 30000); // Every 30 seconds
    }

    analyzeErrorPatterns() {
        // Group errors by type and message
        const patterns = {};
        
        this.errors.forEach(error => {
            const key = `${error.type}_${error.subtype || 'general'}`;
            
            if (!patterns[key]) {
                patterns[key] = {
                    count: 0,
                    firstOccurrence: error.timestamp,
                    lastOccurrence: error.timestamp,
                    examples: [],
                    contexts: []
                };
            }
            
            patterns[key].count++;
            patterns[key].lastOccurrence = error.timestamp;
            
            if (patterns[key].examples.length < 5) {
                patterns[key].examples.push(error);
            }
            
            // Collect context information
            if (error.url && !patterns[key].contexts.includes(error.url)) {
                patterns[key].contexts.push(error.url);
            }
        });
        
        // Store patterns for ML processing
        this.errorPatterns = patterns;
        
        // Detect anomalies
        Object.entries(patterns).forEach(([key, pattern]) => {
            // Spike detection
            if (pattern.count > 10 && (pattern.lastOccurrence - pattern.firstOccurrence) < 60000) {
                this.captureError({
                    type: 'anomaly',
                    subtype: 'error-spike',
                    pattern: key,
                    count: pattern.count,
                    duration: pattern.lastOccurrence - pattern.firstOccurrence,
                    timestamp: Date.now()
                });
            }
        });
    }

    captureError(error) {
        // Enrich error with context
        error = {
            ...error,
            sessionId: this.sessionId,
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight
            },
            screen: {
                width: window.screen.width,
                height: window.screen.height
            },
            devicePixelRatio: window.devicePixelRatio,
            language: navigator.language,
            platform: navigator.platform
        };
        
        this.errors.push(error);
        
        // Keep only last 1000 errors
        if (this.errors.length > 1000) {
            this.errors.shift();
        }
        
        // Send critical errors immediately
        if (error.type === 'javascript' || error.type === 'network') {
            this.sendToBackend([error]);
        }
    }

    async sendToBackend(errors = null) {
        const payload = {
            sessionId: this.sessionId,
            timestamp: Date.now(),
            errors: errors || this.errors.slice(-100), // Send last 100 errors
            metrics: this.getMetricsSummary(),
            patterns: Object.fromEntries(this.errorPatterns),
            userAgent: navigator.userAgent,
            url: window.location.href
        };
        
        try {
            const response = await fetch('/api/v1/monitoring/frontend-errors', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            if (response.ok) {
                const result = await response.json();
                
                // Apply any suggested fixes from ML
                if (result.suggestions) {
                    this.applySuggestions(result.suggestions);
                }
            }
        } catch (error) {
            console.warn('Failed to send monitoring data:', error);
        }
    }

    getMetricsSummary() {
        return {
            pageLoadTime: this.metrics.pageLoadTime,
            domInteractive: this.metrics.domInteractive,
            resourceCount: this.metrics.resourceTimings.length,
            slowResources: this.metrics.resourceTimings.filter(r => r.duration > 3000).length,
            memoryUsage: this.metrics.memoryUsage.slice(-1)[0],
            errorCount: this.errors.length,
            interactionCount: this.metrics.userInteractions.length,
            sessionDuration: performance.now() - this.startTime
        };
    }

    applySuggestions(suggestions) {
        suggestions.forEach(suggestion => {
            console.log('ML Suggestion:', suggestion);
            
            // Auto-apply safe fixes
            if (suggestion.autoApply && suggestion.type === 'performance') {
                switch (suggestion.action) {
                    case 'lazy-load-images':
                        this.enableLazyLoading();
                        break;
                    case 'reduce-animations':
                        this.reduceAnimations();
                        break;
                    case 'cache-resources':
                        this.enableResourceCaching();
                        break;
                }
            }
        });
    }

    enableLazyLoading() {
        document.querySelectorAll('img:not([loading])').forEach(img => {
            img.loading = 'lazy';
        });
    }

    reduceAnimations() {
        const style = document.createElement('style');
        style.textContent = `
            @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after {
                    animation-duration: 0.01ms !important;
                    animation-iteration-count: 1 !important;
                    transition-duration: 0.01ms !important;
                }
            }
        `;
        document.head.appendChild(style);
    }

    enableResourceCaching() {
        // Send cache headers recommendation to backend
        this.sendToBackend([{
            type: 'recommendation',
            subtype: 'enable-caching',
            timestamp: Date.now()
        }]);
    }

    startReporting() {
        // Send reports every 60 seconds
        setInterval(() => {
            this.sendToBackend();
        }, 60000);
        
        // Send report before page unload
        window.addEventListener('beforeunload', () => {
            this.sendToBackend();
        });
    }

    // Public API for manual error reporting
    reportError(message, context = {}) {
        this.captureError({
            type: 'manual',
            message,
            context,
            timestamp: Date.now()
        });
    }
}

// Initialize monitor
document.addEventListener('DOMContentLoaded', () => {
    window.performanceMonitor = new FrontendPerformanceMonitor();
    console.log('Frontend Performance Monitor initialized');
});