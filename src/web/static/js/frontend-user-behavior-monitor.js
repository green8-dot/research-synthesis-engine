/**
 * Frontend User Behavior Monitor
 * Advanced user behavior analytics with ML-powered insights and predictions
 */

class FrontendUserBehaviorMonitor {
    constructor() {
        this.userId = this.getUserId();
        this.sessionData = {
            sessionId: this.generateSessionId(),
            startTime: Date.now(),
            pageViews: [],
            interactions: [],
            mouseMovements: [],
            scrollEvents: [],
            focusEvents: [],
            formInteractions: [],
            searchQueries: [],
            featureUsage: new Map(),
            userJourney: [],
            heatmapData: [],
            attentionMap: new Map()
        };
        this.behaviorPatterns = {
            clickPatterns: [],
            navigationPatterns: [],
            engagementScore: 0,
            frustrationIndicators: [],
            conversionEvents: []
        };
        this.mlFeatures = {
            sessionFeatures: {},
            userFeatures: {},
            contextFeatures: {}
        };
        this.init();
    }

    init() {
        this.setupPageTracking();
        this.setupInteractionTracking();
        this.setupMouseTracking();
        this.setupScrollTracking();
        this.setupFormTracking();
        this.setupSearchTracking();
        this.setupFeatureUsageTracking();
        this.setupEngagementTracking();
        this.setupHeatmapCollection();
        this.setupMLFeatureExtraction();
        this.startAnalytics();
    }

    getUserId() {
        let userId = localStorage.getItem('user_behavior_id');
        if (!userId) {
            userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            localStorage.setItem('user_behavior_id', userId);
        }
        return userId;
    }

    generateSessionId() {
        return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    setupPageTracking() {
        // Track initial page load
        this.trackPageView();

        // Track navigation changes
        const originalPushState = history.pushState;
        const originalReplaceState = history.replaceState;
        
        history.pushState = function(...args) {
            originalPushState.apply(history, args);
            window.behaviorMonitor.trackPageView();
        };
        
        history.replaceState = function(...args) {
            originalReplaceState.apply(history, args);
            window.behaviorMonitor.trackPageView();
        };
        
        window.addEventListener('popstate', () => {
            this.trackPageView();
        });
    }

    trackPageView() {
        const pageView = {
            url: window.location.href,
            title: document.title,
            referrer: document.referrer,
            timestamp: Date.now(),
            loadTime: performance.now(),
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight
            }
        };
        
        this.sessionData.pageViews.push(pageView);
        this.sessionData.userJourney.push({
            type: 'page_view',
            data: pageView
        });
        
        this.analyzeUserJourney();
    }

    setupInteractionTracking() {
        // Click tracking with detailed context
        document.addEventListener('click', (event) => {
            const target = event.target;
            const interaction = {
                type: 'click',
                timestamp: Date.now(),
                element: {
                    tagName: target.tagName,
                    id: target.id,
                    className: target.className,
                    text: target.textContent?.substring(0, 50),
                    href: target.href,
                    dataAttributes: this.getDataAttributes(target)
                },
                position: {
                    x: event.clientX,
                    y: event.clientY,
                    pageX: event.pageX,
                    pageY: event.pageY
                },
                context: {
                    ctrlKey: event.ctrlKey,
                    shiftKey: event.shiftKey,
                    altKey: event.altKey,
                    button: event.button
                }
            };
            
            this.sessionData.interactions.push(interaction);
            this.analyzeClickPattern(interaction);
            this.updateHeatmap(interaction.position);
        });

        // Double click tracking
        document.addEventListener('dblclick', (event) => {
            this.sessionData.interactions.push({
                type: 'double_click',
                timestamp: Date.now(),
                target: event.target.tagName,
                position: { x: event.clientX, y: event.clientY }
            });
        });

        // Context menu tracking
        document.addEventListener('contextmenu', (event) => {
            this.sessionData.interactions.push({
                type: 'context_menu',
                timestamp: Date.now(),
                target: event.target.tagName,
                position: { x: event.clientX, y: event.clientY }
            });
        });

        // Key press tracking (privacy-conscious)
        document.addEventListener('keydown', (event) => {
            // Only track special keys, not actual text input
            if (event.key in ['Enter', 'Escape', 'Tab', 'Delete', 'Backspace'] || 
                event.ctrlKey || event.altKey || event.metaKey) {
                this.sessionData.interactions.push({
                    type: 'key_press',
                    key: event.key,
                    timestamp: Date.now(),
                    modifiers: {
                        ctrl: event.ctrlKey,
                        alt: event.altKey,
                        shift: event.shiftKey,
                        meta: event.metaKey
                    }
                });
            }
        });
    }

    setupMouseTracking() {
        let mouseTracker = {
            movements: [],
            lastPosition: null,
            lastTime: Date.now()
        };

        // Sample mouse movements (not every pixel to avoid performance issues)
        document.addEventListener('mousemove', (event) => {
            const now = Date.now();
            
            // Sample every 100ms
            if (now - mouseTracker.lastTime > 100) {
                const movement = {
                    x: event.clientX,
                    y: event.clientY,
                    timestamp: now,
                    velocity: mouseTracker.lastPosition ? 
                        this.calculateVelocity(mouseTracker.lastPosition, { x: event.clientX, y: event.clientY }, now - mouseTracker.lastTime) : 0
                };
                
                mouseTracker.movements.push(movement);
                mouseTracker.lastPosition = { x: event.clientX, y: event.clientY };
                mouseTracker.lastTime = now;
                
                // Keep only last 100 movements
                if (mouseTracker.movements.length > 100) {
                    // Analyze patterns before removing
                    this.analyzeMousePatterns(mouseTracker.movements.slice(0, 50));
                    mouseTracker.movements = mouseTracker.movements.slice(-50);
                }
            }
        });

        // Mouse hover tracking
        let hoverTimeout;
        document.addEventListener('mouseover', (event) => {
            clearTimeout(hoverTimeout);
            hoverTimeout = setTimeout(() => {
                this.sessionData.interactions.push({
                    type: 'hover',
                    target: event.target.tagName,
                    id: event.target.id,
                    duration: 500, // Hovered for at least 500ms
                    timestamp: Date.now()
                });
            }, 500);
        });

        document.addEventListener('mouseout', () => {
            clearTimeout(hoverTimeout);
        });

        this.sessionData.mouseMovements = mouseTracker.movements;
    }

    setupScrollTracking() {
        let scrollData = {
            maxDepth: 0,
            totalDistance: 0,
            scrollEvents: [],
            lastPosition: 0,
            lastTime: Date.now()
        };

        const trackScroll = () => {
            const now = Date.now();
            const currentPosition = window.pageYOffset;
            const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPercentage = (currentPosition / maxScroll) * 100;
            
            scrollData.maxDepth = Math.max(scrollData.maxDepth, scrollPercentage);
            scrollData.totalDistance += Math.abs(currentPosition - scrollData.lastPosition);
            
            const scrollEvent = {
                position: currentPosition,
                percentage: scrollPercentage,
                direction: currentPosition > scrollData.lastPosition ? 'down' : 'up',
                velocity: Math.abs(currentPosition - scrollData.lastPosition) / (now - scrollData.lastTime),
                timestamp: now
            };
            
            scrollData.scrollEvents.push(scrollEvent);
            scrollData.lastPosition = currentPosition;
            scrollData.lastTime = now;
            
            // Analyze scroll patterns
            this.analyzeScrollBehavior(scrollEvent);
            
            // Keep only last 50 events
            if (scrollData.scrollEvents.length > 50) {
                scrollData.scrollEvents.shift();
            }
        };

        // Throttled scroll tracking
        let scrollTimeout;
        window.addEventListener('scroll', () => {
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(trackScroll, 100);
        });

        this.sessionData.scrollEvents = scrollData.scrollEvents;
    }

    setupFormTracking() {
        // Track form focus
        document.addEventListener('focusin', (event) => {
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA' || event.target.tagName === 'SELECT') {
                this.sessionData.formInteractions.push({
                    type: 'focus',
                    field: {
                        name: event.target.name,
                        type: event.target.type,
                        id: event.target.id
                    },
                    timestamp: Date.now()
                });
            }
        });

        // Track form blur
        document.addEventListener('focusout', (event) => {
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA' || event.target.tagName === 'SELECT') {
                this.sessionData.formInteractions.push({
                    type: 'blur',
                    field: {
                        name: event.target.name,
                        type: event.target.type,
                        id: event.target.id,
                        filled: event.target.value !== ''
                    },
                    timestamp: Date.now()
                });
            }
        });

        // Track form changes
        document.addEventListener('change', (event) => {
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA' || event.target.tagName === 'SELECT') {
                this.sessionData.formInteractions.push({
                    type: 'change',
                    field: {
                        name: event.target.name,
                        type: event.target.type,
                        id: event.target.id
                    },
                    timestamp: Date.now()
                });
            }
        });

        // Track form submissions
        document.addEventListener('submit', (event) => {
            const formData = {
                type: 'submit',
                form: {
                    id: event.target.id,
                    name: event.target.name,
                    action: event.target.action,
                    method: event.target.method
                },
                fields: Array.from(event.target.elements).map(el => ({
                    name: el.name,
                    type: el.type,
                    filled: el.value !== ''
                })),
                timestamp: Date.now()
            };
            
            this.sessionData.formInteractions.push(formData);
            this.trackConversion('form_submission', formData);
        });
    }

    setupSearchTracking() {
        // Track search inputs
        document.addEventListener('input', (event) => {
            if (event.target.type === 'search' || 
                event.target.getAttribute('role') === 'searchbox' ||
                event.target.className?.includes('search')) {
                
                // Debounce search tracking
                clearTimeout(this.searchTimeout);
                this.searchTimeout = setTimeout(() => {
                    this.sessionData.searchQueries.push({
                        query: event.target.value,
                        field: event.target.id || event.target.name,
                        timestamp: Date.now()
                    });
                }, 500);
            }
        });
    }

    setupFeatureUsageTracking() {
        // Track button clicks by feature
        document.addEventListener('click', (event) => {
            const target = event.target.closest('button, a, [role="button"]');
            if (target) {
                const feature = this.identifyFeature(target);
                if (feature) {
                    const count = this.sessionData.featureUsage.get(feature) || 0;
                    this.sessionData.featureUsage.set(feature, count + 1);
                }
            }
        });
    }

    setupEngagementTracking() {
        // Track time on page
        let pageStartTime = Date.now();
        let totalEngagedTime = 0;
        let isEngaged = true;

        // Track active/idle state
        let idleTimeout;
        const resetIdleTimer = () => {
            if (!isEngaged) {
                isEngaged = true;
                pageStartTime = Date.now();
            }
            clearTimeout(idleTimeout);
            idleTimeout = setTimeout(() => {
                if (isEngaged) {
                    totalEngagedTime += Date.now() - pageStartTime;
                    isEngaged = false;
                }
            }, 30000); // 30 seconds of inactivity
        };

        // Events that indicate engagement
        ['mousemove', 'keypress', 'scroll', 'click', 'touchstart'].forEach(event => {
            document.addEventListener(event, resetIdleTimer);
        });

        // Track tab visibility
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                if (isEngaged) {
                    totalEngagedTime += Date.now() - pageStartTime;
                    isEngaged = false;
                }
            } else {
                pageStartTime = Date.now();
                isEngaged = true;
            }
        });

        // Calculate engagement score periodically
        setInterval(() => {
            const currentEngagedTime = isEngaged ? 
                totalEngagedTime + (Date.now() - pageStartTime) : totalEngagedTime;
            
            this.behaviorPatterns.engagementScore = this.calculateEngagementScore({
                timeEngaged: currentEngagedTime,
                interactions: this.sessionData.interactions.length,
                scrollDepth: this.sessionData.scrollEvents.length > 0 ? 
                    Math.max(...this.sessionData.scrollEvents.map(e => e.percentage)) : 0,
                featureUsage: this.sessionData.featureUsage.size
            });
        }, 10000);
    }

    setupHeatmapCollection() {
        // Divide viewport into grid cells
        const gridSize = 50; // 50x50 pixel cells
        
        const updateHeatmapGrid = (x, y) => {
            const cellX = Math.floor(x / gridSize);
            const cellY = Math.floor(y / gridSize);
            const key = `${cellX}_${cellY}`;
            
            const current = this.sessionData.attentionMap.get(key) || 0;
            this.sessionData.attentionMap.set(key, current + 1);
        };

        // Update heatmap on clicks
        document.addEventListener('click', (event) => {
            updateHeatmapGrid(event.clientX, event.clientY);
        });

        // Update heatmap on hovers
        document.addEventListener('mousemove', (event) => {
            // Sample periodically
            if (Math.random() < 0.1) { // 10% sampling rate
                updateHeatmapGrid(event.clientX, event.clientY);
            }
        });
    }

    setupMLFeatureExtraction() {
        // Extract features for ML model
        setInterval(() => {
            this.mlFeatures.sessionFeatures = {
                duration: Date.now() - this.sessionData.startTime,
                pageViews: this.sessionData.pageViews.length,
                interactionCount: this.sessionData.interactions.length,
                clickCount: this.sessionData.interactions.filter(i => i.type === 'click').length,
                scrollEvents: this.sessionData.scrollEvents.length,
                formInteractions: this.sessionData.formInteractions.length,
                searchQueries: this.sessionData.searchQueries.length,
                featureUsageCount: this.sessionData.featureUsage.size,
                engagementScore: this.behaviorPatterns.engagementScore
            };

            this.mlFeatures.userFeatures = {
                returnVisitor: this.isReturnVisitor(),
                deviceType: this.getDeviceType(),
                browserType: this.getBrowserType(),
                timeOfDay: new Date().getHours(),
                dayOfWeek: new Date().getDay()
            };

            this.mlFeatures.contextFeatures = {
                currentPage: window.location.pathname,
                referrer: document.referrer,
                screenResolution: `${window.screen.width}x${window.screen.height}`,
                connectionType: navigator.connection?.effectiveType || 'unknown'
            };

            // Send features for ML processing
            this.sendMLFeatures();
        }, 30000); // Every 30 seconds
    }

    // Analysis Methods
    analyzeClickPattern(interaction) {
        const patterns = this.behaviorPatterns.clickPatterns;
        
        // Detect rapid clicks (frustration indicator)
        const recentClicks = patterns.filter(p => 
            Date.now() - p.timestamp < 2000
        );
        
        if (recentClicks.length >= 3) {
            this.behaviorPatterns.frustrationIndicators.push({
                type: 'rapid_clicks',
                count: recentClicks.length,
                timestamp: Date.now()
            });
        }

        // Detect click dead zones (UI issues)
        const samePositionClicks = patterns.filter(p =>
            Math.abs(p.position.x - interaction.position.x) < 10 &&
            Math.abs(p.position.y - interaction.position.y) < 10 &&
            Date.now() - p.timestamp < 5000
        );

        if (samePositionClicks.length >= 2) {
            this.behaviorPatterns.frustrationIndicators.push({
                type: 'repeated_same_position_clicks',
                position: interaction.position,
                timestamp: Date.now()
            });
        }

        patterns.push(interaction);
        
        // Keep only last 100 patterns
        if (patterns.length > 100) {
            patterns.shift();
        }
    }

    analyzeMousePatterns(movements) {
        // Calculate average velocity
        const avgVelocity = movements.reduce((sum, m) => sum + m.velocity, 0) / movements.length;
        
        // Detect erratic movements (confusion indicator)
        const velocityVariance = this.calculateVariance(movements.map(m => m.velocity));
        
        if (velocityVariance > 1000) {
            this.behaviorPatterns.frustrationIndicators.push({
                type: 'erratic_mouse_movement',
                variance: velocityVariance,
                timestamp: Date.now()
            });
        }
    }

    analyzeScrollBehavior(scrollEvent) {
        // Detect fast scrolling (scanning behavior)
        if (scrollEvent.velocity > 100) {
            this.sessionData.userJourney.push({
                type: 'fast_scroll',
                velocity: scrollEvent.velocity,
                timestamp: scrollEvent.timestamp
            });
        }

        // Detect scroll bouncing (looking for something)
        const recentScrolls = this.sessionData.scrollEvents.slice(-5);
        const directions = recentScrolls.map(s => s.direction);
        const bounces = directions.filter((d, i) => i > 0 && d !== directions[i - 1]).length;
        
        if (bounces >= 3) {
            this.behaviorPatterns.frustrationIndicators.push({
                type: 'scroll_bouncing',
                bounces,
                timestamp: Date.now()
            });
        }
    }

    analyzeUserJourney() {
        const journey = this.sessionData.userJourney;
        
        // Detect navigation patterns
        if (journey.length >= 2) {
            const lastTwo = journey.slice(-2);
            
            // Back navigation detection
            if (lastTwo[0].type === 'page_view' && lastTwo[1].type === 'page_view' &&
                lastTwo[0].data.url === lastTwo[1].data.referrer) {
                this.behaviorPatterns.navigationPatterns.push({
                    type: 'back_navigation',
                    from: lastTwo[0].data.url,
                    to: lastTwo[1].data.url,
                    timestamp: Date.now()
                });
            }
        }

        // Detect looping behavior
        const pageViews = journey.filter(j => j.type === 'page_view');
        const urlCounts = {};
        pageViews.forEach(pv => {
            urlCounts[pv.data.url] = (urlCounts[pv.data.url] || 0) + 1;
        });
        
        Object.entries(urlCounts).forEach(([url, count]) => {
            if (count >= 3) {
                this.behaviorPatterns.frustrationIndicators.push({
                    type: 'page_loop',
                    url,
                    visitCount: count,
                    timestamp: Date.now()
                });
            }
        });
    }

    // Helper Methods
    getDataAttributes(element) {
        const attrs = {};
        Array.from(element.attributes).forEach(attr => {
            if (attr.name.startsWith('data-')) {
                attrs[attr.name] = attr.value;
            }
        });
        return attrs;
    }

    calculateVelocity(pos1, pos2, timeDelta) {
        const distance = Math.sqrt(
            Math.pow(pos2.x - pos1.x, 2) + 
            Math.pow(pos2.y - pos1.y, 2)
        );
        return distance / timeDelta;
    }

    calculateVariance(values) {
        const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
        return values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
    }

    calculateEngagementScore(metrics) {
        // Weighted engagement score
        const weights = {
            timeEngaged: 0.3,
            interactions: 0.25,
            scrollDepth: 0.2,
            featureUsage: 0.25
        };
        
        // Normalize metrics
        const normalized = {
            timeEngaged: Math.min(metrics.timeEngaged / 300000, 1), // 5 minutes max
            interactions: Math.min(metrics.interactions / 50, 1),
            scrollDepth: metrics.scrollDepth / 100,
            featureUsage: Math.min(metrics.featureUsage / 10, 1)
        };
        
        return Object.keys(weights).reduce((score, key) => 
            score + (normalized[key] * weights[key] * 100), 0
        );
    }

    identifyFeature(element) {
        // Try to identify feature from element attributes
        return element.dataset.feature || 
               element.getAttribute('aria-label') || 
               element.id || 
               element.textContent?.substring(0, 30);
    }

    isReturnVisitor() {
        const visits = parseInt(localStorage.getItem('visit_count') || '0');
        localStorage.setItem('visit_count', (visits + 1).toString());
        return visits > 0;
    }

    getDeviceType() {
        const width = window.innerWidth;
        if (width < 768) return 'mobile';
        if (width < 1024) return 'tablet';
        return 'desktop';
    }

    getBrowserType() {
        const ua = navigator.userAgent;
        if (ua.includes('Chrome')) return 'chrome';
        if (ua.includes('Firefox')) return 'firefox';
        if (ua.includes('Safari')) return 'safari';
        if (ua.includes('Edge')) return 'edge';
        return 'other';
    }

    trackConversion(type, data) {
        this.behaviorPatterns.conversionEvents.push({
            type,
            data,
            timestamp: Date.now()
        });
    }

    updateHeatmap(position) {
        this.sessionData.heatmapData.push({
            x: position.x,
            y: position.y,
            timestamp: Date.now()
        });
        
        // Keep only last 1000 points
        if (this.sessionData.heatmapData.length > 1000) {
            this.sessionData.heatmapData.shift();
        }
    }

    async sendMLFeatures() {
        const payload = {
            userId: this.userId,
            sessionId: this.sessionData.sessionId,
            features: this.mlFeatures,
            patterns: this.behaviorPatterns,
            timestamp: Date.now()
        };
        
        try {
            const response = await fetch('/api/v1/monitoring/user-behavior', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            if (response.ok) {
                const result = await response.json();
                
                // Apply ML predictions
                if (result.predictions) {
                    this.applyPredictions(result.predictions);
                }
            }
        } catch (error) {
            console.warn('Failed to send behavior data:', error);
        }
    }

    applyPredictions(predictions) {
        // Apply personalization based on predictions
        if (predictions.nextLikelyAction) {
            console.log('Predicted next action:', predictions.nextLikelyAction);
            // Could pre-load resources or highlight relevant UI elements
        }
        
        if (predictions.churnRisk > 0.7) {
            // User might leave - trigger retention mechanism
            this.triggerRetention();
        }
        
        if (predictions.conversionProbability > 0.8) {
            // High conversion probability - streamline process
            this.optimizeForConversion();
        }
    }

    triggerRetention() {
        // Implement retention strategies
        console.log('Triggering retention mechanism');
    }

    optimizeForConversion() {
        // Implement conversion optimization
        console.log('Optimizing for conversion');
    }

    startAnalytics() {
        // Send analytics every 30 seconds
        setInterval(() => {
            this.sendMLFeatures();
        }, 30000);
        
        // Send before page unload
        window.addEventListener('beforeunload', () => {
            this.sendMLFeatures();
        });
    }
}

// Initialize monitor
document.addEventListener('DOMContentLoaded', () => {
    window.behaviorMonitor = new FrontendUserBehaviorMonitor();
    console.log('User Behavior Monitor initialized');
});