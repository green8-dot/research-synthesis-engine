/**
 * Research Synthesis Engine - Main JavaScript
 * Modern, responsive UI interactions
 */

class ResearchSynthesisApp {
    constructor() {
        this.api = axios.create({
            baseURL: '/api/v1',
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        this.initializeEventListeners();
        this.updateSystemTime();
        this.checkSystemStatus();
        
        // Update time every second
        setInterval(() => this.updateSystemTime(), 1000);
        
        // Check system status every 60 seconds (optimized)
        setInterval(() => this.checkSystemStatus(), 60000);
    }

    initializeEventListeners() {
        document.addEventListener('DOMContentLoaded', () => {
            this.initializeNavigation();
            this.initializeTooltips();
            this.initializeModals();
            this.initializeForms();
        });
    }

    initializeNavigation() {
        // Highlight active navigation item
        const currentPath = window.location.pathname;
        document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
            const href = link.getAttribute('href');
            if (href && currentPath.startsWith(href) && href !== '/') {
                link.classList.add('active');
            }
        });
    }

    initializeTooltips() {
        // Initialize Bootstrap tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(tooltipTriggerEl => {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    initializeModals() {
        // Modal event handlers
        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('shown.bs.modal', (e) => {
                const firstInput = modal.querySelector('input, textarea, select');
                if (firstInput) {
                    firstInput.focus();
                }
            });
        });
    }

    initializeForms() {
        // Form validation and submission
        document.querySelectorAll('form[data-submit-api]').forEach(form => {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleFormSubmission(form);
            });
        });
    }

    updateSystemTime() {
        const timeElement = document.getElementById('system-time');
        if (timeElement) {
            const now = new Date();
            timeElement.textContent = `Time: ${now.toLocaleTimeString()}`;
        }
    }

    async checkSystemStatus() {
        try {
            const response = await this.api.get('/health');
            this.updateSystemStatusIndicators(true);
        } catch (error) {
            this.updateSystemStatusIndicators(false);
        }
    }

    updateSystemStatusIndicators(isOnline) {
        document.querySelectorAll('.status-indicator').forEach(indicator => {
            const dot = indicator.querySelector('.status-dot');
            const text = indicator.querySelector('.status-text');
            
            if (dot) {
                dot.className = `status-dot ${isOnline ? 'online' : 'offline'}`;
            }
            
            if (text) {
                text.textContent = isOnline ? 'Online' : 'Offline';
            }
        });
    }

    showFlashMessage(message, type = 'info', duration = 5000) {
        // For error/danger messages, use the prominent fixed notification system
        if (type === 'danger' || type === 'error') {
            this.showProminentError(message, duration);
            return;
        }
        
        const flashContainer = document.getElementById('flash-messages');
        if (!flashContainer) return;

        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        flashContainer.appendChild(alert);

        // Auto-dismiss after duration
        setTimeout(() => {
            if (alert.parentNode) {
                bootstrap.Alert.getOrCreateInstance(alert).close();
            }
        }, duration);
    }

    showProminentError(message, duration = 8000) {
        // Remove any existing prominent error
        const existing = document.getElementById('prominent-error-notification');
        if (existing) existing.remove();

        // Create prominent fixed error notification
        const errorNotification = document.createElement('div');
        errorNotification.id = 'prominent-error-notification';
        errorNotification.className = 'prominent-error-notification show';
        errorNotification.innerHTML = `
            <div class="error-content">
                <div class="error-header">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <strong>Action Failed</strong>
                    <button type="button" class="btn-close btn-close-white ms-auto" onclick="this.parentElement.parentElement.parentElement.remove()"></button>
                </div>
                <div class="error-message">${message}</div>
                <div class="error-progress">
                    <div class="progress-bar"></div>
                </div>
            </div>
        `;

        // Add to body for maximum visibility
        document.body.appendChild(errorNotification);

        // Animate progress bar and auto-dismiss
        const progressBar = errorNotification.querySelector('.progress-bar');
        if (progressBar && duration > 0) {
            progressBar.style.transition = `width ${duration}ms linear`;
            progressBar.style.width = '0%';
            
            // Start progress animation
            setTimeout(() => {
                progressBar.style.width = '100%';
            }, 100);

            // Auto-dismiss after duration
            setTimeout(() => {
                if (errorNotification.parentNode) {
                    errorNotification.classList.remove('show');
                    setTimeout(() => errorNotification.remove(), 300);
                }
            }, duration);
        }

        // Add click to dismiss
        errorNotification.addEventListener('click', (e) => {
            if (e.target === errorNotification) {
                errorNotification.classList.remove('show');
                setTimeout(() => errorNotification.remove(), 300);
            }
        });
    }

    showLoadingOverlay(show = true, title = 'Processing...', message = 'Please wait while we complete your request', progress = 0) {
        const overlay = document.getElementById('global-loading-overlay');
        const titleElement = document.getElementById('loading-title');
        const messageElement = document.getElementById('loading-message');
        const progressElement = document.getElementById('loading-progress');
        
        if (overlay) {
            if (show) {
                if (titleElement) titleElement.textContent = title;
                if (messageElement) messageElement.textContent = message;
                if (progressElement && progress > 0) {
                    progressElement.style.width = progress + '%';
                    progressElement.setAttribute('aria-valuenow', progress);
                }
                overlay.style.display = 'flex';
            } else {
                overlay.style.display = 'none';
                // Reset progress when hiding
                if (progressElement) {
                    progressElement.style.width = '0%';
                    progressElement.setAttribute('aria-valuenow', 0);
                }
            }
        }
    }

    updateLoadingProgress(progress, title, message) {
        const overlay = document.getElementById('global-loading-overlay');
        if (overlay && overlay.style.display === 'flex') {
            const titleElement = document.getElementById('loading-title');
            const messageElement = document.getElementById('loading-message');
            const progressElement = document.getElementById('loading-progress');
            
            if (titleElement && title) titleElement.textContent = title;
            if (messageElement && message) messageElement.textContent = message;
            if (progressElement && progress !== undefined) {
                progressElement.style.width = progress + '%';
                progressElement.setAttribute('aria-valuenow', progress);
            }
        }
    }

    showGlobalLoading(options = {}) {
        const {
            title = 'Processing...',
            message = 'Please wait while we complete your request',
            progress = 0,
            timeout = 0
        } = options;
        
        this.showLoadingOverlay(true, title, message, progress);
        
        // Auto-hide after timeout if specified
        if (timeout > 0) {
            setTimeout(() => {
                this.showLoadingOverlay(false);
            }, timeout);
        }
    }

    hideGlobalLoading() {
        this.showLoadingOverlay(false);
    }

    async handleFormSubmission(form) {
        const apiEndpoint = form.dataset.submitApi;
        const method = form.dataset.method || 'POST';
        const formData = new FormData(form);
        const data = Object.fromEntries(formData);

        // Handle checkboxes separately since they don't appear in FormData if unchecked
        const checkboxes = form.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            data[checkbox.name] = checkbox.checked;
        });

        console.log('Form submission:', { apiEndpoint, method, data });
        this.showGlobalLoading({
            title: 'Submitting...',
            message: 'Processing your request'
        });

        try {
            const response = await this.api.request({
                method: method,
                url: apiEndpoint,
                data: data
            });

            console.log('API response:', response);
            this.showFlashMessage('Operation completed successfully!', 'success');
            
            // Trigger custom event
            form.dispatchEvent(new CustomEvent('formSubmitSuccess', {
                detail: { response: response.data }
            }));

        } catch (error) {
            console.error('Form submission error:', error);
            console.error('Error details:', {
                message: error.message,
                response: error.response,
                status: error.response?.status,
                data: error.response?.data
            });
            
            this.showFlashMessage(
                error.response?.data?.message || error.message || 'An error occurred. Please try again.',
                'danger'
            );

            form.dispatchEvent(new CustomEvent('formSubmitError', {
                detail: { error: error }
            }));
        } finally {
            this.hideGlobalLoading();
        }
    }

    // API Helper Methods
    async fetchData(endpoint, params = {}) {
        try {
            const response = await this.api.get(endpoint, { params });
            return response.data;
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            throw error;
        }
    }

    async postData(endpoint, data = {}) {
        try {
            const response = await this.api.post(endpoint, data);
            return response.data;
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            throw error;
        }
    }

    // Utility Methods
    formatNumber(num) {
        return new Intl.NumberFormat().format(num);
    }

    formatDate(date, options = {}) {
        return new Date(date).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            ...options
        });
    }

    formatDateTime(date) {
        return new Date(date).toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Animation helpers
    animateIn(element, animationClass = 'fade-in') {
        element.classList.add(animationClass);
    }

    animateOut(element, animationClass = 'fade-out') {
        return new Promise(resolve => {
            element.classList.add(animationClass);
            element.addEventListener('animationend', () => {
                element.remove();
                resolve();
            }, { once: true });
        });
    }
}

// Dashboard-specific functionality
class Dashboard {
    constructor(app) {
        this.app = app;
        this.charts = {};
        this.refreshInterval = 120000; // 2 minutes (optimized)
    }

    async initialize() {
        await this.loadDashboardData();
        this.startAutoRefresh();
    }

    async loadDashboardData() {
        try {
            const [systemStats, recentActivity, sourcesStatus] = await Promise.all([
                this.app.fetchData('/research/stats'),
                this.app.fetchData('/research/activity'),
                this.app.fetchData('/scraping/sources/status')
            ]);

            this.updateSystemStats(systemStats);
            this.updateRecentActivity(recentActivity);
            this.updateSourcesStatus(sourcesStatus.sources || []);
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
            this.app.showFlashMessage('Failed to load dashboard data', 'warning');
        }
    }

    updateSystemStats(stats) {
        // Update stat cards with correct ID mapping
        const statMappings = {
            'total_articles': 'stat-articles',
            'total_reports': 'stat-reports', 
            'active_sources': 'stat-sources',
            'total_entities': 'stat-entities'
        };
        
        Object.entries(stats).forEach(([key, value]) => {
            const elementId = statMappings[key];
            if (elementId) {
                const element = document.getElementById(elementId);
                if (element) {
                    element.textContent = this.app.formatNumber(value);
                }
            }
        });
    }

    updateRecentActivity(activities) {
        const container = document.getElementById('recent-activity');
        if (!container) return;

        container.innerHTML = activities.map(activity => `
            <div class="d-flex align-items-center mb-3">
                <div class="flex-shrink-0">
                    <div class="status-dot ${activity.type}"></div>
                </div>
                <div class="flex-grow-1 ms-3">
                    <div class="fw-medium">${activity.title}</div>
                    <small class="text-muted">${this.app.formatDateTime(activity.timestamp)}</small>
                </div>
            </div>
        `).join('');
    }

    updateSourcesStatus(sources) {
        const container = document.getElementById('sources-status');
        if (!container) return;

        container.innerHTML = sources.map(source => `
            <tr>
                <td>
                    <div class="status-indicator">
                        <span class="status-dot ${source.status}"></span>
                        ${source.name}
                    </div>
                </td>
                <td>${source.last_scraped ? this.app.formatDateTime(source.last_scraped) : 'Never'}</td>
                <td>
                    <span class="badge badge-status-${source.status}">${source.status}</span>
                </td>
            </tr>
        `).join('');
    }

    startAutoRefresh() {
        setInterval(() => this.loadDashboardData(), this.refreshInterval);
    }
}

// Initialize the application
let app;
let dashboard;

document.addEventListener('DOMContentLoaded', () => {
    app = new ResearchSynthesisApp();
    
    // Initialize debug console
    window.debugConsole = new DebugConsole();
    console.info('🐛 Debug Console initialized. Press Ctrl+Shift+D to toggle, Ctrl+Shift+T to test APIs');
    
    // Initialize dashboard if on dashboard page
    if (window.location.pathname === '/dashboard' || window.location.pathname === '/') {
        dashboard = new Dashboard(app);
        dashboard.initialize();
    }
});

// Debug Console Class
class DebugConsole {
    constructor() {
        this.logs = [];
        this.maxLogs = 100;
        this.isVisible = false;
        this.autoScroll = true;
        this.initDebugPanel();
        this.setupKeyboardShortcuts();
        this.interceptConsole();
    }

    initDebugPanel() {
        const debugPanel = document.createElement('div');
        debugPanel.id = 'debug-panel';
        debugPanel.className = 'debug-panel';
        debugPanel.innerHTML = `
            <div class="debug-header">
                <div class="debug-title">
                    <i class="fas fa-bug me-2"></i>Frontend Debug Console
                </div>
                <div class="debug-controls">
                    <button class="btn btn-sm btn-outline-light" onclick="debugConsole.testAllAPIs()">
                        <i class="fas fa-vial me-1"></i>Test APIs
                    </button>
                    <button class="btn btn-sm btn-outline-light" onclick="debugConsole.exportLogs()">
                        <i class="fas fa-download me-1"></i>Export
                    </button>
                    <button class="btn btn-sm btn-outline-light" onclick="debugConsole.clearLogs()">
                        <i class="fas fa-trash me-1"></i>Clear
                    </button>
                    <button class="btn btn-sm btn-outline-light" onclick="debugConsole.toggle()">
                        <i class="fas fa-times me-1"></i>Close
                    </button>
                </div>
            </div>
            <div class="debug-content">
                <div class="debug-tabs">
                    <button class="debug-tab active" onclick="debugConsole.showTab('logs')">
                        <i class="fas fa-list me-1"></i>Logs
                    </button>
                    <button class="debug-tab" onclick="debugConsole.showTab('network')">
                        <i class="fas fa-network-wired me-1"></i>Network
                    </button>
                    <button class="debug-tab" onclick="debugConsole.showTab('storage')">
                        <i class="fas fa-database me-1"></i>Storage
                    </button>
                    <button class="debug-tab" onclick="debugConsole.showTab('system')">
                        <i class="fas fa-cogs me-1"></i>System
                    </button>
                </div>
                <div class="debug-body">
                    <div id="debug-logs" class="debug-tab-content active">
                        <div class="debug-log-container" id="debug-log-container"></div>
                    </div>
                    <div id="debug-network" class="debug-tab-content">
                        <div class="debug-network-container" id="debug-network-container">
                            <div class="text-center text-muted p-3">
                                <i class="fas fa-network-wired fa-2x mb-2"></i>
                                <div>Network requests will appear here</div>
                            </div>
                        </div>
                    </div>
                    <div id="debug-storage" class="debug-tab-content">
                        <div class="debug-storage-container" id="debug-storage-container"></div>
                    </div>
                    <div id="debug-system" class="debug-tab-content">
                        <div class="debug-system-container" id="debug-system-container"></div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(debugPanel);
        this.panel = debugPanel;
        
        this.updateStorageTab();
        this.updateSystemTab();
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+Shift+D to toggle debug panel
            if (e.ctrlKey && e.shiftKey && e.key === 'D') {
                e.preventDefault();
                this.toggle();
            }
            // Ctrl+Shift+T to test all APIs
            if (e.ctrlKey && e.shiftKey && e.key === 'T') {
                e.preventDefault();
                this.testAllAPIs();
            }
        });
    }

    interceptConsole() {
        const originalLog = console.log;
        const originalError = console.error;
        const originalWarn = console.warn;
        const originalInfo = console.info;

        console.log = (...args) => {
            this.addLog('log', args);
            originalLog.apply(console, args);
        };

        console.error = (...args) => {
            this.addLog('error', args);
            originalError.apply(console, args);
        };

        console.warn = (...args) => {
            this.addLog('warn', args);
            originalWarn.apply(console, args);
        };

        console.info = (...args) => {
            this.addLog('info', args);
            originalInfo.apply(console, args);
        };
    }

    addLog(level, args) {
        const timestamp = new Date().toLocaleTimeString();
        const message = args.map(arg => 
            typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
        ).join(' ');

        const logEntry = { timestamp, level, message };
        this.logs.unshift(logEntry);

        if (this.logs.length > this.maxLogs) {
            this.logs.pop();
        }

        this.updateLogsDisplay();
    }

    addNetworkLog(method, url, status, duration, data = null) {
        const container = document.getElementById('debug-network-container');
        if (!container) return;

        const networkEntry = document.createElement('div');
        networkEntry.className = `debug-network-entry status-${status}`;
        networkEntry.innerHTML = `
            <div class="debug-network-header">
                <span class="debug-network-method ${method.toLowerCase()}">${method}</span>
                <span class="debug-network-url">${url}</span>
                <span class="debug-network-status status-${status >= 200 && status < 300 ? 'success' : 'error'}">${status}</span>
                <span class="debug-network-time">${duration}ms</span>
                <span class="debug-network-timestamp">${new Date().toLocaleTimeString()}</span>
            </div>
            ${data ? `<div class="debug-network-data"><pre>${JSON.stringify(data, null, 2)}</pre></div>` : ''}
        `;

        container.insertBefore(networkEntry, container.firstChild);

        // Keep only last 50 network entries
        const entries = container.querySelectorAll('.debug-network-entry');
        if (entries.length > 50) {
            entries[entries.length - 1].remove();
        }
    }

    updateLogsDisplay() {
        const container = document.getElementById('debug-log-container');
        if (!container) return;

        container.innerHTML = this.logs.map(log => `
            <div class="debug-log-entry debug-log-${log.level}">
                <span class="debug-log-time">${log.timestamp}</span>
                <span class="debug-log-level">${log.level.toUpperCase()}</span>
                <span class="debug-log-message">${log.message}</span>
            </div>
        `).join('');

        if (this.autoScroll) {
            container.scrollTop = 0;
        }
    }

    updateStorageTab() {
        const container = document.getElementById('debug-storage-container');
        if (!container) return;

        const localStorage = this.getLocalStorageData();
        const sessionStorage = this.getSessionStorageData();

        container.innerHTML = `
            <div class="debug-storage-section">
                <h6><i class="fas fa-hdd me-2"></i>Local Storage</h6>
                <div class="debug-storage-items">
                    ${localStorage.length ? localStorage.map(item => `
                        <div class="debug-storage-item">
                            <strong>${item.key}:</strong> ${item.value}
                        </div>
                    `).join('') : '<div class="text-muted">No items</div>'}
                </div>
            </div>
            <div class="debug-storage-section">
                <h6><i class="fas fa-clock me-2"></i>Session Storage</h6>
                <div class="debug-storage-items">
                    ${sessionStorage.length ? sessionStorage.map(item => `
                        <div class="debug-storage-item">
                            <strong>${item.key}:</strong> ${item.value}
                        </div>
                    `).join('') : '<div class="text-muted">No items</div>'}
                </div>
            </div>
        `;
    }

    updateSystemTab() {
        const container = document.getElementById('debug-system-container');
        if (!container) return;

        container.innerHTML = `
            <div class="debug-system-info">
                <div class="debug-system-section">
                    <h6><i class="fas fa-globe me-2"></i>Browser Info</h6>
                    <div><strong>User Agent:</strong> ${navigator.userAgent}</div>
                    <div><strong>Platform:</strong> ${navigator.platform}</div>
                    <div><strong>Language:</strong> ${navigator.language}</div>
                    <div><strong>Online:</strong> ${navigator.onLine ? 'Yes' : 'No'}</div>
                </div>
                <div class="debug-system-section">
                    <h6><i class="fas fa-window-maximize me-2"></i>Window Info</h6>
                    <div><strong>Location:</strong> ${window.location.href}</div>
                    <div><strong>Screen Size:</strong> ${screen.width}x${screen.height}</div>
                    <div><strong>Window Size:</strong> ${window.innerWidth}x${window.innerHeight}</div>
                    <div><strong>Device Pixel Ratio:</strong> ${window.devicePixelRatio}</div>
                </div>
                <div class="debug-system-section">
                    <h6><i class="fas fa-memory me-2"></i>Performance</h6>
                    <div><strong>DOM Nodes:</strong> ${document.getElementsByTagName('*').length}</div>
                    <div><strong>Load Time:</strong> ${performance.now().toFixed(2)}ms</div>
                    <div><strong>Memory:</strong> ${this.getMemoryInfo()}</div>
                </div>
            </div>
        `;
    }

    getLocalStorageData() {
        const items = [];
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            const value = localStorage.getItem(key);
            items.push({ key, value: this.truncateString(value, 100) });
        }
        return items;
    }

    getSessionStorageData() {
        const items = [];
        for (let i = 0; i < sessionStorage.length; i++) {
            const key = sessionStorage.key(i);
            const value = sessionStorage.getItem(key);
            items.push({ key, value: this.truncateString(value, 100) });
        }
        return items;
    }

    getMemoryInfo() {
        if ('memory' in performance) {
            const memory = performance.memory;
            return `Used: ${(memory.usedJSHeapSize / 1048576).toFixed(2)}MB / Limit: ${(memory.jsHeapSizeLimit / 1048576).toFixed(2)}MB`;
        }
        return 'Not available';
    }

    truncateString(str, length) {
        return str && str.length > length ? str.substring(0, length) + '...' : str;
    }

    toggle() {
        this.isVisible = !this.isVisible;
        this.panel.classList.toggle('visible', this.isVisible);
        
        if (this.isVisible) {
            this.updateStorageTab();
            this.updateSystemTab();
        }
    }

    showTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.debug-tab').forEach(tab => {
            tab.classList.toggle('active', tab.onclick.toString().includes(tabName));
        });

        // Update tab content
        document.querySelectorAll('.debug-tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `debug-${tabName}`);
        });

        if (tabName === 'storage') {
            this.updateStorageTab();
        } else if (tabName === 'system') {
            this.updateSystemTab();
        }
    }

    clearLogs() {
        this.logs = [];
        this.updateLogsDisplay();
        document.getElementById('debug-network-container').innerHTML = `
            <div class="text-center text-muted p-3">
                <i class="fas fa-network-wired fa-2x mb-2"></i>
                <div>Network requests will appear here</div>
            </div>
        `;
    }

    exportLogs() {
        const data = {
            timestamp: new Date().toISOString(),
            logs: this.logs,
            system: {
                userAgent: navigator.userAgent,
                url: window.location.href,
                screenSize: `${screen.width}x${screen.height}`,
                windowSize: `${window.innerWidth}x${window.innerHeight}`
            }
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `debug-logs-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    async testAllAPIs() {
        console.info('🧪 Starting comprehensive API tests...');
        
        const tests = [
            { name: 'Health Check', url: '/api/v1/health', method: 'GET' },
            { name: 'Knowledge Stats', url: '/api/v1/knowledge/', method: 'GET' },
            { name: 'Knowledge Graph Export', url: '/api/v1/knowledge/graph/export', method: 'GET' },
            { name: 'Research Stats', url: '/api/v1/research/stats', method: 'GET' },
            { name: 'Research Activity', url: '/api/v1/research/activity', method: 'GET' },
            { name: 'Reports List', url: '/api/v1/reports/', method: 'GET' },
            { name: 'Scraping Status', url: '/api/v1/scraping/', method: 'GET' },
            { name: 'Sources Status', url: '/api/v1/scraping/sources/status', method: 'GET' },
            { name: 'Scraping Jobs', url: '/api/v1/scraping/jobs', method: 'GET' },
            { name: 'Scraping Config', url: '/api/v1/scraping/configure', method: 'GET' }
        ];

        let passed = 0;
        let failed = 0;

        for (const test of tests) {
            try {
                const startTime = performance.now();
                const response = await axios.get(test.url);
                const endTime = performance.now();
                const duration = Math.round(endTime - startTime);

                this.addNetworkLog(test.method, test.url, response.status, duration);
                console.log(`✅ ${test.name}: ${response.status} (${duration}ms)`);
                passed++;
            } catch (error) {
                const duration = 0;
                const status = error.response?.status || 0;
                this.addNetworkLog(test.method, test.url, status, duration);
                console.error(`❌ ${test.name}: ${status} - ${error.message}`);
                failed++;
            }
        }

        console.info(`🧪 API Tests completed: ${passed} passed, ${failed} failed`);
    }
}

// Export for global access
window.RSApp = {
    app: () => app,
    dashboard: () => dashboard,
    showMessage: (msg, type) => app?.showFlashMessage(msg, type),
    showLoading: (show, title, message, progress) => app?.showLoadingOverlay(show, title, message, progress),
    showGlobalLoading: (options) => app?.showGlobalLoading(options),
    hideGlobalLoading: () => app?.hideGlobalLoading(),
    updateLoadingProgress: (progress, title, message) => app?.updateLoadingProgress(progress, title, message)
};

// Global debug console
window.debugConsole = null;

// Image Lazy Loading Implementation
class LazyImageLoader {
    constructor() {
        this.imageObserver = null;
        this.init();
    }
    
    init() {
        if ('IntersectionObserver' in window) {
            this.imageObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        if (img.dataset.src) {
                            img.src = img.dataset.src;
                            img.classList.add('loaded');
                            img.removeAttribute('data-src');
                            observer.unobserve(img);
                        }
                    }
                });
            }, {
                rootMargin: '50px 0px',
                threshold: 0.01
            });
            
            // Observe all images with lazy class
            this.observeImages();
        } else {
            // Fallback for browsers without IntersectionObserver
            this.loadAllImages();
        }
    }
    
    observeImages() {
        const lazyImages = document.querySelectorAll('img.lazy');
        lazyImages.forEach(img => {
            this.imageObserver.observe(img);
        });
    }
    
    loadAllImages() {
        const lazyImages = document.querySelectorAll('img.lazy');
        lazyImages.forEach(img => {
            if (img.dataset.src) {
                img.src = img.dataset.src;
                img.classList.add('loaded');
            }
        });
    }
}

// Initialize lazy loading on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    window.lazyLoader = new LazyImageLoader();
});