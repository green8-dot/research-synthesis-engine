/**
 * Audit Display and Real-time Monitoring System
 * Provides comprehensive frontend updates for audit trail, KPIs, and system health
 */

class AuditDisplay {
    constructor() {
        this.refreshInterval = 5000; // 5 seconds
        this.auditData = [];
        this.systemHealth = {};
        this.kpiCache = {};
        this.init();
    }

    init() {
        // Start monitoring
        this.startRealTimeMonitoring();
        this.createAuditPanel();
        this.setupEventListeners();
    }

    createAuditPanel() {
        // Create floating audit panel if it doesn't exist
        if (!document.getElementById('audit-panel')) {
            const auditPanelHtml = `
                <div id="audit-panel" class="audit-panel collapsed">
                    <div class="audit-header" onclick="auditDisplay.togglePanel()">
                        <i class="fas fa-shield-alt me-2"></i>
                        <span>System Audit Monitor</span>
                        <span class="audit-status-badge ms-auto">
                            <span class="badge bg-success" id="audit-status">Active</span>
                        </span>
                        <i class="fas fa-chevron-up ms-2" id="audit-toggle-icon"></i>
                    </div>
                    <div class="audit-content">
                        <div class="audit-tabs">
                            <button class="audit-tab active" onclick="auditDisplay.showTab('realtime')">
                                <i class="fas fa-stream"></i> Real-time
                            </button>
                            <button class="audit-tab" onclick="auditDisplay.showTab('health')">
                                <i class="fas fa-heartbeat"></i> Health
                            </button>
                            <button class="audit-tab" onclick="auditDisplay.showTab('kpis')">
                                <i class="fas fa-chart-line"></i> KPIs
                            </button>
                            <button class="audit-tab" onclick="auditDisplay.showTab('jobs')">
                                <i class="fas fa-tasks"></i> Jobs
                            </button>
                        </div>
                        <div id="audit-tab-content" class="audit-tab-content">
                            <div id="realtime-tab" class="tab-pane active">
                                <div class="audit-stream" id="audit-stream"></div>
                            </div>
                            <div id="health-tab" class="tab-pane">
                                <div class="system-health-grid" id="system-health-grid"></div>
                            </div>
                            <div id="kpis-tab" class="tab-pane">
                                <div class="kpi-grid" id="kpi-grid"></div>
                            </div>
                            <div id="jobs-tab" class="tab-pane">
                                <div class="jobs-list" id="jobs-list"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <style>
                    .audit-panel {
                        position: fixed;
                        bottom: 0;
                        right: 20px;
                        width: 400px;
                        background: rgba(30, 30, 40, 0.95);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        border-radius: 12px 12px 0 0;
                        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.3);
                        z-index: 1050;
                        transition: all 0.3s ease;
                        backdrop-filter: blur(10px);
                    }
                    
                    .audit-panel.collapsed {
                        height: 50px;
                    }
                    
                    .audit-panel:not(.collapsed) {
                        height: 500px;
                    }
                    
                    .audit-header {
                        padding: 15px;
                        background: rgba(40, 40, 50, 0.9);
                        border-radius: 12px 12px 0 0;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        color: #fff;
                        font-weight: 500;
                    }
                    
                    .audit-content {
                        padding: 15px;
                        max-height: 450px;
                        overflow-y: auto;
                        display: none;
                    }
                    
                    .audit-panel:not(.collapsed) .audit-content {
                        display: block;
                    }
                    
                    .audit-tabs {
                        display: flex;
                        gap: 10px;
                        margin-bottom: 15px;
                    }
                    
                    .audit-tab {
                        flex: 1;
                        padding: 8px;
                        background: rgba(255, 255, 255, 0.05);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        border-radius: 6px;
                        color: #aaa;
                        font-size: 0.85rem;
                        cursor: pointer;
                        transition: all 0.2s;
                    }
                    
                    .audit-tab.active {
                        background: rgba(100, 100, 255, 0.2);
                        border-color: #6464ff;
                        color: #fff;
                    }
                    
                    .audit-stream {
                        max-height: 350px;
                        overflow-y: auto;
                    }
                    
                    .audit-entry {
                        padding: 10px;
                        margin-bottom: 8px;
                        background: rgba(255, 255, 255, 0.05);
                        border-radius: 6px;
                        border-left: 3px solid;
                        font-size: 0.85rem;
                        animation: slideIn 0.3s ease;
                    }
                    
                    .audit-entry.success {
                        border-color: #27ae60;
                    }
                    
                    .audit-entry.warning {
                        border-color: #f39c12;
                    }
                    
                    .audit-entry.error {
                        border-color: #e74c3c;
                    }
                    
                    .audit-entry.info {
                        border-color: #3498db;
                    }
                    
                    @keyframes slideIn {
                        from {
                            opacity: 0;
                            transform: translateX(20px);
                        }
                        to {
                            opacity: 1;
                            transform: translateX(0);
                        }
                    }
                    
                    .system-health-grid {
                        display: grid;
                        grid-template-columns: repeat(2, 1fr);
                        gap: 10px;
                    }
                    
                    .health-card {
                        padding: 15px;
                        background: rgba(255, 255, 255, 0.05);
                        border-radius: 8px;
                        text-align: center;
                    }
                    
                    .health-value {
                        font-size: 1.8rem;
                        font-weight: bold;
                        color: #fff;
                    }
                    
                    .health-label {
                        font-size: 0.75rem;
                        color: #aaa;
                        text-transform: uppercase;
                        margin-top: 5px;
                    }
                    
                    .kpi-grid {
                        display: grid;
                        gap: 10px;
                    }
                    
                    .kpi-item {
                        display: flex;
                        justify-content: space-between;
                        padding: 10px;
                        background: rgba(255, 255, 255, 0.05);
                        border-radius: 6px;
                    }
                    
                    .kpi-name {
                        color: #aaa;
                        font-size: 0.85rem;
                    }
                    
                    .kpi-value {
                        color: #fff;
                        font-weight: 600;
                    }
                    
                    .tab-pane {
                        display: none;
                    }
                    
                    .tab-pane.active {
                        display: block;
                    }
                    
                    .jobs-list {
                        max-height: 350px;
                        overflow-y: auto;
                    }
                    
                    .job-item {
                        padding: 10px;
                        margin-bottom: 8px;
                        background: rgba(255, 255, 255, 0.05);
                        border-radius: 6px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    }
                    
                    .job-status {
                        padding: 3px 8px;
                        border-radius: 12px;
                        font-size: 0.75rem;
                        font-weight: 500;
                    }
                    
                    .job-status.running {
                        background: rgba(52, 152, 219, 0.2);
                        color: #3498db;
                    }
                    
                    .job-status.completed {
                        background: rgba(39, 174, 96, 0.2);
                        color: #27ae60;
                    }
                    
                    .job-status.failed {
                        background: rgba(231, 76, 60, 0.2);
                        color: #e74c3c;
                    }
                </style>
            `;
            
            document.body.insertAdjacentHTML('beforeend', auditPanelHtml);
        }
    }

    togglePanel() {
        const panel = document.getElementById('audit-panel');
        const icon = document.getElementById('audit-toggle-icon');
        
        if (panel) {
            panel.classList.toggle('collapsed');
            icon.className = panel.classList.contains('collapsed') ? 
                'fas fa-chevron-up ms-2' : 'fas fa-chevron-down ms-2';
        }
    }

    showTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.audit-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        event.target.classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        const tabPane = document.getElementById(`${tabName}-tab`);
        if (tabPane) {
            tabPane.classList.add('active');
        }
        
        // Load tab-specific data
        switch(tabName) {
            case 'health':
                this.loadSystemHealth();
                break;
            case 'kpis':
                this.loadKPIs();
                break;
            case 'jobs':
                this.loadJobs();
                break;
        }
    }

    async startRealTimeMonitoring() {
        // Initial load
        await this.updateAllData();
        
        // Set up intervals for different update frequencies
        setInterval(() => this.updateRealTimeStream(), this.refreshInterval);
        setInterval(() => this.updateKPIs(), 10000); // KPIs every 10 seconds
        setInterval(() => this.updateSystemHealth(), 15000); // Health every 15 seconds
        setInterval(() => this.updateJobs(), 5000); // Jobs every 5 seconds
    }

    async updateAllData() {
        await Promise.all([
            this.loadSystemHealth(),
            this.loadKPIs(),
            this.loadJobs(),
            this.updateRealTimeStream()
        ]);
    }

    async updateRealTimeStream() {
        try {
            // Fetch recent audit logs
            const response = await fetch('/api/v1/audit/recent');
            if (response.ok) {
                const data = await response.json();
                this.displayAuditEntries(data.logs || []);
            }
        } catch (error) {
            console.error('Error fetching audit logs:', error);
        }
    }

    displayAuditEntries(logs) {
        const stream = document.getElementById('audit-stream');
        if (!stream) return;
        
        // Clear old entries if too many
        if (stream.children.length > 50) {
            stream.innerHTML = '';
        }
        
        logs.forEach(log => {
            const entryId = `audit-${log.id || Date.now()}`;
            if (!document.getElementById(entryId)) {
                const entry = document.createElement('div');
                entry.id = entryId;
                entry.className = `audit-entry ${log.level || 'info'}`;
                entry.innerHTML = `
                    <div class="d-flex justify-content-between">
                        <small class="text-muted">${new Date(log.timestamp).toLocaleTimeString()}</small>
                        <span class="badge bg-${this.getLevelColor(log.level)}">${log.level || 'info'}</span>
                    </div>
                    <div class="mt-1">${log.message}</div>
                `;
                stream.insertBefore(entry, stream.firstChild);
            }
        });
    }

    getLevelColor(level) {
        const colors = {
            'success': 'success',
            'error': 'danger',
            'warning': 'warning',
            'info': 'info',
            'debug': 'secondary'
        };
        return colors[level] || 'secondary';
    }

    async loadSystemHealth() {
        try {
            const response = await fetch('/api/v1/system-status');
            if (response.ok) {
                const data = await response.json();
                this.displaySystemHealth(data);
            }
        } catch (error) {
            console.error('Error loading system health:', error);
        }
    }

    displaySystemHealth(data) {
        const grid = document.getElementById('system-health-grid');
        if (!grid) return;
        
        const metrics = {
            'CPU Usage': `${data.cpu_percent || 0}%`,
            'Memory': `${data.memory_percent || 0}%`,
            'Disk': `${data.disk_percent || 0}%`,
            'Uptime': this.formatUptime(data.uptime || 0),
            'Active Jobs': data.active_jobs || 0,
            'Health Score': `${data.health_score || 100}%`
        };
        
        grid.innerHTML = Object.entries(metrics).map(([label, value]) => `
            <div class="health-card">
                <div class="health-value">${value}</div>
                <div class="health-label">${label}</div>
            </div>
        `).join('');
    }

    formatUptime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }

    async loadKPIs() {
        try {
            const response = await fetch('/api/v1/kpi/dashboard');
            if (response.ok) {
                const data = await response.json();
                this.displayKPIs(data);
                this.updateDashboardKPIs(data);
            }
        } catch (error) {
            console.error('Error loading KPIs:', error);
        }
    }

    displayKPIs(data) {
        const grid = document.getElementById('kpi-grid');
        if (!grid) return;
        
        const kpis = [
            { name: 'Total Articles', value: data.total_articles || 0 },
            { name: 'Reports Generated', value: data.total_reports || 0 },
            { name: 'Active Sources', value: data.active_sources || 0 },
            { name: 'Entities Tracked', value: data.total_entities || 0 },
            { name: 'Scraping Success Rate', value: `${data.success_rate || 0}%` },
            { name: 'Avg Response Time', value: `${data.avg_response_time || 0}ms` }
        ];
        
        grid.innerHTML = kpis.map(kpi => `
            <div class="kpi-item">
                <span class="kpi-name">${kpi.name}</span>
                <span class="kpi-value">${kpi.value}</span>
            </div>
        `).join('');
    }

    updateDashboardKPIs(data) {
        // Update main dashboard KPIs if they exist
        const updateElement = (id, value) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
                // Add update animation
                element.style.animation = 'pulse 0.5s ease';
                setTimeout(() => element.style.animation = '', 500);
            }
        };
        
        updateElement('stat-articles', data.total_articles || 0);
        updateElement('stat-reports', data.total_reports || 0);
        updateElement('stat-sources', data.active_sources || 0);
        updateElement('stat-entities', data.total_entities || 0);
        updateElement('active-jobs', data.active_jobs || 0);
    }

    async loadJobs() {
        try {
            const response = await fetch('/api/v1/scraping/jobs?limit=10');
            if (response.ok) {
                const data = await response.json();
                this.displayJobs(data.jobs || []);
            }
        } catch (error) {
            console.error('Error loading jobs:', error);
        }
    }

    displayJobs(jobs) {
        const list = document.getElementById('jobs-list');
        if (!list) return;
        
        if (jobs.length === 0) {
            list.innerHTML = '<div class="text-center text-muted">No active jobs</div>';
            return;
        }
        
        list.innerHTML = jobs.map(job => `
            <div class="job-item">
                <div>
                    <div class="text-white">${job.job_type || 'Unknown'}</div>
                    <small class="text-muted">ID: ${job.id}</small>
                </div>
                <span class="job-status ${job.status}">${job.status}</span>
            </div>
        `).join('');
    }

    async updateKPIs() {
        const activeTab = document.querySelector('.tab-pane.active');
        if (activeTab && activeTab.id === 'kpis-tab') {
            await this.loadKPIs();
        }
    }

    async updateSystemHealth() {
        const activeTab = document.querySelector('.tab-pane.active');
        if (activeTab && activeTab.id === 'health-tab') {
            await this.loadSystemHealth();
        }
    }

    async updateJobs() {
        const activeTab = document.querySelector('.tab-pane.active');
        if (activeTab && activeTab.id === 'jobs-tab') {
            await this.loadJobs();
        }
    }

    setupEventListeners() {
        // Listen for custom events
        window.addEventListener('kpi-update', (event) => {
            this.updateDashboardKPIs(event.detail);
        });
        
        window.addEventListener('job-update', (event) => {
            this.loadJobs();
        });
        
        window.addEventListener('audit-log', (event) => {
            this.displayAuditEntries([event.detail]);
        });
    }
}

// Initialize audit display when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    window.auditDisplay = new AuditDisplay();
    
    // Add CSS for pulse animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
    `;
    document.head.appendChild(style);
});

// Export for use in other scripts
window.AuditDisplay = AuditDisplay;