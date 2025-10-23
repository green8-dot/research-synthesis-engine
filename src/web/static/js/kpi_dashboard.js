/**
 * Enhanced KPI Dashboard with Percentage Changes and Trends
 */
class KPIDashboard {
    constructor() {
        this.updateInterval = null;
        this.isLoading = false;
    }

    async init() {
        console.log('Initializing enhanced KPI dashboard...');
        
        // Record current metrics first
        await this.recordCurrentMetrics();
        
        // Load enhanced KPIs
        await this.loadEnhancedKPIs();
        
        // Set up auto-refresh every 30 seconds
        this.updateInterval = setInterval(() => {
            this.loadEnhancedKPIs();
        }, 30000);
        
        console.log('KPI dashboard initialized successfully');
    }

    async recordCurrentMetrics() {
        try {
            await fetch('/api/v1/kpi/record', { method: 'POST' });
            console.log('Current metrics recorded');
        } catch (error) {
            console.warn('Failed to record metrics:', error);
        }
    }

    async loadEnhancedKPIs() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        
        try {
            const response = await fetch('/api/v1/kpi/dashboard');
            const data = await response.json();
            
            this.updateDashboardKPIs(data.metrics);
            this.updateOverallStatus(data.summary);
            
        } catch (error) {
            console.error('Failed to load enhanced KPIs:', error);
            this.showKPIError();
        } finally {
            this.isLoading = false;
        }
    }

    updateDashboardKPIs(metrics) {
        const metricMappings = {
            'total_articles': 'articles',
            'total_reports': 'reports', 
            'active_sources': 'sources',
            'total_entities': 'entities'
        };

        for (const [metricKey, metricData] of Object.entries(metrics)) {
            const elementKey = metricMappings[metricKey];
            if (!elementKey) continue;

            // Update the number
            const numberElement = document.getElementById(`stat-${elementKey}`);
            if (numberElement) {
                this.animateNumber(numberElement, metricData.current_value);
            }

            // Update the change indicator
            const changeElement = document.getElementById(`change-${elementKey}`);
            if (changeElement) {
                this.updateChangeIndicator(changeElement, metricData);
            }
        }
    }

    animateNumber(element, targetValue) {
        const currentValue = parseInt(element.textContent) || 0;
        
        if (currentValue === targetValue) return;

        const duration = 1000;
        const startTime = Date.now();
        
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const currentNum = Math.floor(currentValue + (targetValue - currentValue) * progress);
            element.textContent = currentNum.toLocaleString();
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }

    updateChangeIndicator(element, metricData) {
        const { daily_change, daily_change_pct, weekly_change, weekly_change_pct, trend, trend_icon, display_change } = metricData;
        
        // Determine trend class and message
        let trendClass = 'text-muted';
        let iconClass = 'fa-minus';
        
        if (trend_icon) {
            iconClass = trend_icon.split(' ')[0].replace('fa-', '');
            trendClass = trend_icon.split(' ').slice(1).join(' ');
        }
        
        // Build the change HTML with timeframe reference
        let changeHtml = `<i class="fas fa-${iconClass} me-1"></i>`;
        
        if (daily_change !== 0) {
            changeHtml += `<span class="me-2">${daily_change > 0 ? '+' : ''}${daily_change} today</span>`;
            if (daily_change_pct !== 0) {
                changeHtml += `<small class="text-muted">(${daily_change_pct > 0 ? '+' : ''}${daily_change_pct}%)</small>`;
            }
        } else if (weekly_change !== 0) {
            changeHtml += `<span class="me-2">${weekly_change > 0 ? '+' : ''}${weekly_change} this week</span>`;
            if (weekly_change_pct !== 0) {
                changeHtml += `<small class="text-muted">(${weekly_change_pct > 0 ? '+' : ''}${weekly_change_pct}%)</small>`;
            }
        } else {
            changeHtml += '<span class="text-muted">No recent changes</span>';
        }

        element.innerHTML = changeHtml;
        element.className = `stat-change ${this.getTrendColorClass(trend)}`;
        
        // Add tooltip with more info
        element.title = this.getTooltipText(metricData);
    }

    getTrendColorClass(trend) {
        switch (trend) {
            case 'strong_up':
                return 'text-success fw-bold';
            case 'up':
                return 'text-success';
            case 'stable':
                return 'text-muted';
            case 'down':
                return 'text-warning';
            case 'strong_down':
                return 'text-danger fw-bold';
            default:
                return 'text-muted';
        }
    }

    getTooltipText(metricData) {
        const { daily_change, daily_change_pct, weekly_change, weekly_change_pct, trend } = metricData;
        
        let tooltip = `Daily: ${daily_change > 0 ? '+' : ''}${daily_change} (${daily_change_pct}%)\n`;
        tooltip += `Weekly: ${weekly_change > 0 ? '+' : ''}${weekly_change} (${weekly_change_pct}%)\n`;
        tooltip += `Trend: ${trend.replace('_', ' ')}`;
        
        return tooltip;
    }

    updateOverallStatus(summary) {
        // You can add overall status updates here if needed
        console.log('Overall system status:', summary.overall_status);
        
        // Update page title or header with status if elements exist
        const statusElements = document.querySelectorAll('[data-system-status]');
        statusElements.forEach(element => {
            element.textContent = summary.status_message || '';
            element.className = `system-status ${summary.overall_status}`;
        });
    }

    showKPIError() {
        // Show error state in KPI cards
        const changeElements = document.querySelectorAll('[id^="change-"]');
        changeElements.forEach(element => {
            element.innerHTML = '<i class="fas fa-exclamation-triangle text-danger me-1"></i>Error';
            element.className = 'stat-change text-danger';
        });
    }

    async loadHistoricalData(metric, days = 7) {
        try {
            const response = await fetch(`/api/v1/kpi/historical/${metric}?days=${days}`);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Failed to load historical data:', error);
            return null;
        }
    }

    async getTrendAnalysis() {
        try {
            const response = await fetch('/api/v1/kpi/trends');
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Failed to get trend analysis:', error);
            return null;
        }
    }

    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
}

// Global instance
let kpiDashboard = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Wait a moment for other scripts to load
    setTimeout(() => {
        kpiDashboard = new KPIDashboard();
        kpiDashboard.init().catch(console.error);
    }, 1000);
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (kpiDashboard) {
        kpiDashboard.destroy();
    }
});