#!/usr/bin/env python3
"""
OrbitScope ML Intelligence Suite - Service Health Monitoring Dashboard
Real-time monitoring and alerting for all AI services

ORGANIZED AND OPTIMIZED - Operational Motto
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Set, Optional, Any
import psutil
import motor.motor_asyncio
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int

@dataclass
class ServiceHealth:
    """Health status of individual services"""
    service_id: str
    service_name: str
    status: str  # healthy, unhealthy, unknown
    response_time: float
    last_check: datetime
    error_count: int
    uptime_percent: float

@dataclass
class Alert:
    """System alert"""
    id: str
    level: AlertLevel
    message: str
    service_id: Optional[str]
    timestamp: datetime
    acknowledged: bool = False

class HealthMonitor:
    """Comprehensive health monitoring system"""
    
    def __init__(self):
        self.app = FastAPI(
            title="OrbitScope Health Monitor",
            description="Real-time service health monitoring",
            version="3.1.0"
        )
        
        # State management
        self.connected_websockets: Set[WebSocket] = set()
        self.service_registry = {}
        self.system_metrics_history: List[SystemMetrics] = []
        self.service_health_history: Dict[str, List[ServiceHealth]] = {}
        self.alerts: List[Alert] = []
        self.alert_counter = 0
        
        # MongoDB connection for storing metrics
        self.db_client = None
        self.metrics_collection = None
        
        # Monitoring configuration
        self.monitor_interval = 10  # seconds
        self.alert_thresholds = {
            'cpu_critical': 90.0,
            'cpu_warning': 80.0,
            'memory_critical': 95.0,
            'memory_warning': 85.0,
            'disk_critical': 95.0,
            'disk_warning': 90.0,
            'response_time_warning': 5.0,
            'response_time_critical': 10.0
        }
        
        self._setup_routes()
        logger.info("Health Monitor initialized")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the health monitoring dashboard"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/health/overview")
        async def health_overview():
            """Get overall system health overview"""
            current_metrics = await self._collect_system_metrics()
            service_statuses = await self._get_all_service_health()
            
            healthy_services = sum(1 for s in service_statuses if s.status == "healthy")
            total_services = len(service_statuses)
            
            return {
                "system_status": "healthy" if current_metrics.cpu_percent < 80 and current_metrics.memory_percent < 85 else "warning",
                "services_healthy": f"{healthy_services}/{total_services}",
                "cpu_usage": current_metrics.cpu_percent,
                "memory_usage": current_metrics.memory_percent,
                "disk_usage": current_metrics.disk_percent,
                "active_alerts": len([a for a in self.alerts if not a.acknowledged]),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @self.app.get("/api/metrics/system")
        async def system_metrics():
            """Get current system metrics"""
            current_metrics = await self._collect_system_metrics()
            return asdict(current_metrics)
        
        @self.app.get("/api/metrics/history")
        async def metrics_history(hours: int = 1):
            """Get historical system metrics"""
            cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)
            
            recent_metrics = [
                asdict(m) for m in self.system_metrics_history
                if m.timestamp.timestamp() > cutoff_time
            ]
            
            return {"metrics": recent_metrics, "hours": hours}
        
        @self.app.get("/api/services/health")
        async def all_services_health():
            """Get health status of all services"""
            services = await self._get_all_service_health()
            return {"services": [asdict(s) for s in services]}
        
        @self.app.get("/api/services/{service_id}/health")
        async def service_health(service_id: str):
            """Get health status of specific service"""
            health = await self._check_service_health(service_id)
            if health:
                return asdict(health)
            else:
                return {"error": "Service not found"}
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get all alerts"""
            return {"alerts": [asdict(a) for a in self.alerts]}
        
        @self.app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str):
            """Acknowledge an alert"""
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    await self._broadcast_update("alert_acknowledged", asdict(alert))
                    return {"status": "acknowledged"}
            
            return {"error": "Alert not found"}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self._handle_websocket(websocket)
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connections for real-time updates"""
        await websocket.accept()
        self.connected_websockets.add(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.connected_websockets)}")
        
        try:
            # Send initial data
            await websocket.send_json({
                "type": "initial_data",
                "data": {
                    "system_metrics": asdict(await self._collect_system_metrics()),
                    "services": [asdict(s) for s in await self._get_all_service_health()],
                    "alerts": [asdict(a) for a in self.alerts]
                }
            })
            
            # Keep connection alive and handle incoming messages
            while True:
                try:
                    data = await websocket.receive_text()
                    # Handle client messages if needed
                    logger.info(f"Received WebSocket message: {data}")
                except WebSocketDisconnect:
                    break
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        
        finally:
            self.connected_websockets.discard(websocket)
            logger.info(f"WebSocket disconnected. Total: {len(self.connected_websockets)}")
    
    async def _broadcast_update(self, update_type: str, data: Any):
        """Broadcast update to all connected WebSockets"""
        if not self.connected_websockets:
            return
        
        message = {
            "type": update_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        disconnected_websockets = set()
        
        for websocket in self.connected_websockets:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send WebSocket update: {e}")
                disconnected_websockets.add(websocket)
        
        # Remove disconnected websockets
        self.connected_websockets -= disconnected_websockets
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Get network statistics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Get active network connections
            active_connections = len(psutil.net_connections())
            
            metrics = SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                active_connections=active_connections
            )
            
            # Store in history (keep last 1000 entries)
            self.system_metrics_history.append(metrics)
            if len(self.system_metrics_history) > 1000:
                self.system_metrics_history = self.system_metrics_history[-1000:]
            
            # Check for alerts
            await self._check_system_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=0, memory_percent=0, disk_percent=0,
                network_bytes_sent=0, network_bytes_recv=0, active_connections=0
            )
    
    async def _get_all_service_health(self) -> List[ServiceHealth]:
        """Get health status of all registered services"""
        services = []
        
        # Predefined service endpoints for health checking
        service_endpoints = {
            "unified_gateway": "http://localhost:8000/health",
            "semantic_search": "http://localhost:8001/health",
            "sentiment_analysis": "http://localhost:8002/health",
            "advanced_nlp": "http://localhost:8003/health",
            "system_health": "http://localhost:8004/health",
            "resource_manager": "http://localhost:8005/health",
            "knowledge_graph": "http://localhost:8006/health",
            "web_scraper": "http://localhost:8007/health",
            "market_intelligence": "http://localhost:8008/health",
            "financial_analysis": "http://localhost:8009/health",
            "secure_plaid": "http://localhost:8010/health"
        }
        
        for service_id, health_url in service_endpoints.items():
            health = await self._check_service_health(service_id, health_url)
            if health:
                services.append(health)
        
        return services
    
    async def _check_service_health(self, service_id: str, health_url: str = None) -> Optional[ServiceHealth]:
        """Check health of a specific service"""
        if not health_url:
            # Default health URL pattern
            port_map = {
                "unified_gateway": 8000, "semantic_search": 8001, "sentiment_analysis": 8002,
                "advanced_nlp": 8003, "system_health": 8004, "resource_manager": 8005,
                "knowledge_graph": 8006, "web_scraper": 8007, "market_intelligence": 8008,
                "financial_analysis": 8009, "secure_plaid": 8010
            }
            port = port_map.get(service_id, 8000)
            health_url = f"http://localhost:{port}/health"
        
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(health_url, timeout=5.0)
                
                response_time = time.time() - start_time
                
                status = "healthy" if response.status_code == 200 else "unhealthy"
                
                # Get or create service history
                if service_id not in self.service_health_history:
                    self.service_health_history[service_id] = []
                
                health = ServiceHealth(
                    service_id=service_id,
                    service_name=service_id.replace('_', ' ').title(),
                    status=status,
                    response_time=response_time,
                    last_check=datetime.now(timezone.utc),
                    error_count=0,  # Will be calculated from history
                    uptime_percent=100.0  # Will be calculated from history
                )
                
                # Add to history
                self.service_health_history[service_id].append(health)
                if len(self.service_health_history[service_id]) > 100:
                    self.service_health_history[service_id] = self.service_health_history[service_id][-100:]
                
                # Calculate uptime percentage
                recent_checks = self.service_health_history[service_id][-20:]  # Last 20 checks
                if recent_checks:
                    healthy_count = sum(1 for h in recent_checks if h.status == "healthy")
                    health.uptime_percent = (healthy_count / len(recent_checks)) * 100
                
                # Check for service alerts
                await self._check_service_alerts(health)
                
                return health
                
        except Exception as e:
            logger.error(f"Health check failed for {service_id}: {e}")
            
            health = ServiceHealth(
                service_id=service_id,
                service_name=service_id.replace('_', ' ').title(),
                status="unhealthy",
                response_time=0,
                last_check=datetime.now(timezone.utc),
                error_count=1,
                uptime_percent=0
            )
            
            await self._check_service_alerts(health)
            return health
    
    async def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics for alert conditions"""
        # CPU alerts
        if metrics.cpu_percent >= self.alert_thresholds['cpu_critical']:
            await self._create_alert(
                AlertLevel.CRITICAL,
                f"Critical CPU usage: {metrics.cpu_percent:.1f}%",
                None
            )
        elif metrics.cpu_percent >= self.alert_thresholds['cpu_warning']:
            await self._create_alert(
                AlertLevel.WARNING,
                f"High CPU usage: {metrics.cpu_percent:.1f}%",
                None
            )
        
        # Memory alerts
        if metrics.memory_percent >= self.alert_thresholds['memory_critical']:
            await self._create_alert(
                AlertLevel.CRITICAL,
                f"Critical memory usage: {metrics.memory_percent:.1f}%",
                None
            )
        elif metrics.memory_percent >= self.alert_thresholds['memory_warning']:
            await self._create_alert(
                AlertLevel.WARNING,
                f"High memory usage: {metrics.memory_percent:.1f}%",
                None
            )
        
        # Disk alerts
        if metrics.disk_percent >= self.alert_thresholds['disk_critical']:
            await self._create_alert(
                AlertLevel.CRITICAL,
                f"Critical disk usage: {metrics.disk_percent:.1f}%",
                None
            )
        elif metrics.disk_percent >= self.alert_thresholds['disk_warning']:
            await self._create_alert(
                AlertLevel.WARNING,
                f"High disk usage: {metrics.disk_percent:.1f}%",
                None
            )
    
    async def _check_service_alerts(self, health: ServiceHealth):
        """Check service health for alert conditions"""
        if health.status == "unhealthy":
            await self._create_alert(
                AlertLevel.ERROR,
                f"Service {health.service_name} is unhealthy",
                health.service_id
            )
        elif health.response_time >= self.alert_thresholds['response_time_critical']:
            await self._create_alert(
                AlertLevel.CRITICAL,
                f"Service {health.service_name} critical response time: {health.response_time:.2f}s",
                health.service_id
            )
        elif health.response_time >= self.alert_thresholds['response_time_warning']:
            await self._create_alert(
                AlertLevel.WARNING,
                f"Service {health.service_name} slow response time: {health.response_time:.2f}s",
                health.service_id
            )
    
    async def _create_alert(self, level: AlertLevel, message: str, service_id: Optional[str]):
        """Create a new alert"""
        # Check if similar alert already exists and is not acknowledged
        for existing_alert in self.alerts:
            if (existing_alert.level == level and 
                existing_alert.message == message and 
                existing_alert.service_id == service_id and 
                not existing_alert.acknowledged):
                return  # Don't create duplicate alerts
        
        self.alert_counter += 1
        alert = Alert(
            id=f"alert_{self.alert_counter}",
            level=level,
            message=message,
            service_id=service_id,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        logger.warning(f"Alert created: {level.value} - {message}")
        
        # Broadcast alert to connected clients
        await self._broadcast_update("new_alert", asdict(alert))
    
    def _get_dashboard_html(self) -> str:
        """Generate HTML for the health monitoring dashboard"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OrbitScope Health Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .header h1 { margin: 0; font-size: 2.5em; }
        .header p { margin: 5px 0 0 0; opacity: 0.9; }
        .dashboard { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
        .card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metric { text-align: center; }
        .metric-value { font-size: 3em; font-weight: bold; color: #333; }
        .metric-label { font-size: 1.2em; color: #666; margin-top: 10px; }
        .status-healthy { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-error { color: #e74c3c; }
        .status-critical { color: #c0392b; }
        .services-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
        .service-card { padding: 15px; border-radius: 8px; background: #f8f9fa; border-left: 4px solid #ddd; }
        .service-healthy { border-left-color: #27ae60; }
        .service-unhealthy { border-left-color: #e74c3c; }
        .alert { padding: 10px; margin: 5px 0; border-radius: 5px; }
        .alert-info { background: #d1ecf1; border: 1px solid #bee5eb; }
        .alert-warning { background: #fff3cd; border: 1px solid #ffeaa7; }
        .alert-error { background: #f8d7da; border: 1px solid #f5c6cb; }
        .alert-critical { background: #d1ecf1; border: 1px solid #c3e6cb; background: #f8d7da; }
        .chart-container { position: relative; height: 300px; margin: 20px 0; }
        .live-indicator { display: inline-block; width: 10px; height: 10px; background: #27ae60; border-radius: 50%; margin-right: 10px; animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    </style>
</head>
<body>
    <div class="header">
        <h1><span class="live-indicator"></span>OrbitScope Health Monitor</h1>
        <p>Real-time monitoring for OrbitScope ML Intelligence Suite - Organized and Optimized</p>
    </div>
    
    <div class="dashboard">
        <div class="card">
            <div class="metric">
                <div class="metric-value status-healthy" id="cpu-usage">--</div>
                <div class="metric-label">CPU Usage %</div>
            </div>
        </div>
        
        <div class="card">
            <div class="metric">
                <div class="metric-value status-healthy" id="memory-usage">--</div>
                <div class="metric-label">Memory Usage %</div>
            </div>
        </div>
        
        <div class="card">
            <div class="metric">
                <div class="metric-value status-healthy" id="services-status">--</div>
                <div class="metric-label">Services Online</div>
            </div>
        </div>
    </div>
    
    <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-top: 20px;">
        <div class="card">
            <h3>System Performance</h3>
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h3>Recent Alerts</h3>
            <div id="alerts-container">
                <p>No recent alerts</p>
            </div>
        </div>
    </div>
    
    <div class="card" style="margin-top: 20px;">
        <h3>AI Services Status</h3>
        <div class="services-grid" id="services-container">
            <p>Loading services...</p>
        </div>
    </div>

    <script>
        // WebSocket connection for real-time updates
        const ws = new WebSocket('ws://localhost:8015/ws');
        
        // Chart setup
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'CPU %',
                    data: [],
                    borderColor: '#3498db',
                    tension: 0.4
                }, {
                    label: 'Memory %',
                    data: [],
                    borderColor: '#e74c3c',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });

        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            
            if (message.type === 'initial_data' || message.type === 'system_update') {
                updateDashboard(message.data);
            } else if (message.type === 'new_alert') {
                addAlert(message.data);
            }
        };

        function updateDashboard(data) {
            // Update system metrics
            if (data.system_metrics) {
                const cpu = Math.round(data.system_metrics.cpu_percent);
                const memory = Math.round(data.system_metrics.memory_percent);
                
                document.getElementById('cpu-usage').textContent = cpu;
                document.getElementById('memory-usage').textContent = memory;
                
                // Update colors based on thresholds
                updateMetricColor('cpu-usage', cpu, 80, 90);
                updateMetricColor('memory-usage', memory, 85, 95);
                
                // Update chart
                const now = new Date().toLocaleTimeString();
                performanceChart.data.labels.push(now);
                performanceChart.data.datasets[0].data.push(cpu);
                performanceChart.data.datasets[1].data.push(memory);
                
                // Keep only last 20 points
                if (performanceChart.data.labels.length > 20) {
                    performanceChart.data.labels.shift();
                    performanceChart.data.datasets[0].data.shift();
                    performanceChart.data.datasets[1].data.shift();
                }
                
                performanceChart.update('none');
            }
            
            // Update services
            if (data.services) {
                updateServices(data.services);
            }
            
            // Update alerts
            if (data.alerts) {
                updateAlerts(data.alerts);
            }
        }

        function updateMetricColor(elementId, value, warningThreshold, criticalThreshold) {
            const element = document.getElementById(elementId);
            element.className = 'metric-value ';
            
            if (value >= criticalThreshold) {
                element.className += 'status-critical';
            } else if (value >= warningThreshold) {
                element.className += 'status-warning';
            } else {
                element.className += 'status-healthy';
            }
        }

        function updateServices(services) {
            const container = document.getElementById('services-container');
            const healthyCount = services.filter(s => s.status === 'healthy').length;
            
            document.getElementById('services-status').textContent = `${healthyCount}/${services.length}`;
            
            container.innerHTML = services.map(service => `
                <div class="service-card ${service.status === 'healthy' ? 'service-healthy' : 'service-unhealthy'}">
                    <strong>${service.service_name}</strong><br>
                    <small>Status: ${service.status}</small><br>
                    <small>Response: ${(service.response_time * 1000).toFixed(0)}ms</small><br>
                    <small>Uptime: ${service.uptime_percent.toFixed(1)}%</small>
                </div>
            `).join('');
        }

        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');
            const recentAlerts = alerts.slice(-5).reverse();
            
            if (recentAlerts.length === 0) {
                container.innerHTML = '<p>No recent alerts</p>';
                return;
            }
            
            container.innerHTML = recentAlerts.map(alert => `
                <div class="alert alert-${alert.level}">
                    <strong>${alert.level.toUpperCase()}</strong> - ${alert.message}<br>
                    <small>${new Date(alert.timestamp).toLocaleString()}</small>
                </div>
            `).join('');
        }

        // Periodic refresh fallback
        setInterval(async () => {
            try {
                const response = await fetch('/api/health/overview');
                const data = await response.json();
                // Update basic metrics if WebSocket is not working
            } catch (e) {
                console.log('Fallback refresh failed:', e);
            }
        }, 10000);
    </script>
</body>
</html>
        """
    
    async def start_monitoring(self):
        """Start the monitoring loop"""
        
        async def monitoring_loop():
            while True:
                try:
                    # Collect system metrics
                    metrics = await self._collect_system_metrics()
                    
                    # Check all services
                    services = await self._get_all_service_health()
                    
                    # Broadcast updates to connected clients
                    await self._broadcast_update("system_update", {
                        "system_metrics": asdict(metrics),
                        "services": [asdict(s) for s in services],
                        "alerts": [asdict(a) for a in self.alerts[-10:]]  # Last 10 alerts
                    })
                    
                    # Wait before next monitoring cycle
                    await asyncio.sleep(self.monitor_interval)
                    
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                    await asyncio.sleep(5)  # Shorter wait on error
        
        # Start monitoring task
        asyncio.create_task(monitoring_loop())
        logger.info(f"Health monitoring started - checking every {self.monitor_interval} seconds")

# Create the health monitor instance
health_monitor = HealthMonitor()
app = health_monitor.app

@app.on_event("startup")
async def startup_event():
    """Start monitoring on app startup"""
    await health_monitor.start_monitoring()
    logger.info("OrbitScope Health Monitor started successfully")

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("OrbitScope Health Monitor shutting down")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting OrbitScope Health Monitoring Dashboard")
    logger.info("Operational Motto: Organized and Optimized")
    
    uvicorn.run(
        "health_dashboard:app",
        host="0.0.0.0",
        port=8015,
        reload=False,
        log_level="info"
    )