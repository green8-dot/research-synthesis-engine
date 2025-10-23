#!/usr/bin/env python3
"""
OrbitScope ML Intelligence Suite - Unified API Gateway
Centralized entry point for all AI services with load balancing and monitoring

ORGANIZED AND OPTIMIZED - Operational Motto
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field
import httpx
from enum import Enum
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPED = "stopped"
    UNKNOWN = "unknown"

@dataclass
class ServiceEndpoint:
    """Represents a registered AI service endpoint"""
    name: str
    url: str
    health_check_url: str
    category: str  # phase_1, phase_2, phase_3, support
    priority: int = 1  # 1=highest, 3=lowest
    max_concurrent: int = 10
    timeout: float = 30.0
    retry_count: int = 3
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    response_times: List[float] = field(default_factory=list)
    error_count: int = 0

class UnifiedAPIGateway:
    """Unified API Gateway for OrbitScope ML Intelligence Suite"""
    
    def __init__(self):
        self.app = FastAPI(
            title="OrbitScope ML Intelligence Suite - Unified API Gateway",
            description="Centralized gateway for all AI services",
            version="3.1.0"
        )
        
        # Service registry
        self.services: Dict[str, ServiceEndpoint] = {}
        self.load_balancer_state: Dict[str, int] = {}  # Round-robin state
        
        # Performance metrics
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = datetime.now(timezone.utc)
        
        # Configure middleware
        self._setup_middleware()
        
        # Register routes
        self._register_routes()
        
        # Register AI services
        self._register_ai_services()
        
        logger.info("Unified API Gateway initialized")
    
    def _setup_middleware(self):
        """Configure FastAPI middleware"""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Compression middleware
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                
                logger.info(
                    f"{request.method} {request.url.path} - "
                    f"Status: {response.status_code} - "
                    f"Time: {process_time:.4f}s"
                )
                
                self.total_requests += 1
                response.headers["X-Process-Time"] = str(process_time)
                return response
                
            except Exception as e:
                self.total_errors += 1
                logger.error(f"Request failed: {e}")
                raise
    
    def _register_routes(self):
        """Register API gateway routes"""
        
        @self.app.get("/", tags=["Gateway"])
        async def root():
            """Root endpoint with gateway information"""
            return {
                "service": "OrbitScope ML Intelligence Suite",
                "component": "Unified API Gateway",
                "version": "3.1.0",
                "motto": "Organized and Optimized",
                "status": "operational",
                "registered_services": len(self.services),
                "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds()
            }
        
        @self.app.get("/health", tags=["Gateway"])
        async def health_check():
            """Comprehensive health check of gateway and services"""
            healthy_services = sum(1 for s in self.services.values() if s.status == ServiceStatus.HEALTHY)
            total_services = len(self.services)
            
            gateway_health = {
                "status": "healthy" if healthy_services > 0 else "degraded",
                "gateway_uptime": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
                "registered_services": total_services,
                "healthy_services": healthy_services,
                "unhealthy_services": total_services - healthy_services,
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate": (self.total_errors / max(self.total_requests, 1)) * 100,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return gateway_health
        
        @self.app.get("/services", tags=["Gateway"])
        async def list_services():
            """List all registered services with their status"""
            service_list = []
            
            for service_id, service in self.services.items():
                avg_response_time = (
                    sum(service.response_times[-10:]) / len(service.response_times[-10:])
                    if service.response_times else 0
                )
                
                service_list.append({
                    "id": service_id,
                    "name": service.name,
                    "category": service.category,
                    "status": service.status.value,
                    "url": service.url,
                    "priority": service.priority,
                    "last_health_check": service.last_health_check.isoformat() if service.last_health_check else None,
                    "average_response_time": round(avg_response_time, 3),
                    "error_count": service.error_count
                })
            
            return {"services": service_list}
        
        @self.app.post("/services/{service_id}/health", tags=["Gateway"])
        async def check_service_health(service_id: str):
            """Manually trigger health check for a specific service"""
            if service_id not in self.services:
                raise HTTPException(status_code=404, detail="Service not found")
            
            health_result = await self._health_check_service(service_id)
            return {
                "service_id": service_id,
                "health_check_result": health_result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @self.app.get("/ai/{service_category}/{endpoint:path}", tags=["AI Services"])
        async def proxy_ai_request(
            service_category: str, 
            endpoint: str, 
            request: Request
        ):
            """Proxy requests to AI services with load balancing"""
            return await self._proxy_request(service_category, endpoint, request, "GET")
        
        @self.app.post("/ai/{service_category}/{endpoint:path}", tags=["AI Services"])
        async def proxy_ai_post(
            service_category: str, 
            endpoint: str, 
            request: Request
        ):
            """Proxy POST requests to AI services"""
            return await self._proxy_request(service_category, endpoint, request, "POST")
        
        @self.app.get("/metrics", tags=["Monitoring"])
        async def get_metrics():
            """Get detailed performance metrics"""
            metrics = {
                "gateway": {
                    "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
                    "total_requests": self.total_requests,
                    "total_errors": self.total_errors,
                    "error_rate_percent": (self.total_errors / max(self.total_requests, 1)) * 100,
                    "requests_per_second": self.total_requests / max((datetime.now(timezone.utc) - self.start_time).total_seconds(), 1)
                },
                "services": {}
            }
            
            for service_id, service in self.services.items():
                avg_response_time = (
                    sum(service.response_times[-100:]) / len(service.response_times[-100:])
                    if service.response_times else 0
                )
                
                metrics["services"][service_id] = {
                    "status": service.status.value,
                    "average_response_time_ms": round(avg_response_time * 1000, 2),
                    "error_count": service.error_count,
                    "last_health_check": service.last_health_check.isoformat() if service.last_health_check else None
                }
            
            return metrics
    
    def _register_ai_services(self):
        """Register all AI services in the gateway"""
        
        # Phase 1 Services - Core AI
        phase_1_services = [
            ("semantic_search", "Semantic Search Engine", "http://localhost:8001"),
            ("sentiment_analysis", "Sentiment Analyzer", "http://localhost:8002")
        ]
        
        for service_id, name, base_url in phase_1_services:
            self.services[service_id] = ServiceEndpoint(
                name=name,
                url=base_url,
                health_check_url=f"{base_url}/health",
                category="phase_1",
                priority=1,
                max_concurrent=15
            )
        
        # Phase 2 Services - Advanced Intelligence
        phase_2_services = [
            ("advanced_nlp", "Advanced NLP Processor", "http://localhost:8003"),
            ("system_health", "System Health Predictor", "http://localhost:8004"),
            ("resource_manager", "Intelligent Resource Manager", "http://localhost:8005"),
            ("knowledge_graph", "Intelligent Knowledge Graph", "http://localhost:8006")
        ]
        
        for service_id, name, base_url in phase_2_services:
            self.services[service_id] = ServiceEndpoint(
                name=name,
                url=base_url,
                health_check_url=f"{base_url}/health",
                category="phase_2",
                priority=1,
                max_concurrent=12
            )
        
        # Phase 3 Services - Advanced Analytics
        phase_3_services = [
            ("web_scraper", "Intelligent Web Scraper", "http://localhost:8007"),
            ("market_intelligence", "Predictive Market Intelligence", "http://localhost:8008"),
            ("financial_analysis", "Advanced Financial Analysis", "http://localhost:8009"),
            ("secure_plaid", "Secure Plaid Service", "http://localhost:8010")
        ]
        
        for service_id, name, base_url in phase_3_services:
            self.services[service_id] = ServiceEndpoint(
                name=name,
                url=base_url,
                health_check_url=f"{base_url}/health",
                category="phase_3",
                priority=1,
                max_concurrent=10
            )
        
        # Support Services
        support_services = [
            ("logic_manager", "Logic Manager", "http://localhost:8011"),
            ("config_manager", "Configuration Manager", "http://localhost:8012"),
            ("audit_service", "Audit Service", "http://localhost:8013"),
            ("resource_monitor", "System Resource Monitor", "http://localhost:8014")
        ]
        
        for service_id, name, base_url in support_services:
            self.services[service_id] = ServiceEndpoint(
                name=name,
                url=base_url,
                health_check_url=f"{base_url}/health",
                category="support",
                priority=2,
                max_concurrent=8
            )
        
        logger.info(f"Registered {len(self.services)} AI services in gateway")
    
    async def _proxy_request(self, service_category: str, endpoint: str, request: Request, method: str):
        """Proxy request to appropriate AI service with load balancing"""
        
        # Find available services in the category
        available_services = [
            (service_id, service) for service_id, service in self.services.items()
            if service.category == service_category and service.status == ServiceStatus.HEALTHY
        ]
        
        if not available_services:
            raise HTTPException(
                status_code=503, 
                detail=f"No healthy services available in category: {service_category}"
            )
        
        # Simple round-robin load balancing
        if service_category not in self.load_balancer_state:
            self.load_balancer_state[service_category] = 0
        
        selected_index = self.load_balancer_state[service_category] % len(available_services)
        self.load_balancer_state[service_category] += 1
        
        service_id, selected_service = available_services[selected_index]
        
        # Proxy the request
        target_url = f"{selected_service.url}/{endpoint}"
        
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient() as client:
                if method == "GET":
                    response = await client.get(
                        target_url, 
                        params=dict(request.query_params),
                        timeout=selected_service.timeout
                    )
                else:  # POST
                    body = await request.body()
                    response = await client.post(
                        target_url,
                        content=body,
                        headers=dict(request.headers),
                        timeout=selected_service.timeout
                    )
                
                response_time = time.time() - start_time
                selected_service.response_times.append(response_time)
                
                # Keep only last 100 response times
                if len(selected_service.response_times) > 100:
                    selected_service.response_times = selected_service.response_times[-100:]
                
                return JSONResponse(
                    content=response.json() if response.content else {},
                    status_code=response.status_code,
                    headers={"X-Proxied-Service": service_id, "X-Response-Time": str(response_time)}
                )
        
        except Exception as e:
            selected_service.error_count += 1
            logger.error(f"Error proxying to {service_id}: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Service {service_id} unavailable: {str(e)}"
            )
    
    async def _health_check_service(self, service_id: str) -> Dict[str, Any]:
        """Perform health check on a specific service"""
        if service_id not in self.services:
            return {"error": "Service not found"}
        
        service = self.services[service_id]
        
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(service.health_check_url, timeout=5.0)
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    service.status = ServiceStatus.HEALTHY
                    service.last_health_check = datetime.now(timezone.utc)
                    return {
                        "status": "healthy",
                        "response_time": response_time,
                        "details": response.json() if response.content else {}
                    }
                else:
                    service.status = ServiceStatus.UNHEALTHY
                    return {
                        "status": "unhealthy",
                        "status_code": response.status_code,
                        "response_time": response_time
                    }
        
        except Exception as e:
            service.status = ServiceStatus.UNHEALTHY
            service.error_count += 1
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def start_health_monitoring(self):
        """Start background task for continuous service health monitoring"""
        
        async def health_monitor():
            while True:
                logger.info("Starting health check cycle for all services")
                
                for service_id in self.services.keys():
                    try:
                        await self._health_check_service(service_id)
                    except Exception as e:
                        logger.error(f"Health check failed for {service_id}: {e}")
                
                # Wait 30 seconds before next health check cycle
                await asyncio.sleep(30)
        
        # Start the health monitoring task
        asyncio.create_task(health_monitor())
        logger.info("Health monitoring started - checking services every 30 seconds")

# Global gateway instance
gateway = UnifiedAPIGateway()
app = gateway.app

# Start health monitoring when the app starts
@app.on_event("startup")
async def startup_event():
    """Initialize gateway on startup"""
    await gateway.start_health_monitoring()
    logger.info("OrbitScope Unified API Gateway started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("OrbitScope Unified API Gateway shutting down")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting OrbitScope ML Intelligence Suite - Unified API Gateway")
    logger.info("Operational Motto: Organized and Optimized")
    
    uvicorn.run(
        "unified_gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )