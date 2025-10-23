"""
MoneyMan ML Integration Service

Productivity-focused integration with moneyman_ml system:
- Productivity enhancement recommendations
- Performance optimization insights  
- Resource utilization analysis
- Cost-efficiency suggestions
- Business intelligence integration
"""

import asyncio
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
import aiohttp
from pathlib import Path


class MoneyManIntegrationService:
    """Service for integrating with MoneyMan ML for productivity enhancements"""
    
    def __init__(self):
        self.moneyman_base_url = "http://localhost:8000"  # MoneyMan ML default port
        self.integration_enabled = False
        self.last_health_check = None
        self.productivity_cache = {}
        
        # Initialize connection when first accessed
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure the service is initialized"""
        if not self._initialized:
            await self._initialize_connection()
            self._initialized = True
    
    async def _initialize_connection(self):
        """Initialize connection to MoneyMan ML"""
        try:
            await self._check_moneyman_health()
            if self.integration_enabled:
                logger.info("MoneyMan ML integration enabled")
            else:
                logger.info("MoneyMan ML integration disabled - service not available")
        except Exception as e:
            logger.warning(f"MoneyMan ML integration initialization failed: {e}")
    
    async def _check_moneyman_health(self) -> bool:
        """Check if MoneyMan ML service is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.moneyman_base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        self.integration_enabled = True
                        self.last_health_check = datetime.utcnow()
                        return True
        except Exception as e:
            logger.debug(f"MoneyMan ML health check failed: {e}")
        
        self.integration_enabled = False
        return False
    
    async def get_productivity_recommendations(self, system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get productivity recommendations based on system metrics"""
        await self._ensure_initialized()
        
        if not self.integration_enabled:
            await self._check_moneyman_health()
            if not self.integration_enabled:
                return self._get_fallback_recommendations(system_metrics)
        
        try:
            # Prepare data for MoneyMan ML analysis
            analysis_request = {
                "analysis_type": "productivity_optimization",
                "system_data": {
                    "job_statistics": system_metrics.get("job_statistics", {}),
                    "performance_metrics": system_metrics.get("performance_metrics", {}),
                    "system_health": system_metrics.get("system_health", {}),
                    "resource_utilization": system_metrics.get("resource_utilization", {})
                },
                "timestamp": datetime.utcnow().isoformat(),
                "source_system": "research_synthesis_engine"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.moneyman_base_url}/api/v1/analysis/productivity",
                    json=analysis_request,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._process_productivity_response(result)
                    else:
                        logger.warning(f"MoneyMan ML productivity analysis failed: {response.status}")
                        return self._get_fallback_recommendations(system_metrics)
        
        except Exception as e:
            logger.error(f"Failed to get productivity recommendations: {e}")
            return self._get_fallback_recommendations(system_metrics)
    
    def _process_productivity_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process MoneyMan ML productivity response"""
        recommendations = response.get("recommendations", [])
        
        # Filter for productivity-focused recommendations (skip personality)
        productivity_recommendations = []
        for rec in recommendations:
            if any(keyword in rec.get("category", "").lower() for keyword in 
                   ["productivity", "efficiency", "performance", "optimization", "resource", "cost"]):
                productivity_recommendations.append({
                    "category": rec.get("category", "productivity"),
                    "title": rec.get("title", "Productivity Enhancement"),
                    "description": rec.get("description", ""),
                    "impact_score": rec.get("impact_score", 0),
                    "implementation_effort": rec.get("implementation_effort", "medium"),
                    "expected_improvement": rec.get("expected_improvement", "Unknown")
                })
        
        return {
            "status": "success",
            "source": "moneyman_ml",
            "recommendations": productivity_recommendations,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "total_recommendations": len(productivity_recommendations)
        }
    
    def _get_fallback_recommendations(self, system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback recommendations when MoneyMan ML is unavailable"""
        recommendations = []
        
        # Analyze system metrics for basic recommendations
        job_stats = system_metrics.get("job_statistics", {})
        performance = system_metrics.get("performance_metrics", {})
        health = system_metrics.get("system_health", {})
        
        # Job efficiency recommendations
        if job_stats.get("active_jobs", 0) >= job_stats.get("max_concurrent", 3):
            recommendations.append({
                "category": "resource_optimization",
                "title": "Increase Concurrent Job Capacity",
                "description": "System is at maximum job capacity. Consider increasing max_concurrent_jobs to improve throughput.",
                "impact_score": 8,
                "implementation_effort": "low",
                "expected_improvement": "20-30% throughput increase"
            })
        
        # Performance optimization
        avg_duration = performance.get("average_duration_minutes", 0)
        if avg_duration > 15:
            recommendations.append({
                "category": "performance_optimization",
                "title": "Optimize Job Processing Time",
                "description": f"Average job duration is {avg_duration:.1f} minutes. Investigate bottlenecks in data processing pipeline.",
                "impact_score": 7,
                "implementation_effort": "medium",
                "expected_improvement": "30-50% faster processing"
            })
        
        # Health-based recommendations
        health_score = health.get("score", 100)
        if health_score < 80:
            recommendations.append({
                "category": "system_reliability",
                "title": "Improve System Health",
                "description": f"System health score is {health_score}/100. Address underlying issues to improve reliability.",
                "impact_score": 9,
                "implementation_effort": "high",
                "expected_improvement": "Reduced downtime and errors"
            })
        
        # Resource utilization
        success_rate = performance.get("success_rate", 100)
        if success_rate < 90:
            recommendations.append({
                "category": "efficiency_improvement",
                "title": "Reduce Job Failure Rate",
                "description": f"Job success rate is {success_rate:.1f}%. Implement better error handling and retry mechanisms.",
                "impact_score": 8,
                "implementation_effort": "medium",
                "expected_improvement": "Improved success rate and resource utilization"
            })
        
        return {
            "status": "fallback",
            "source": "internal_analysis",
            "recommendations": recommendations,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "total_recommendations": len(recommendations)
        }
    
    async def get_cost_optimization_insights(self, resource_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Get cost optimization insights from MoneyMan ML"""
        if not self.integration_enabled:
            return self._get_fallback_cost_insights(resource_usage)
        
        try:
            cost_request = {
                "analysis_type": "cost_optimization",
                "resource_data": resource_usage,
                "system_type": "research_synthesis_engine",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.moneyman_base_url}/api/v1/analysis/cost-optimization",
                    json=cost_request,
                    timeout=aiohttp.ClientTimeout(total=20)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._process_cost_response(result)
                    else:
                        return self._get_fallback_cost_insights(resource_usage)
        
        except Exception as e:
            logger.error(f"Cost optimization analysis failed: {e}")
            return self._get_fallback_cost_insights(resource_usage)
    
    def _process_cost_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process MoneyMan ML cost optimization response"""
        return {
            "status": "success",
            "source": "moneyman_ml",
            "cost_insights": response.get("insights", []),
            "potential_savings": response.get("potential_savings", {}),
            "optimization_actions": response.get("optimization_actions", []),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_fallback_cost_insights(self, resource_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback cost optimization insights"""
        insights = []
        
        # Memory usage insights
        memory_usage = resource_usage.get("memory_usage_mb", 0)
        if memory_usage > 1000:  # 1GB
            insights.append({
                "category": "memory_optimization",
                "insight": f"High memory usage detected: {memory_usage}MB",
                "recommendation": "Consider implementing memory caching strategies",
                "potential_saving": "10-20% performance improvement"
            })
        
        # CPU optimization
        cpu_usage = resource_usage.get("cpu_usage_percent", 0)
        if cpu_usage < 30:
            insights.append({
                "category": "resource_utilization", 
                "insight": "Low CPU utilization detected",
                "recommendation": "System can handle additional concurrent jobs",
                "potential_saving": "Better resource utilization"
            })
        
        return {
            "status": "fallback",
            "source": "internal_analysis",
            "cost_insights": insights,
            "potential_savings": {"estimated_efficiency_gain": "5-15%"},
            "optimization_actions": ["Monitor resource usage", "Optimize job concurrency"],
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_business_intelligence_sync(self) -> Dict[str, Any]:
        """Sync business intelligence data with MoneyMan ML"""
        if not self.integration_enabled:
            return {"status": "disabled", "message": "MoneyMan ML integration not available"}
        
        try:
            # Get current system status for sync
            sync_data = {
                "system_id": "research_synthesis_engine",
                "sync_type": "business_intelligence",
                "data": {
                    "active_components": await self._count_system_components(),
                    "processing_capacity": await self._get_processing_capacity(),
                    "operational_metrics": await self._get_operational_metrics()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.moneyman_base_url}/api/v1/sync/business-intelligence",
                    json=sync_data,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "status": "success",
                            "sync_result": result,
                            "synced_at": datetime.utcnow().isoformat()
                        }
                    else:
                        return {
                            "status": "error", 
                            "message": f"Sync failed with status {response.status}"
                        }
        
        except Exception as e:
            logger.error(f"Business intelligence sync failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _count_system_components(self) -> int:
        """Count active system components"""
        try:
            api_dir = Path("D:/orbitscope_ml/research_synthesis/api")
            return len(list(api_dir.glob("*_router.py"))) if api_dir.exists() else 0
        except:
            return 0
    
    async def _get_processing_capacity(self) -> Dict[str, Any]:
        """Get current processing capacity metrics"""
        return {
            "max_concurrent_jobs": 3,
            "current_utilization": 0,  # Would be calculated from actual job manager
            "average_processing_time_minutes": 5  # Placeholder
        }
    
    async def _get_operational_metrics(self) -> Dict[str, Any]:
        """Get operational metrics for business intelligence"""
        return {
            "uptime_hours": 24,  # Placeholder
            "total_jobs_processed": 50,  # Placeholder
            "success_rate_percent": 95,  # Placeholder
            "system_health_score": 100
        }
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get MoneyMan ML integration status"""
        await self._ensure_initialized()
        
        # Check current health
        is_healthy = await self._check_moneyman_health()
        
        return {
            "integration_enabled": self.integration_enabled,
            "service_available": is_healthy,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "moneyman_url": self.moneyman_base_url,
            "features": {
                "productivity_recommendations": self.integration_enabled,
                "cost_optimization": self.integration_enabled,
                "business_intelligence_sync": self.integration_enabled,
                "fallback_analysis": True
            }
        }
    
    async def test_integration(self) -> Dict[str, Any]:
        """Test the integration connection"""
        try:
            # Test health endpoint
            health_result = await self._check_moneyman_health()
            
            if health_result:
                # Test a simple analysis request
                test_metrics = {
                    "job_statistics": {"active_jobs": 1, "total_jobs": 10},
                    "performance_metrics": {"success_rate": 95},
                    "system_health": {"score": 100}
                }
                
                recommendations = await self.get_productivity_recommendations(test_metrics)
                
                return {
                    "status": "success",
                    "health_check": True,
                    "analysis_test": recommendations["status"] == "success",
                    "integration_functional": True,
                    "test_timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "status": "partial",
                    "health_check": False,
                    "analysis_test": False,
                    "integration_functional": False,
                    "fallback_available": True,
                    "test_timestamp": datetime.utcnow().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "integration_functional": False,
                "fallback_available": True,
                "test_timestamp": datetime.utcnow().isoformat()
            }


# Global integration service instance
moneyman_integration = MoneyManIntegrationService()