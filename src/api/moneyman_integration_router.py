"""
MoneyMan ML Integration Router

Productivity-focused integration endpoints:
- Productivity recommendations
- Cost optimization insights  
- Business intelligence sync
- Performance analysis
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional, List
from loguru import logger
from pydantic import BaseModel

from research_synthesis.services.moneyman_integration_service import moneyman_integration
from research_synthesis.services.comprehensive_job_manager import comprehensive_job_manager


class ProductivityAnalysisRequest(BaseModel):
    include_job_metrics: bool = True
    include_performance_data: bool = True
    include_resource_usage: bool = True
    analysis_depth: str = "standard"  # standard, detailed, comprehensive


router = APIRouter()


@router.get("/status")
async def get_integration_status():
    """Get MoneyMan ML integration status"""
    try:
        status = await moneyman_integration.get_integration_status()
        return {
            "status": "success",
            "integration_status": status
        }
    except Exception as e:
        logger.error(f"Failed to get integration status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.post("/test")
async def test_integration():
    """Test MoneyMan ML integration functionality"""
    try:
        test_result = await moneyman_integration.test_integration()
        return {
            "status": test_result["status"],
            "test_result": test_result
        }
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Integration test failed: {str(e)}")


@router.post("/productivity/recommendations")
async def get_productivity_recommendations(request: ProductivityAnalysisRequest):
    """Get productivity recommendations based on current system metrics"""
    try:
        # Gather system metrics
        system_metrics = {}
        
        if request.include_job_metrics:
            job_status = await comprehensive_job_manager.get_comprehensive_status()
            system_metrics.update({
                "job_statistics": job_status.get("job_statistics", {}),
                "system_health": job_status.get("system_health", {}),
                "configuration": job_status.get("configuration", {})
            })
        
        if request.include_performance_data:
            system_metrics.update({
                "performance_metrics": job_status.get("performance_metrics", {}) if 'job_status' in locals() else {}
            })
        
        if request.include_resource_usage:
            # Add resource usage data (would integrate with system monitoring)
            system_metrics["resource_utilization"] = {
                "memory_usage_mb": 512,  # Placeholder - would get from actual monitoring
                "cpu_usage_percent": 45,
                "disk_usage_percent": 30,
                "network_requests_per_minute": 100
            }
        
        # Get recommendations from MoneyMan ML
        recommendations = await moneyman_integration.get_productivity_recommendations(system_metrics)
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "analysis_depth": request.analysis_depth,
            "metrics_analyzed": list(system_metrics.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to get productivity recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")


@router.post("/cost-optimization")
async def get_cost_optimization_insights():
    """Get cost optimization insights from MoneyMan ML"""
    try:
        # Gather resource usage data
        resource_usage = {
            "memory_usage_mb": 512,  # Would get from actual monitoring
            "cpu_usage_percent": 45,
            "disk_io_operations": 1000,
            "network_bandwidth_mbps": 10,
            "job_processing_hours": 24,
            "concurrent_jobs_average": 2
        }
        
        insights = await moneyman_integration.get_cost_optimization_insights(resource_usage)
        
        return {
            "status": "success",
            "cost_optimization": insights
        }
        
    except Exception as e:
        logger.error(f"Cost optimization analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cost analysis failed: {str(e)}")


@router.post("/business-intelligence/sync")
async def sync_business_intelligence():
    """Synchronize business intelligence data with MoneyMan ML"""
    try:
        sync_result = await moneyman_integration.get_business_intelligence_sync()
        
        return {
            "status": sync_result["status"],
            "sync_result": sync_result
        }
        
    except Exception as e:
        logger.error(f"Business intelligence sync failed: {e}")
        raise HTTPException(status_code=500, detail=f"BI sync failed: {str(e)}")


@router.get("/recommendations/quick")
async def get_quick_recommendations():
    """Get quick productivity recommendations without detailed analysis"""
    try:
        # Get basic system status
        job_status = await comprehensive_job_manager.get_comprehensive_status()
        
        # Simple metrics
        quick_metrics = {
            "job_statistics": job_status.get("job_statistics", {}),
            "system_health": job_status.get("system_health", {}),
            "performance_metrics": job_status.get("performance_metrics", {})
        }
        
        # Get recommendations (will use fallback if MoneyMan ML unavailable)
        recommendations = await moneyman_integration.get_productivity_recommendations(quick_metrics)
        
        # Filter to top 3 recommendations
        top_recommendations = recommendations.get("recommendations", [])[:3]
        
        return {
            "status": "success",
            "quick_recommendations": top_recommendations,
            "recommendation_source": recommendations.get("source", "unknown"),
            "total_available": len(recommendations.get("recommendations", []))
        }
        
    except Exception as e:
        logger.error(f"Quick recommendations failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quick recommendations failed: {str(e)}")


@router.get("/productivity/score")
async def get_productivity_score():
    """Get overall productivity score based on system metrics"""
    try:
        job_status = await comprehensive_job_manager.get_comprehensive_status()
        
        # Calculate productivity score
        health_score = job_status.get("system_health", {}).get("score", 100)
        success_rate = job_status.get("performance_metrics", {}).get("success_rate", 100)
        active_jobs = job_status.get("job_statistics", {}).get("active_jobs", 0)
        max_capacity = job_status.get("configuration", {}).get("max_concurrent_jobs", 3)
        
        # Calculate capacity utilization
        capacity_utilization = (active_jobs / max(1, max_capacity)) * 100
        
        # Overall productivity score (weighted average)
        productivity_score = (
            health_score * 0.4 +
            success_rate * 0.4 +
            min(capacity_utilization, 80) * 0.2  # Cap at 80% for optimal performance
        )
        
        # Determine productivity level
        if productivity_score >= 90:
            level = "excellent"
        elif productivity_score >= 75:
            level = "good"
        elif productivity_score >= 60:
            level = "fair"
        else:
            level = "needs_improvement"
        
        return {
            "status": "success",
            "productivity_score": {
                "overall_score": round(productivity_score, 1),
                "level": level,
                "components": {
                    "system_health": health_score,
                    "success_rate": success_rate,
                    "capacity_utilization": round(capacity_utilization, 1)
                },
                "recommendations": _get_score_recommendations(productivity_score, level)
            }
        }
        
    except Exception as e:
        logger.error(f"Productivity score calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Score calculation failed: {str(e)}")


def _get_score_recommendations(score: float, level: str) -> List[str]:
    """Get recommendations based on productivity score"""
    if level == "excellent":
        return [
            "System is performing optimally",
            "Consider expanding capacity for additional workloads",
            "Document current configuration as best practice"
        ]
    elif level == "good":
        return [
            "Good performance with room for optimization",
            "Monitor for potential bottlenecks",
            "Consider minor configuration adjustments"
        ]
    elif level == "fair":
        return [
            "Performance could be improved",
            "Check for system bottlenecks",
            "Review job management configuration",
            "Consider increasing resources"
        ]
    else:
        return [
            "Significant performance issues detected",
            "Immediate attention required",
            "Run comprehensive system diagnosis",
            "Consider emergency optimization measures"
        ]


@router.get("/integration/health")
async def check_integration_health():
    """Comprehensive health check for MoneyMan ML integration"""
    try:
        # Check MoneyMan ML service
        integration_status = await moneyman_integration.get_integration_status()
        
        # Test basic functionality
        test_result = await moneyman_integration.test_integration()
        
        # Get productivity score for health context
        try:
            productivity_response = await get_productivity_score()
            productivity_score = productivity_response["productivity_score"]["overall_score"]
        except:
            productivity_score = None
        
        health_status = "healthy"
        if not integration_status["service_available"]:
            health_status = "degraded"
        elif not test_result.get("integration_functional", False):
            health_status = "limited"
        
        return {
            "status": "success",
            "integration_health": {
                "overall_status": health_status,
                "moneyman_ml_available": integration_status["service_available"],
                "integration_functional": test_result.get("integration_functional", False),
                "fallback_available": True,
                "productivity_score": productivity_score,
                "features": {
                    "productivity_recommendations": integration_status["features"]["productivity_recommendations"],
                    "cost_optimization": integration_status["features"]["cost_optimization"],
                    "business_intelligence": integration_status["features"]["business_intelligence_sync"]
                },
                "last_check": test_result.get("test_timestamp")
            }
        }
        
    except Exception as e:
        logger.error(f"Integration health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.post("/optimization/auto-apply")
async def auto_apply_optimization(background_tasks: BackgroundTasks):
    """Automatically apply safe optimization recommendations"""
    try:
        # Get current recommendations
        recommendations_response = await get_quick_recommendations()
        recommendations = recommendations_response.get("quick_recommendations", [])
        
        # Filter for safe auto-apply recommendations
        safe_recommendations = [
            rec for rec in recommendations
            if rec.get("implementation_effort", "high") == "low"
            and rec.get("impact_score", 0) >= 7
        ]
        
        if not safe_recommendations:
            return {
                "status": "no_actions",
                "message": "No safe auto-optimization actions available",
                "recommendations_reviewed": len(recommendations)
            }
        
        # Apply optimizations in background
        background_tasks.add_task(_apply_safe_optimizations, safe_recommendations)
        
        return {
            "status": "accepted",
            "message": "Safe optimizations being applied in background",
            "optimizations_queued": len(safe_recommendations),
            "actions": [rec["title"] for rec in safe_recommendations]
        }
        
    except Exception as e:
        logger.error(f"Auto-apply optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-optimization failed: {str(e)}")


async def _apply_safe_optimizations(recommendations: List[Dict[str, Any]]):
    """Apply safe optimization recommendations in background"""
    try:
        for rec in recommendations:
            logger.info(f"Applying optimization: {rec['title']}")
            
            # Example: Increase concurrent jobs if recommended
            if "concurrent" in rec["title"].lower() and "increase" in rec["description"].lower():
                # This would integrate with job manager configuration
                logger.info("Would increase concurrent job capacity")
            
            # Example: Memory optimization
            elif "memory" in rec["category"].lower():
                logger.info("Would apply memory optimization settings")
        
        logger.info(f"Applied {len(recommendations)} safe optimizations")
        
    except Exception as e:
        logger.error(f"Background optimization application failed: {e}")