"""
Comprehensive Job Manager API Router

Advanced job management endpoints with:
- Real-time monitoring
- Emergency cleanup
- Performance optimization
- System health tracking
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List
from loguru import logger

from research_synthesis.services.comprehensive_job_manager import comprehensive_job_manager

router = APIRouter()


@router.get("/status")
async def get_comprehensive_status():
    """Get comprehensive job management status"""
    try:
        status = await comprehensive_job_manager.get_comprehensive_status()
        return {
            "status": "success",
            "comprehensive_status": status
        }
    except Exception as e:
        logger.error(f"Failed to get comprehensive status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


@router.post("/cleanup/comprehensive")
async def comprehensive_cleanup():
    """Trigger comprehensive cleanup of all problematic jobs"""
    try:
        logger.info("Comprehensive cleanup triggered")
        result = await comprehensive_job_manager._comprehensive_cleanup()
        
        return {
            "status": "success",
            "cleanup_result": result,
            "summary": f"Cleaned {result['hanging_jobs_cleaned']} hanging, "
                      f"{result['failed_jobs_cleaned']} failed, "
                      f"{result['stuck_jobs_recovered']} stuck jobs"
        }
    except Exception as e:
        logger.error(f"Comprehensive cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.post("/cleanup/force-all")
async def force_cleanup_all():
    """EMERGENCY: Force cleanup of ALL jobs (cancels everything)"""
    try:
        logger.warning("EMERGENCY FORCE CLEANUP TRIGGERED - ALL JOBS WILL BE CANCELLED")
        result = await comprehensive_job_manager.force_cleanup_all()
        
        return {
            "status": result["status"],
            "warning": "ALL JOBS HAVE BEEN FORCEFULLY CANCELLED",
            "cleanup_result": result
        }
    except Exception as e:
        logger.error(f"Force cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Force cleanup failed: {str(e)}")


@router.get("/health")
async def get_system_health():
    """Get detailed system health information"""
    try:
        status = await comprehensive_job_manager.get_comprehensive_status()
        health = status.get("system_health", {})
        
        return {
            "status": "success",
            "health": health,
            "recommendations": _get_health_recommendations(health, status)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/performance")
async def get_performance_metrics():
    """Get detailed performance metrics"""
    try:
        status = await comprehensive_job_manager.get_comprehensive_status()
        performance = status.get("performance_metrics", {})
        
        return {
            "status": "success",
            "performance": performance,
            "analysis": _analyze_performance(performance)
        }
    except Exception as e:
        logger.error(f"Performance metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance retrieval failed: {str(e)}")


@router.post("/optimize")
async def optimize_performance():
    """Analyze and optimize job performance"""
    try:
        result = await comprehensive_job_manager.optimize_job_performance()
        return result
    except Exception as e:
        logger.error(f"Performance optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.get("/monitoring/real-time")
async def get_real_time_monitoring():
    """Get real-time job monitoring data"""
    try:
        status = await comprehensive_job_manager.get_comprehensive_status()
        
        return {
            "status": "success",
            "real_time_data": {
                "active_jobs": status["job_statistics"]["active_jobs"],
                "system_health_score": status["system_health"]["score"],
                "recent_activity": status["job_statistics"]["recent_activity"],
                "background_services": status["background_services"],
                "capacity_utilization": (
                    status["job_statistics"]["active_jobs"] / 
                    max(1, status["configuration"]["max_concurrent_jobs"])
                ) * 100
            },
            "timestamp": status.get("timestamp", "unknown")
        }
    except Exception as e:
        logger.error(f"Real-time monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Monitoring failed: {str(e)}")


@router.post("/configuration/update")
async def update_configuration(
    max_concurrent_jobs: int = None,
    job_timeout_minutes: int = None,
    cleanup_interval_seconds: int = None,
    max_retries: int = None
):
    """Update job manager configuration"""
    try:
        updates = {}
        
        if max_concurrent_jobs is not None:
            if 1 <= max_concurrent_jobs <= 10:
                comprehensive_job_manager.max_concurrent_jobs = max_concurrent_jobs
                updates["max_concurrent_jobs"] = max_concurrent_jobs
            else:
                raise HTTPException(status_code=400, detail="max_concurrent_jobs must be between 1 and 10")
        
        if job_timeout_minutes is not None:
            if 5 <= job_timeout_minutes <= 120:
                comprehensive_job_manager.job_timeout_minutes = job_timeout_minutes
                updates["job_timeout_minutes"] = job_timeout_minutes
            else:
                raise HTTPException(status_code=400, detail="job_timeout_minutes must be between 5 and 120")
        
        if cleanup_interval_seconds is not None:
            if 60 <= cleanup_interval_seconds <= 3600:
                comprehensive_job_manager.cleanup_interval_seconds = cleanup_interval_seconds
                updates["cleanup_interval_seconds"] = cleanup_interval_seconds
            else:
                raise HTTPException(status_code=400, detail="cleanup_interval_seconds must be between 60 and 3600")
        
        if max_retries is not None:
            if 0 <= max_retries <= 5:
                comprehensive_job_manager.max_retries = max_retries
                updates["max_retries"] = max_retries
            else:
                raise HTTPException(status_code=400, detail="max_retries must be between 0 and 5")
        
        logger.info(f"Configuration updated: {updates}")
        
        return {
            "status": "success",
            "updates_applied": updates,
            "current_configuration": {
                "max_concurrent_jobs": comprehensive_job_manager.max_concurrent_jobs,
                "job_timeout_minutes": comprehensive_job_manager.job_timeout_minutes,
                "cleanup_interval_seconds": comprehensive_job_manager.cleanup_interval_seconds,
                "max_retries": comprehensive_job_manager.max_retries
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")


@router.get("/jobs/detailed")
async def get_detailed_job_list():
    """Get detailed list of all jobs with enhanced information"""
    try:
        status = await comprehensive_job_manager.get_comprehensive_status()
        
        # This would include more detailed job information
        # For now, return the comprehensive status
        return {
            "status": "success",
            "job_details": status["job_statistics"],
            "active_jobs": list(comprehensive_job_manager.active_jobs),
            "system_capacity": {
                "current_load": status["job_statistics"]["active_jobs"],
                "max_capacity": status["configuration"]["max_concurrent_jobs"],
                "utilization_percent": (
                    status["job_statistics"]["active_jobs"] / 
                    max(1, status["configuration"]["max_concurrent_jobs"])
                ) * 100
            }
        }
    except Exception as e:
        logger.error(f"Detailed job list retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Job list retrieval failed: {str(e)}")


@router.post("/emergency/reset")
async def emergency_system_reset():
    """EMERGENCY: Reset entire job management system"""
    try:
        logger.critical("EMERGENCY SYSTEM RESET TRIGGERED")
        
        # Force cleanup all jobs
        cleanup_result = await comprehensive_job_manager.force_cleanup_all()
        
        # Reset internal state
        comprehensive_job_manager.active_jobs.clear()
        comprehensive_job_manager.job_metrics.clear()
        comprehensive_job_manager.job_queue.clear()
        comprehensive_job_manager.total_jobs_processed = 0
        comprehensive_job_manager.total_jobs_failed = 0
        comprehensive_job_manager.system_health_score = 100.0
        
        logger.info("Emergency system reset completed")
        
        return {
            "status": "success",
            "warning": "EMERGENCY RESET PERFORMED - ALL STATE CLEARED",
            "cleanup_result": cleanup_result,
            "system_state": "reset_to_default"
        }
    except Exception as e:
        logger.error(f"Emergency reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Emergency reset failed: {str(e)}")


def _get_health_recommendations(health: Dict[str, Any], status: Dict[str, Any]) -> List[str]:
    """Generate health recommendations based on current status"""
    recommendations = []
    
    health_score = health.get("score", 0)
    
    if health_score < 50:
        recommendations.append("CRITICAL: System health is severely degraded - immediate attention required")
    elif health_score < 80:
        recommendations.append("WARNING: System health is degraded - investigate issues")
    
    job_stats = status.get("job_statistics", {})
    performance = status.get("performance_metrics", {})
    
    # Check for stuck jobs
    if job_stats.get("active_jobs", 0) >= job_stats.get("max_concurrent", 3):
        recommendations.append("System at maximum capacity - consider cleanup or increasing limits")
    
    # Check failure rate
    success_rate = performance.get("success_rate", 100)
    if success_rate < 70:
        recommendations.append("Low success rate detected - investigate common failure causes")
    
    # Check average duration
    avg_duration = performance.get("average_duration_minutes", 0)
    if avg_duration > 30:
        recommendations.append("Jobs taking longer than expected - check for performance issues")
    
    if not recommendations:
        recommendations.append("System is operating normally")
    
    return recommendations


def _analyze_performance(performance: Dict[str, Any]) -> Dict[str, str]:
    """Analyze performance metrics and provide insights"""
    analysis = {}
    
    success_rate = performance.get("success_rate", 0)
    avg_duration = performance.get("average_duration_minutes", 0)
    
    # Success rate analysis
    if success_rate >= 95:
        analysis["success_rate"] = "Excellent"
    elif success_rate >= 85:
        analysis["success_rate"] = "Good"
    elif success_rate >= 70:
        analysis["success_rate"] = "Fair - needs improvement"
    else:
        analysis["success_rate"] = "Poor - immediate attention needed"
    
    # Duration analysis
    if avg_duration <= 5:
        analysis["performance"] = "Fast"
    elif avg_duration <= 15:
        analysis["performance"] = "Normal"
    elif avg_duration <= 30:
        analysis["performance"] = "Slow"
    else:
        analysis["performance"] = "Very slow - optimization needed"
    
    return analysis