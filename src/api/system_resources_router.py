"""
System Resources API Router
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional, List
from loguru import logger

from research_synthesis.services.system_resource_monitor import resource_monitor

router = APIRouter()

@router.get("/")
async def get_resources_status():
    """Get system resources monitoring status"""
    return {
        "service": "System Resources Monitor",
        "status": "active" if resource_monitor.monitoring_active else "inactive",
        "description": "Real-time system resource monitoring (CPU, memory, disk)",
        "endpoints": [
            "/current - Get current resource usage",
            "/summary - Get resource summary with trends",
            "/history - Get historical resource data",
            "/start - Start resource monitoring",
            "/stop - Stop resource monitoring"
        ],
        "monitoring_info": {
            "history_minutes": resource_monitor.history_minutes,
            "snapshots_available": len(resource_monitor.snapshots),
            "monitoring_active": resource_monitor.monitoring_active
        }
    }

@router.get("/current")
async def get_current_resources():
    """Get current system resource snapshot"""
    try:
        snapshot = await resource_monitor.get_current_snapshot()
        return {
            "status": "success",
            "timestamp": snapshot.timestamp.isoformat(),
            "resources": {
                "cpu": {
                    "percent": snapshot.cpu_percent,
                    "status": "healthy" if snapshot.cpu_percent < 50 else "warning" if snapshot.cpu_percent < 80 else "critical"
                },
                "memory": {
                    "percent": snapshot.memory_percent,
                    "used_gb": snapshot.memory_used_gb,
                    "total_gb": snapshot.memory_total_gb,
                    "status": "healthy" if snapshot.memory_percent < 70 else "warning" if snapshot.memory_percent < 90 else "critical"
                },
                "disk": {
                    "percent": snapshot.disk_percent,
                    "used_gb": snapshot.disk_used_gb,
                    "total_gb": snapshot.disk_total_gb,
                    "status": "healthy" if snapshot.disk_percent < 80 else "warning" if snapshot.disk_percent < 95 else "critical"
                },
                "processes": {
                    "count": snapshot.process_count,
                    "python_memory_mb": snapshot.python_memory_mb
                },
                "network": {
                    "bytes_sent": snapshot.network_bytes_sent,
                    "bytes_recv": snapshot.network_bytes_recv
                }
            }
        }
    except Exception as e:
        logger.error(f"Failed to get current resources: {e}")
        raise HTTPException(status_code=500, detail=f"Resource monitoring failed: {str(e)}")

@router.get("/summary")
async def get_resources_summary():
    """Get comprehensive resource summary with trends and health indicators"""
    try:
        summary = resource_monitor.get_resource_summary()
        return {
            "status": "success",
            "resource_summary": summary
        }
    except Exception as e:
        logger.error(f"Failed to get resource summary: {e}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

@router.get("/history")
async def get_resources_history(minutes: Optional[int] = None):
    """Get historical resource data"""
    try:
        history = resource_monitor.get_history_data(minutes)
        return {
            "status": "success",
            "data_points": len(history),
            "time_range_minutes": minutes or resource_monitor.history_minutes,
            "history": history
        }
    except Exception as e:
        logger.error(f"Failed to get resource history: {e}")
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")

@router.post("/start")
async def start_resource_monitoring(background_tasks: BackgroundTasks, interval_seconds: int = 5):
    """Start background resource monitoring"""
    try:
        if resource_monitor.monitoring_active:
            return {
                "status": "already_running",
                "message": "Resource monitoring is already active"
            }
            
        background_tasks.add_task(resource_monitor.start_monitoring, interval_seconds)
        return {
            "status": "success",
            "message": f"Resource monitoring started with {interval_seconds}s interval"
        }
    except Exception as e:
        logger.error(f"Failed to start resource monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@router.post("/stop")
async def stop_resource_monitoring():
    """Stop background resource monitoring"""
    try:
        await resource_monitor.stop_monitoring()
        return {
            "status": "success",
            "message": "Resource monitoring stopped"
        }
    except Exception as e:
        logger.error(f"Failed to stop resource monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

@router.get("/health")
async def get_system_health():
    """Get overall system health status"""
    try:
        if not resource_monitor.snapshots:
            # Get a fresh snapshot if no data available
            snapshot = await resource_monitor.get_current_snapshot()
        else:
            snapshot = resource_monitor.snapshots[-1]
            
        health_indicators = resource_monitor._get_health_indicators(snapshot)
        
        # Overall health logic
        critical_count = sum(1 for status in health_indicators.values() if status == "critical")
        warning_count = sum(1 for status in health_indicators.values() if status == "warning")
        
        if critical_count > 0:
            overall_status = "critical"
        elif warning_count > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"
            
        return {
            "status": "success",
            "overall_health": overall_status,
            "indicators": health_indicators,
            "summary": {
                "cpu_percent": snapshot.cpu_percent,
                "memory_percent": snapshot.memory_percent,
                "disk_percent": snapshot.disk_percent,
                "python_memory_mb": snapshot.python_memory_mb
            },
            "recommendations": _get_health_recommendations(health_indicators, snapshot)
        }
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

def _get_health_recommendations(indicators: Dict[str, str], snapshot) -> List[str]:
    """Generate health recommendations based on indicators"""
    recommendations = []
    
    if indicators.get("cpu") == "critical":
        recommendations.append("High CPU usage detected - consider closing unnecessary applications")
    elif indicators.get("cpu") == "warning":
        recommendations.append("CPU usage is elevated - monitor for performance impact")
        
    if indicators.get("memory") == "critical":
        recommendations.append(f"Memory usage is very high ({snapshot.memory_percent:.1f}%) - restart application or increase system RAM")
    elif indicators.get("memory") == "warning":
        recommendations.append("Memory usage is elevated - monitor for memory leaks")
        
    if indicators.get("disk") == "critical":
        recommendations.append("Disk space is critically low - free up space immediately")
    elif indicators.get("disk") == "warning":
        recommendations.append("Disk space is running low - consider cleanup or expansion")
        
    if indicators.get("python") == "critical":
        recommendations.append(f"Python process using {snapshot.python_memory_mb:.1f}MB - possible memory leak")
    elif indicators.get("python") == "warning":
        recommendations.append("Python process memory usage is elevated")
        
    if not recommendations:
        recommendations.append("System resources are healthy")
        
    return recommendations