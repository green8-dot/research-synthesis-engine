"""
Memory Management API Router
Provides endpoints for memory monitoring and management
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def memory_status():
    """Get current memory status"""
    try:
        from research_synthesis.utils.memory_manager import get_memory_stats
        return get_memory_stats()
    except Exception as e:
        logger.error(f"Error getting memory status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends")
async def memory_trends(minutes: int = 30):
    """Get memory usage trends for the specified time period"""
    try:
        from research_synthesis.utils.memory_manager import get_memory_trends
        return get_memory_trends(minutes)
    except Exception as e:
        logger.error(f"Error getting memory trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def force_memory_cleanup():
    """Force immediate memory cleanup and garbage collection"""
    try:
        from research_synthesis.utils.memory_manager import force_memory_cleanup
        result = force_memory_cleanup()
        return {
            "message": "Memory cleanup completed successfully",
            "details": result
        }
    except Exception as e:
        logger.error(f"Error during forced memory cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/detailed")
async def detailed_memory_info():
    """Get detailed memory information including system stats"""
    try:
        from research_synthesis.utils.memory_manager import memory_manager
        import psutil
        import gc
        
        # Get current stats
        current_stats = memory_manager.get_current_stats()
        
        # Get system information
        system_memory = psutil.virtual_memory()
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get garbage collection stats
        gc_stats = gc.get_stats()
        
        return {
            "current_stats": current_stats,
            "system_memory": {
                "total_gb": round(system_memory.total / (1024**3), 2),
                "available_gb": round(system_memory.available / (1024**3), 2),
                "used_percent": round(system_memory.percent, 1),
                "free_gb": round(system_memory.free / (1024**3), 2)
            },
            "process_memory": {
                "rss_mb": round(memory_info.rss / (1024**2), 2),
                "vms_mb": round(memory_info.vms / (1024**2), 2),
                "percent": round(process.memory_percent(), 2)
            },
            "garbage_collection": {
                "generations": len(gc_stats),
                "stats": gc_stats,
                "thresholds": gc.get_threshold(),
                "counts": gc.get_count()
            },
            "memory_manager": {
                "is_running": memory_manager.is_running,
                "cleanup_interval": memory_manager.cleanup_interval,
                "monitoring_interval": memory_manager.monitoring_interval,
                "history_size": len(memory_manager.memory_history)
            }
        }
    except Exception as e:
        logger.error(f"Error getting detailed memory info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start")
async def start_memory_manager():
    """Start the memory management system"""
    try:
        from research_synthesis.utils.memory_manager import start_memory_management
        start_memory_management()
        return {"message": "Memory management system started successfully"}
    except Exception as e:
        logger.error(f"Error starting memory manager: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")  
async def stop_memory_manager():
    """Stop the memory management system"""
    try:
        from research_synthesis.utils.memory_manager import stop_memory_management
        stop_memory_management()
        return {"message": "Memory management system stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping memory manager: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def memory_manager_config():
    """Get memory manager configuration"""
    try:
        from research_synthesis.utils.memory_manager import memory_manager
        return {
            "cleanup_interval_seconds": memory_manager.cleanup_interval,
            "monitoring_interval_seconds": memory_manager.monitoring_interval,
            "memory_warning_threshold_percent": memory_manager.memory_warning_threshold,
            "memory_critical_threshold_percent": memory_manager.memory_critical_threshold,
            "max_history_size": memory_manager.max_history_size,
            "is_running": memory_manager.is_running
        }
    except Exception as e:
        logger.error(f"Error getting memory manager config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/config")
async def update_memory_manager_config(
    cleanup_interval: int = None,
    monitoring_interval: int = None,
    memory_warning_threshold: float = None,
    memory_critical_threshold: float = None
):
    """Update memory manager configuration"""
    try:
        from research_synthesis.utils.memory_manager import memory_manager
        
        updated = {}
        
        if cleanup_interval is not None:
            memory_manager.cleanup_interval = max(60, cleanup_interval)  # Minimum 1 minute
            updated["cleanup_interval"] = memory_manager.cleanup_interval
            
        if monitoring_interval is not None:
            memory_manager.monitoring_interval = max(30, monitoring_interval)  # Minimum 30 seconds
            updated["monitoring_interval"] = memory_manager.monitoring_interval
            
        if memory_warning_threshold is not None:
            memory_manager.memory_warning_threshold = max(50.0, min(95.0, memory_warning_threshold))
            updated["memory_warning_threshold"] = memory_manager.memory_warning_threshold
            
        if memory_critical_threshold is not None:
            memory_manager.memory_critical_threshold = max(80.0, min(99.0, memory_critical_threshold))
            updated["memory_critical_threshold"] = memory_manager.memory_critical_threshold
            
        return {
            "message": "Memory manager configuration updated successfully",
            "updated_settings": updated,
            "note": "Changes will take effect on next monitoring cycle"
        }
        
    except Exception as e:
        logger.error(f"Error updating memory manager config: {e}")
        raise HTTPException(status_code=500, detail=str(e))