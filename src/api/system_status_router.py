"""
System Status API Router
Provides comprehensive system status endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime

router = APIRouter()

@router.get("/", response_model=Dict[str, Any])
async def get_system_status_api(force_refresh: Optional[bool] = False):
    """Get comprehensive system status"""
    try:
        from research_synthesis.utils.system_status import get_system_status
        status = await get_system_status(force_refresh=force_refresh)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@router.get("/component/{component_name}")
async def get_component_status_api(component_name: str):
    """Get status for a specific component"""
    try:
        from research_synthesis.utils.system_status import system_status_service
        component_status = await system_status_service.get_component_status(component_name)
        return component_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get component status: {str(e)}")

@router.get("/health")
async def system_health_check():
    """Simple health check endpoint"""
    try:
        from research_synthesis.utils.system_status import get_system_status
        status = await get_system_status()
        
        health_status = status.get("system_health", "unknown")
        
        if health_status == "critical":
            raise HTTPException(status_code=503, detail="System has critical issues")
        elif health_status == "unhealthy":
            raise HTTPException(status_code=503, detail="System is unhealthy")
        
        return {
            "status": "healthy",
            "health": health_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/databases")
async def get_database_status_api():
    """Get database status (backward compatibility endpoint)"""
    try:
        from research_synthesis.utils.system_status import get_database_status
        return await get_database_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database status: {str(e)}")

@router.get("/summary")
async def get_system_summary():
    """Get system status summary for dashboard widgets"""
    try:
        from research_synthesis.utils.system_status import get_system_status
        status = await get_system_status()
        
        summary = status.get("summary", {})
        components = status.get("components", {})
        
        # Count by status
        status_counts = {}
        for component_info in components.values():
            comp_status = component_info.get("status", "unknown")
            status_counts[comp_status] = status_counts.get(comp_status, 0) + 1
        
        return {
            "system_health": status.get("system_health"),
            "health_percentage": status.get("health_percentage"),
            "total_components": summary.get("total_components", 0),
            "connected_components": summary.get("connected", 0),
            "critical_issues": summary.get("critical_down", 0),
            "status_breakdown": status_counts,
            "recommendations": status.get("recommendations", []),
            "timestamp": status.get("timestamp")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system summary: {str(e)}")

@router.post("/refresh")
async def refresh_system_status():
    """Force refresh of system status"""
    try:
        from research_synthesis.utils.system_status import get_system_status
        status = await get_system_status(force_refresh=True)
        return {
            "message": "System status refreshed",
            "system_health": status.get("system_health"),
            "timestamp": status.get("timestamp")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh status: {str(e)}")